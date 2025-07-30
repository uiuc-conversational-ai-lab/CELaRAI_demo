# =============================================================================
# chunking.py
# =============================================================================
"""
@module chunking
@author @sumukshashidhar

This module implements two modes of chunking for the YourBench pipeline:
1) "fast_chunking" (the default), which chunks by purely length-based rules.
2) "semantic_chunking" (requires explicit config), which uses sentence embeddings
   and a similarity threshold to decide chunk boundaries.

Usage:
------
Typically, you do not call this module directly. Instead, the handler.py
automatically invokes run(config) if the corresponding pipeline setting
(pipeline.chunking.run) is enabled.

The run(config) function:
1. Loads a dataset specified by the pipeline configuration.
2. Depending on the configured chunking mode:
   - fast_chunking (default): Chunks text solely based on maximum token length,
     ignoring sentence similarity.
   - semantic_chunking (requires pipeline.chunking.chunking_configuration.chunking_mode="semantic_chunking"):
     Splits each document into single-hop chunks, guided by user-defined token
     length constraints (l_min_tokens, l_max_tokens) and a similarity threshold (tau_threshold).
     Uses a transformer model specified in config['model_roles']['chunking'], or a default.
3. Creates multi-hop chunks by sampling subsets of single-hop chunks and concatenating them.
4. Computes optional readability and perplexity metrics for each chunk if debug mode is enabled
   and required packages (textstat, evaluate) are available.
5. Saves the dataset containing new columns:
   - "chunks" (list of single-hop segments)
   - "multihop_chunks" (list of multi-hop segment groups)
   - "chunk_info_metrics" (various statistics)
   - "chunking_model" (the model used for embeddings; default string if fast_chunking)

Error Handling and Logging:
---------------------------
- All warnings, errors, and debugging information are logged to both the console
  and a dedicated log file at logs/chunking.log.
- If any critical errors occur while loading or processing data, the process
  logs the exception and attempts a graceful exit without crashing the entire
  pipeline.

Debug Visualization:
--------------------
- In semantic_chunking mode, if debug mode is on, the module will generate a plot
  of average consecutive sentence similarities and save it to plots/aggregated_similarities.png.
"""

import os
import re
import time
from typing import Any, Dict, Optional
from dataclasses import asdict, dataclass

import numpy as np
from loguru import logger  # type: ignore
from tqdm.auto import tqdm

from yourbench.utils.chunking_utils import split_into_token_chunks
from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset


# Try importing torch-related libraries
_torch_available = False
try:
    import torch
    import torch.nn.functional as F
    from torch.amp import autocast

    _torch_available = True
    logger.info("PyTorch is available.")
except ImportError:
    logger.info("PyTorch is not available. Semantic chunking features requiring torch will be disabled.")

    # Define dummy autocast if torch not found
    class DummyAutocast:
        def __enter__(self):
            pass

        def __exit__(self, type, value, traceback):
            pass

    def autocast(device_type):
        return DummyAutocast()  # type: ignore


# Try importing transformers
_transformers_available = False
try:
    from transformers import AutoModel, AutoTokenizer

    _transformers_available = True
    logger.info("Transformers library is available.")
except ImportError:
    logger.info(
        "Transformers library is not available. Semantic chunking features requiring transformers will be disabled."
    )
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore


try:
    import evaluate

    # Attempt to load perplexity metric from evaluate
    _perplexity_metric = evaluate.load("perplexity", module_type="metric", model_id="gpt2")
    logger.info("Loaded 'perplexity' metric with model_id='gpt2'.")
except Exception as perplexity_load_error:
    logger.info(
        f"Could not load perplexity metric from 'evaluate'. Skipping perplexity. Error: {perplexity_load_error}"
    )
    _perplexity_metric = None

try:
    # Attempt to import textstat for readability metrics
    import textstat

    _use_textstat = True
except ImportError:
    logger.info("Package 'textstat' not installed. Readability metrics will be skipped.")
    _use_textstat = False


# -----------------------------------------------------------------------------
# Dataclasses for cleaner configuration and result handling
# -----------------------------------------------------------------------------
@dataclass
class ChunkingParameters:
    l_min_tokens: int = 64
    l_max_tokens: int = 128
    tau_threshold: float = 0.3
    h_min: int = 2
    h_max: int = 3
    num_multihops_factor: int = 2
    chunking_mode: str = "fast_chunking"  # "fast_chunking" or "semantic_chunking"


@dataclass
class SingleHopChunk:
    chunk_id: Any
    chunk_text: str


@dataclass
class MultiHopChunk:
    chunk_ids: list[str]
    chunks_text: list[str]


@dataclass
class ChunkInfoMetrics:
    token_count: float
    unique_token_ratio: float
    bigram_diversity: float
    perplexity: float
    avg_token_length: float
    flesch_reading_ease: float
    gunning_fog: float


def _parse_chunking_parameters(config: Dict[str, Any]) -> ChunkingParameters:
    """
    Extracts the chunking parameters from the config dictionary, falling back
    to default values if keys are missing. The chunking_mode defaults to
    "fast_chunking" unless explicitly set to "semantic_chunking."
    """
    chunking_params = config.get("pipeline", {}).get("chunking", {}).get("chunking_configuration", {})
    return ChunkingParameters(
        l_min_tokens=chunking_params.get("l_min_tokens", 128),
        l_max_tokens=chunking_params.get("l_max_tokens", 256),
        tau_threshold=chunking_params.get("tau_threshold", 0.7),
        h_min=chunking_params.get("h_min", 2),
        h_max=chunking_params.get("h_max", 5),
        num_multihops_factor=chunking_params.get("num_multihops_factor", 1),
        chunking_mode=chunking_params.get("chunking_mode", "fast_chunking"),
    )


def run(config: Dict[str, Any]) -> None:
    """
    Main pipeline entry point for the chunking stage.

    Args:
        config (Dict[str, Any]): The entire pipeline configuration dictionary.

    Returns:
        None. This function saves the updated dataset containing chunked
        documents to disk or the Hugging Face Hub, based on the config.

    Raises:
        RuntimeError: If a critical error is encountered that prevents chunking.
                      The error is logged, and execution attempts a graceful exit.
    """
    # Retrieve chunking configuration from config
    chunking_config = config.get("pipeline", {}).get("chunking", {})
    if chunking_config is None or not chunking_config.get("run", False):
        logger.info("Chunking stage is disabled. Skipping.")
        return

    logger.info("Starting chunking stage...")

    # Attempt to load dataset
    dataset = custom_load_dataset(config=config, subset="summarized")
    logger.info(f"Loaded summarized subset with {len(dataset)} rows for chunking.")

    # Retrieve chunking parameters into a dataclass
    params = _parse_chunking_parameters(config)
    l_min_tokens = params.l_min_tokens
    l_max_tokens = params.l_max_tokens
    tau_threshold = params.tau_threshold
    h_min = params.h_min
    h_max = params.h_max
    num_multihops_factor = params.num_multihops_factor
    chunking_mode = params.chunking_mode.lower().strip()

    # Check debug setting
    debug_mode: bool = config.get("settings", {}).get("debug", False)
    if debug_mode is False:
        # If not debug mode, skip perplexity and readability to save time
        logger.debug("Skipping perplexity and readability metrics (debug mode off).")
        local_perplexity_metric = None
        local_use_textstat = False
    else:
        local_perplexity_metric = _perplexity_metric
        local_use_textstat = _use_textstat

    # We'll only load the chunking model if in semantic_chunking mode
    tokenizer = None
    model = None
    device = "cpu"
    model_name = "no_model_for_fast_chunking"

    if chunking_mode == "semantic_chunking":
        # Check if required libraries are installed
        if not _torch_available or not _transformers_available:
            logger.error(
                "Semantic chunking requires 'torch' and 'transformers' libraries. "
                "Please install them (e.g., pip install yourbench[semantic]) or use 'fast_chunking' mode."
            )
            return  # Exit if dependencies are missing for semantic chunking

        try:
            # Extract model name from config if available
            model_name_list = config.get("model_roles", {}).get("chunking", [])
            if model_name_list is None or len(model_name_list) == 0:
                logger.info(
                    "No chunking model specified in config['model_roles']['chunking']. "
                    "Using default 'intfloat/multilingual-e5-large-instruct'."
                )
                model_name = "intfloat/multilingual-e5-large-instruct"
            else:
                model_name = model_name_list[0]

            logger.info(f"Using chunking model: '{model_name}'")
            # Determine device only if torch is available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore
            model = AutoModel.from_pretrained(model_name).to(device).eval()  # type: ignore
        except Exception as model_error:
            logger.error(f"Error loading tokenizer/model '{model_name}': {model_error}")
            logger.warning("Chunking stage cannot proceed with semantic_chunking. Exiting.")
            return
    else:
        logger.info("Using fast_chunking mode: purely length-based chunking with no embeddings.")

    # Prepare data structures
    all_single_hop_chunks: list[list[SingleHopChunk]] = []
    all_multihop_chunks: list[list[MultiHopChunk]] = []
    all_chunk_info_metrics: list[list[ChunkInfoMetrics]] = []
    all_similarities: list[list[float]] = []

    # Process each document in the dataset
    start_time = time.time()
    total_docs = len(dataset)
    logger.info(f"Starting chunking process for {total_docs} documents")

    for idx, row in enumerate(tqdm(dataset, desc="Chunking documents", ncols=100)):
        doc_start_time = time.time()
        logger.info(
            f"[{idx + 1}/{total_docs}] Processing document ID={row.get('document_id', f'doc_{idx}')} ({len(row.get('document_text', ''))} chars)"
        )
        doc_text = row.get("document_text", "")
        doc_id = row.get("document_id", f"doc_{idx}")
        logger.info(f"[{idx}] doc_id={row.get('document_id')} | text_len={len(doc_text)} | preview={doc_text[:100]!r}")

        # If text is empty or missing
        if doc_text is None or not doc_text.strip():
            logger.warning(f"Document at index {idx} has empty text. Storing empty chunks.")
            doc_process_time = time.time() - doc_start_time
            logger.info(f"Completed document {idx + 1}/{total_docs} in {doc_process_time:.2f}s")
            all_single_hop_chunks.append([])
            all_multihop_chunks.append([])
            all_chunk_info_metrics.append([])
            continue

        if (idx + 1) % 1 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_doc = elapsed_time / (idx + 1)
            remaining_docs = total_docs - (idx + 1)
            estimated_remaining = avg_time_per_doc * remaining_docs
            progress_pct = (idx + 1) / total_docs * 100

            logger.info(f"Progress: {progress_pct:.1f}% | Completed {idx + 1}/{total_docs} documents")
            logger.info(
                f"Avg time per doc: {avg_time_per_doc:.2f}s | Est. remaining: {estimated_remaining / 60:.1f} minutes"
            )

        # Split the document into sentences
        sentences = _split_into_sentences(doc_text)

        if sentences is None or len(sentences) == 0:
            logger.warning(f"No valid sentences found for doc at index {idx}, doc_id={doc_id}.")
            all_single_hop_chunks.append([])
            all_multihop_chunks.append([])
            all_chunk_info_metrics.append([])
            continue

        # Depending on the chunking mode:
        if chunking_mode == "semantic_chunking":
            # Debug log showing current dependency state
            logger.debug(
                f"Semantic chunking check: torch={_torch_available}, transformers={_transformers_available}, model_loaded={model is not None}, tokenizer_loaded={tokenizer is not None}"
            )

            # Ensure dependencies one last time before computation
            if not _torch_available or not _transformers_available or model is None or tokenizer is None:
                logger.error("Cannot perform semantic chunking due to missing dependencies or model loading issues.")
                # Add empty lists and continue to avoid crashing the loop for this document
                all_single_hop_chunks.append([])
                all_multihop_chunks.append([])
                all_chunk_info_metrics.append([])
                continue

            # 1) Compute embeddings for sentences
            sentence_embeddings = _compute_embeddings(tokenizer, model, texts=sentences, device=device, max_len=512)
            # 2) Compute consecutive sentence similarities
            consecutive_sims: list[float] = []
            for sentence_index in range(len(sentences) - 1):
                cos_sim = float(
                    F.cosine_similarity(
                        sentence_embeddings[sentence_index].unsqueeze(0),
                        sentence_embeddings[sentence_index + 1].unsqueeze(0),
                        dim=1,
                    )[0]
                )
                consecutive_sims.append(cos_sim)
            if consecutive_sims:
                all_similarities.append(consecutive_sims)

            # 3) Create single-hop chunks with semantic logic
            single_hop_chunks = _chunk_document_semantic(
                sentences=sentences,
                similarities=consecutive_sims,
                l_min_tokens=l_min_tokens,
                l_max_tokens=l_max_tokens,
                tau=tau_threshold,
                doc_id=doc_id,
            )
        else:
            # Debug line for fast chunking
            logger.info(
                f"[{doc_id}] Performing fast_chunking on {len(sentences)} sentences (l_max_tokens={l_max_tokens})"
            )

            # Fast chunking: purely length-based
            single_hop_chunks = _chunk_document_fast(
                sentences=sentences,
                l_max_tokens=l_max_tokens,
                doc_id=doc_id,
            )

        # Create multi-hop chunks
        multihop = _multihop_chunking(
            single_hop_chunks,
            h_min=h_min,
            h_max=h_max,
            num_multihops_factor=num_multihops_factor,
        )

        # Compute metrics (token_count, perplexity, readability, etc.)
        chunk_metrics = _compute_info_density_metrics(single_hop_chunks, local_perplexity_metric, local_use_textstat)

        # Accumulate
        all_single_hop_chunks.append(single_hop_chunks)
        all_multihop_chunks.append(multihop)
        all_chunk_info_metrics.append(chunk_metrics)

    # Optional: Save aggregated similarity plot only if in semantic_chunking and debug
    if chunking_mode == "semantic_chunking" and all_similarities and debug_mode:
        _plot_aggregated_similarities(all_similarities)

    # Convert dataclasses back to dicts for safe addition to the dataset
    dataset = dataset.add_column(
        "chunks",
        [[asdict(chunk) for chunk in chunk_list] for chunk_list in all_single_hop_chunks],
    )
    dataset = dataset.add_column(
        "multihop_chunks",
        [[asdict(mh) for mh in multihop_list] for multihop_list in all_multihop_chunks],
    )
    dataset = dataset.add_column(
        "chunk_info_metrics",
        [[asdict(cm) for cm in metric_list] for metric_list in all_chunk_info_metrics],
    )
    dataset = dataset.add_column("chunking_model", [model_name] * len(dataset))

    # Save updated dataset
    custom_save_dataset(dataset=dataset, config=config, subset="chunked")
    logger.success("Chunking stage completed successfully.")


def _split_into_sentences(text: str) -> list[str]:
    """
    Splits the input text into sentences using a simple rule-based approach
    that looks for punctuation delimiters ('.', '!', '?').

    Args:
        text (str): The full document text to be split.

    Returns:
        list[str]: A list of sentence strings.
    """
    # Replace newlines with spaces for consistency
    normalized_text = text.replace("\n", " ").strip()
    if normalized_text is None or normalized_text == "":
        return []

    # Split using capturing parentheses to retain delimiters, then recombine.
    segments = re.split(r"([.!?])", normalized_text)
    sentences: list[str] = []
    for i in range(0, len(segments), 2):
        if i + 1 < len(segments):
            # Combine the text and delimiter
            candidate = (segments[i] + segments[i + 1]).strip()
        else:
            # If no delimiter segment, use the text directly
            candidate = segments[i].strip()
        if candidate:
            sentences.append(candidate)
    return sentences


def _compute_embeddings(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    texts: list[str],
    device: "torch.device",
    max_len: int = 512,
    batch_size: int = 16,
) -> "list[torch.Tensor]":
    """
    Computes sentence embeddings by mean pooling the last hidden states,
    normalized to unit length.

    Args:
        tokenizer (AutoTokenizer): A Hugging Face tokenizer.
        model (AutoModel): A pretrained transformer model to generate embeddings.
        texts (list[str]): The list of sentence strings to be embedded.
        device (torch.device): The device on which to run inference (CPU or GPU).
        max_len (int): Max sequence length for tokenization.
        batch_size (int): Batch size.
    Returns:
        list[torch.Tensor]: A list of PyTorch tensors (one per sentence).
    """
    embeddings = []
    model.eval()

    # Determine autocast device type string
    autocast_device_type = "cuda" if _torch_available and torch.cuda.is_available() else "cpu"

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_dict = tokenizer(batch_texts, max_length=max_len, padding=True, truncation=True, return_tensors="pt").to(
            device
        )

        with torch.no_grad():
            # Use autocast context manager
            with autocast(autocast_device_type):
                outputs = model(**batch_dict)
                last_hidden_states = outputs.last_hidden_state
                attention_mask = batch_dict["attention_mask"]

                # Zero out non-attended tokens
                last_hidden_states = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

                # Mean pooling
                sum_hidden = last_hidden_states.sum(dim=1)
                valid_token_counts = attention_mask.sum(dim=1, keepdim=True)
                batch_embeddings = sum_hidden / valid_token_counts.clamp(min=1e-9)

                # Normalize
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

        embeddings.extend(batch_embeddings.cpu())

    return embeddings


def _chunk_document_semantic(
    sentences: list[str],
    similarities: list[float],
    l_min_tokens: int,
    l_max_tokens: int,
    tau: float,
    doc_id: str,
) -> list[SingleHopChunk]:
    """
    Creates single-hop chunks from sentences using semantic guidance. Ensures each
    chunk is at least l_min_tokens in length and at most l_max_tokens, introducing
    a chunk boundary when consecutive sentence similarity is below threshold tau.

    Args:
        sentences (list[str]): The list of sentences for a single document.
        similarities (list[float]): Cosine similarities between consecutive sentences.
        l_min_tokens (int): Minimum tokens per chunk.
        l_max_tokens (int): Maximum tokens per chunk.
        tau (float): Similarity threshold for introducing a chunk boundary.
        doc_id (str): Unique identifier for the document.

    Returns:
        list[SingleHopChunk]: A list of SingleHopChunk objects.
    """
    chunks: list[SingleHopChunk] = []
    current_chunk: list[str] = []
    current_len: int = 0
    chunk_index: int = 0

    for i, sentence in enumerate(sentences):
        sentence_token_count = len(sentence.split())

        # If one sentence alone exceeds l_max, finalize the current chunk if non-empty,
        # then store this sentence as its own chunk.
        if sentence_token_count >= l_max_tokens:
            # Dump the current chunk
            if len(current_chunk) > 0:
                chunk_str = " ".join(current_chunk)
                chunks.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=chunk_str))
                chunk_index += 1
                current_chunk = []
                current_len = 0
            # Store the sentence alone
            chunks.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=sentence))
            chunk_index += 1
            continue

        # Otherwise, add this sentence to the current chunk
        current_chunk.append(sentence)
        current_len += sentence_token_count

        # If we exceed l_max, close the current chunk and start a new one
        if current_len >= l_max_tokens:
            chunk_str = " ".join(current_chunk)
            chunks.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=chunk_str))
            chunk_index += 1
            current_chunk = []
            current_len = 0
            continue

        # If we have at least l_min tokens and the next sentence similarity is below threshold, break here
        if (current_len >= l_min_tokens) and (i < len(sentences) - 1):
            if similarities[i] < tau:
                chunk_str = " ".join(current_chunk)
                chunks.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=chunk_str))
                chunk_index += 1
                current_chunk = []
                current_len = 0

    # Any leftover
    if len(current_chunk) > 0:
        chunk_str = " ".join(current_chunk)
        chunks.append(SingleHopChunk(chunk_id=f"{doc_id}_{chunk_index}", chunk_text=chunk_str))

    return chunks


def _chunk_document_fast(
    sentences: list[str],
    l_max_tokens: int,
    doc_id: str,
    show_progress: bool = True,
) -> list[SingleHopChunk]:
    """
    Uses token-based chunking with optional overlap, based on tiktoken.

    Args:
        sentences (list[str]): Sentences of the document.
        l_max_tokens (int): Max tokens per chunk.
        doc_id (str): Unique identifier for the document.
        show_progress (bool): Show progress bar (ignored here, kept for API symmetry).

    Returns:
        list[SingleHopChunk]: A list of token-based chunks.
    """
    text = " ".join(sentences)
    chunk_texts = split_into_token_chunks(
        text,
        chunk_tokens=l_max_tokens,
        overlap=0,
    )

    return [SingleHopChunk(chunk_id=f"{doc_id}_{i}", chunk_text=chunk) for i, chunk in enumerate(chunk_texts)]


def _multihop_chunking(
    single_hop_chunks: list[SingleHopChunk],
    h_min: int,
    h_max: int,
    num_multihops_factor: int,
) -> list[MultiHopChunk]:
    """
    Creates multi-hop chunks via numpy random sampling.

    Generates combinations of size effective_h_max, slices them to sizes
    between h_min and effective_h_max, and collects unique combinations.
    Target number = max(1, total_single_hops // num_multihops_factor).
    Actual number may be less due to sampling/de-duplication.

    Args:
        single_hop_chunks: List of single-hop chunks.
        h_min: Min single-hops per multi-hop.
        h_max: Max single-hops per multi-hop.
        num_multihops_factor: Factor to determine target multi-hop count.

    Returns:
        List of unique MultiHopChunk objects.
    """
    total_single_hops = len(single_hop_chunks)
    logger.info(f"Starting multi-hop chunking, total single chunks: {total_single_hops}")

    if not single_hop_chunks:
        logger.warning("Empty input 'single_hop_chunks'. Returning [].")
        return []
    if not (0 < h_min <= h_max):
        logger.warning(f"Invalid hop range h_min={h_min}, h_max={h_max}. Returning [].")
        return []

    effective_h_max = min(h_max, total_single_hops)
    if h_min > effective_h_max:
        logger.warning(f"h_min ({h_min}) > effective_h_max ({effective_h_max}). Cannot form chunks. Returning [].")
        return []

    if num_multihops_factor <= 0:
        logger.info("num_multihops_factor <= 0. Targeting all single hops.")
        num_multihops_target = total_single_hops
    else:
        num_multihops_target = max(1, total_single_hops // num_multihops_factor)

    if np.prod((num_multihops_target, effective_h_max)) > total_single_hops:
        logger.warning(
            f"Target {num_multihops_target} is too high for given sample size: {total_single_hops} and effective_h_max: {effective_h_max}"
        )
        num_multihops_target = total_single_hops // effective_h_max

    logger.info(
        f"Targeting ~{num_multihops_target} multi-hop chunks, effective h_max: {effective_h_max}, h_min: {h_min}"
    )

    rng = np.random.default_rng()

    # Generate initial index combinations (size effective_h_max)
    initial_indices = rng.choice(
        total_single_hops,
        size=(num_multihops_target, effective_h_max),
        replace=False,  # Unique indices per combination
    )

    # Generate random slice sizes
    slice_sizes = rng.integers(low=h_min, high=effective_h_max, size=num_multihops_target, endpoint=True)

    # Slice, sort, tuple for hashing, and collect unique combinations
    unique_combo_indices_set = {
        tuple(np.sort(initial_indices[i][: slice_sizes[i]])) for i in range(num_multihops_target)
    }

    logger.info(f"Generated {len(unique_combo_indices_set)} unique index combinations.")

    if not unique_combo_indices_set:
        logger.warning("No unique combinations generated.")
        return []

    # --- Build MultiHopChunk Objects ---
    final_multihop_chunks = [
        MultiHopChunk(
            chunk_ids=[single_hop_chunks[idx].chunk_id for idx in combo_indices],
            chunks_text=[single_hop_chunks[idx].chunk_text for idx in combo_indices],
        )
        for combo_indices in unique_combo_indices_set
        # combo_indices guaranteed non-empty by h_min >= 1
    ]

    logger.info(f"Created {len(final_multihop_chunks)} multi-hop chunks.")
    return final_multihop_chunks


def _compute_info_density_metrics(
    chunks: list[SingleHopChunk],
    local_perplexity_metric: Optional[Any],
    local_use_textstat: bool,
) -> list[ChunkInfoMetrics]:
    """
    Computes optional statistics for each chunk, including token count, perplexity,
    readability (flesch, gunning fog), and basic lexical diversity metrics.

    Args:
        chunks (list[SingleHopChunk]): The list of single-hop chunk objects.
        local_perplexity_metric (Optional[Any]): If provided, used to compute
                                                 perplexity (from evaluate.load("perplexity")).
        local_use_textstat (bool): If True, compute text readability metrics using textstat.

    Returns:
        list[ChunkInfoMetrics]: One object per chunk with fields like:
          - token_count
          - unique_token_ratio
          - bigram_diversity
          - perplexity
          - avg_token_length
          - flesch_reading_ease
          - gunning_fog
    """
    results: list[ChunkInfoMetrics] = []

    for chunk in chunks:
        chunk_text: str = chunk.chunk_text
        tokens = chunk_text.strip().split()
        token_count: int = len(tokens)

        # Compute metrics step by step
        unique_token_ratio = 0.0
        if token_count > 0:
            unique_toks = len({t.lower() for t in tokens})
            unique_token_ratio = float(unique_toks / token_count)

        # Bigram diversity
        bigram_diversity = 0.0
        if token_count > 1:
            bigrams = []
            for i in range(token_count - 1):
                bigrams.append((tokens[i].lower(), tokens[i + 1].lower()))
            unique_bigrams = len(set(bigrams))
            bigram_diversity = float(unique_bigrams / len(bigrams))

        # Perplexity
        ppl_score: float = 0.0
        if local_perplexity_metric is not None and token_count > 0:
            try:
                result = local_perplexity_metric.compute(data=[chunk_text], batch_size=1)
                ppl_score = result.get("mean_perplexity", 0.0)
            except Exception as e:
                logger.warning(f"Could not compute perplexity for chunk. Error: {e}")

        # Average token length
        avg_token_length = 0.0
        if token_count > 0:
            avg_len = sum(len(t) for t in tokens) / token_count
            avg_token_length = float(avg_len)

        # Readability
        flesch_reading_ease = 0.0
        gunning_fog = 0.0
        if local_use_textstat is True and chunk_text.strip():
            try:
                flesch_reading_ease = float(textstat.flesch_reading_ease(chunk_text))
                gunning_fog = float(textstat.gunning_fog(chunk_text))
            except Exception as e:
                logger.warning(f"Textstat error: {e}")

        results.append(
            ChunkInfoMetrics(
                token_count=float(token_count),
                unique_token_ratio=unique_token_ratio,
                bigram_diversity=bigram_diversity,
                perplexity=ppl_score,
                avg_token_length=avg_token_length,
                flesch_reading_ease=flesch_reading_ease,
                gunning_fog=gunning_fog,
            )
        )

    return results


def _plot_aggregated_similarities(all_similarities: list[list[float]]) -> None:
    """
    Plots the average cosine similarity for each sentence-pair position across
    all documents, with shaded regions representing one standard deviation.

    Args:
        all_similarities (list[list[float]]): A list of lists, where each
            sub-list is the array of consecutive sentence similarities for
            a particular document.
    """
    if all_similarities is None or len(all_similarities) == 0:
        logger.debug("No similarities to plot. Skipping aggregated similarity plot.")
        return

    # Check if matplotlib is available before trying to plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib not found. Skipping similarity plot generation.")
        return

    plt.figure(figsize=(10, 6))
    max_len = max(len(sims) for sims in all_similarities)

    avg_sim: list[float] = []
    std_sim: list[float] = []
    counts: list[int] = []

    for position in range(max_len):
        vals = [s[position] for s in all_similarities if position < len(s)]
        if vals:
            mean_val = sum(vals) / len(vals)
            variance = sum((v - mean_val) ** 2 for v in vals) / len(vals)
            stddev_val = variance**0.5

            avg_sim.append(mean_val)
            std_sim.append(stddev_val)
            counts.append(len(vals))
        else:
            break

    # X-axis positions
    x_positions = list(range(len(avg_sim)))
    plt.plot(x_positions, avg_sim, "b-", label="Avg Similarity")

    # Create confidence interval region
    lower_bound = [max(0, a - s) for a, s in zip(avg_sim, std_sim)]
    upper_bound = [min(1, a + s) for a, s in zip(avg_sim, std_sim)]
    plt.fill_between(x_positions, lower_bound, upper_bound, alpha=0.3, color="blue")

    # Plot data points with size reflecting how many docs contributed
    max_count = max(counts) if counts else 1
    sizes = [30.0 * (c / max_count) for c in counts]
    plt.scatter(x_positions, avg_sim, s=sizes, alpha=0.5, color="navy")

    plt.title("Average Consecutive Sentence Similarity Across Documents")
    plt.xlabel("Sentence Pair Index")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plot_path: str = os.path.join("plots", "aggregated_similarities.png")
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # Changed dpi to 300
    plt.close()
    logger.info(f"Saved aggregated similarity plot at '{plot_path}'.")


# Make sure main guard exists if this file is runnable directly (optional but good practice)
if __name__ == "__main__":
    # Example configuration for testing (replace with actual loading if needed)
    test_config = {
        "pipeline": {
            "chunking": {
                "run": True,
                "chunking_configuration": {
                    "chunking_mode": "fast_chunking"  # or "semantic_chunking" if deps installed
                },
                # Add other necessary config keys like dataset paths etc.
            }
        },
        "settings": {"debug": True},
        # Add dataset config, model roles etc.
    }
    # Basic logger setup for standalone execution
    logger.add("logs/chunking_standalone.log", rotation="10 MB")
    logger.info("Running chunking module standalone (example)...")
    # Note: You'd need a valid dataset configuration for run() to work fully.
    # run(test_config)
    logger.info("Standalone example finished.")
