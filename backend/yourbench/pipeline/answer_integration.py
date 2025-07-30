# ============================================================
# answer_integration.py
# ============================================================

import random
import re
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset
from loguru import logger

from yourbench.utils.dataset_engine import custom_load_dataset, custom_save_dataset
from yourbench.utils.inference_engine import InferenceCall, run_inference

# Import the unified parsing function
from yourbench.utils.parsing_engine import parse_qa_pairs_from_response, shuffle_mcq
from yourbench.utils.prompts import (
    ANSWER_INTEGRATION_SYSTEM_PROMPT,
    ANSWER_INTEGRATION_USER_PROMPT,
)


@dataclass
class AnswerIntegrationRow:
    """
    Represents an answer integration row containing the necessary information

    Attributes:
        chunk_id: A string identifier for the chunk from which this question was generated.
        document_id: Identifier for the parent document.
        gem_id: Identifier for the specific question-answer pair.
        question: The generated question text.
        answer: The LLM-produced short answer or reasoning.
        raw_response: The full, unedited response from the model.
    """

    chunk_id: str
    chunk_text: str
    document_id: str
    gem_id: str
    document_text: str 
    document_filename: str
    document_summary: str
    question: str
    answer: str
    raw_response: str


@dataclass
class AnswerIntegrationConfig:
    run: bool = False
    source_subset: str = ""
    answer_file: str = ""
    output_subset: str = ""


@dataclass
class DocumentRow:
    document_summary: str = "No summary available."
    document_filename: str = ""
    document_id: str = ""
    document_text: str = ""
    chunks: list[dict[str, Any]] = field(default_factory=list)


def run(config: dict[str, Any]) -> None:
    """
    Executes the Single-Shot Question Generation stage of the pipeline.
    """
    stage_config = _load_stage_config(config)
    if not stage_config.run:
        logger.info("single_shot_question_generation stage is disabled. Skipping.")
        return

    dataset = custom_load_dataset(config=config, subset="chunked")
    answers = _load_answers(stage_config)
    logger.info(f"Loaded chunked subset with {len(dataset)} rows for Single-shot question generation.")

    inference_calls, call_index_mapping, chunk_mapping = _build_inference_calls(dataset, answers, stage_config)
    if not inference_calls:
        logger.warning("No inference calls were created for single_shot_question_generation.")
        return

    responses_dict = _execute_inference(inference_calls, config)
    if not responses_dict:
        return

    question_dataset = _process_responses_and_build_dataset(responses_dict, call_index_mapping, chunk_mapping, stage_config)
    if question_dataset is None or len(question_dataset) == 0:
        logger.warning("No valid questions produced in single_shot_question_generation.")
        return

    custom_save_dataset(dataset=question_dataset, config=config, subset="answer_integrated")
    logger.success("Answer integration completed successfully.")


def _load_stage_config(config: dict[str, Any]) -> AnswerIntegrationConfig:
    """
    Extract the stage-specific configuration from the pipeline config.
    """
    pipeline_config = config.get("pipeline", {})
    stage_config_dict = pipeline_config.get("answer_integration", {})

    return AnswerIntegrationConfig(
        run=stage_config_dict.get("run", False),
        source_subset=stage_config_dict.get("source_subset", ""),
        answer_file=stage_config_dict.get("answer_file", ""),
        output_subset=stage_config_dict.get("output_subset", ""),
    )


def _load_answers(config: AnswerIntegrationConfig) -> dict:
    """
    Load the answers from the specified file in the configuration.
    Returns a dictionary mapping document IDs to their answers.
    """
    if not config.answer_file:
        logger.warning("No answer file specified in the configuration.")
        return {}

    try:
        if config.answer_file.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(config.answer_file)
            df["story_name"] = df["story_name"] + ".md"
            answers = {}
            for _, row in df.iterrows():
                key = row["story_name"]
                qa_pair = {"question": row["question"], "answer": row["answer"], "gem_id": row["gem_id"]}
                answers.setdefault(key, []).append(qa_pair)
            return answers
        else:
            raise ValueError("Unsupported answer file format. Only CSV is currently supported.")
    except Exception as e:
        logger.error(f"Failed to load answers from {config.answer_file}: {e}")
        return {}


def _build_inference_calls(dataset, answers, stage_config: AnswerIntegrationConfig):
    """
    Create the InferenceCall objects needed for answer integration.
    Returns the list of calls and a parallel mapping of (row_index, doc_id, doc_filename, doc_text, doc_sum, question, answer, gem_id).
    """

    system_prompt = ANSWER_INTEGRATION_SYSTEM_PROMPT

    system_message = {"role": "system", "content": system_prompt}
    inference_calls = []
    call_index_mapping = []

    chunk_mapping = {}
    for row_index, row in enumerate(dataset):
        doc_row = DocumentRow(
            document_summary=row.get("document_summary", "No summary available."),
            document_filename=row.get("document_filename", f"Document_{row_index}"),
            document_id=row.get("document_id", f"doc_{row_index}"),
            document_text=row.get("document_text", ""),
            chunks=row.get("chunks", []),
        )

        chunks = ""
        for chunk in doc_row.chunks:
            chunk_id = chunk.get("chunk_id", "")
            chunk_text = chunk.get("chunk_text", "")
            chunks += f"\n\nchunk_id: {chunk_id}\nchunk_text: {chunk_text}"
            chunk_mapping[chunk_id] = chunk_text
        chunks = chunks.strip()

        # Build user messages for each answer
        for answer_info in answers.get(doc_row.document_filename, []):
            answer_text = answer_info.get("answer", "")
            question_text = answer_info.get("question", "")
            gem_id = answer_info.get("gem_id", "")

            user_prompt_str = ANSWER_INTEGRATION_USER_PROMPT.format(
                question=question_text,
                answer=answer_text,
                chunks=chunks,
            )
            user_message = {"role": "user", "content": user_prompt_str}

            inference_call = InferenceCall(messages=[system_message, user_message], tags=["answer_integration"])
            inference_calls.append(inference_call)
            call_index_mapping.append((row_index, doc_row.document_id, doc_row.document_filename, doc_row.document_text, doc_row.document_summary, question_text, answer_text, gem_id))

    return inference_calls, call_index_mapping, chunk_mapping


def _execute_inference(inference_calls, config: dict[str, Any]):
    """
    Sends the prepared inference calls to the LLM(s). Returns a dict of responses.
    """
    logger.info(f"Sending {len(inference_calls)} calls to inference for single-shot question generation.")
    try:
        return run_inference(
            config=config,
            step_name="answer_integration",
            inference_calls=inference_calls,
        )
    except Exception as err:
        logger.error(f"Inference failed for answer_integration: {err}")
        return {}


def _process_responses_and_build_dataset(
    responses_dict: dict[str, list[str]],
    call_index_mapping: list[tuple],
    chunk_mapping: dict[str, str],
    stage_config: AnswerIntegrationConfig,
) -> Dataset:
    """
    Take the LLM responses, parse them, and build a Hugging Face Dataset
    of single-shot question rows.
    """
    question_dataset_rows = []

    for model_name, model_responses in responses_dict.items():
        logger.info(f"Processing {len(model_responses)} responses from model: {model_name}")
        if len(model_responses) != len(call_index_mapping):
            logger.error(
                f"Model '{model_name}' returned {len(model_responses)} responses but expected {len(call_index_mapping)}. Mismatch."
            )

        for idx, raw_response in enumerate(model_responses):
            if idx >= len(call_index_mapping):
                break

            row_index, doc_id, doc_filename, doc_text, doc_sum, question, answer, gem_id = call_index_mapping[idx]
            chunk_id = _parse_chunk_id_from_response(raw_response)
            chunk_id = chunk_id.strip() if chunk_id else ""

            # If parsing fails or returns nothing, still create a fallback row
            if not chunk_id:
                logger.warning(
                    f"No parseable XML found (or empty list) for row_index={row_index}, gem_id={gem_id}, model={model_name}. Creating fallback row."
                )
                continue

            if chunk_id not in chunk_mapping:
                logger.warning(
                    f"Chunk ID '{chunk_id}' not found in chunk mapping for row_index={row_index}, gem_id={gem_id}, model={model_name}. Skipping."
                )
                continue

            # Otherwise, build final row
            question_row = AnswerIntegrationRow(
                chunk_id=chunk_id,
                chunk_text=chunk_mapping.get(chunk_id, ""),
                document_id=doc_id,
                gem_id=gem_id,
                document_text=doc_text,
                document_filename=doc_filename,
                document_summary=doc_sum,
                # Use the question and answer from the mapping
                question=question,
                answer=answer,
                raw_response=raw_response,
            )
            question_dataset_rows.append(question_row.__dict__)

    if not question_dataset_rows:
        return None

    logger.info(f"Constructing dataset with {len(question_dataset_rows)} answer integrated.")
    column_names = list(question_dataset_rows[0].keys())
    final_data = {column: [row[column] for row in question_dataset_rows] for column in column_names}
    return Dataset.from_dict(final_data)


def _parse_chunk_id_from_response(raw_response: str) -> str:
    """
    Parse the raw response from the model to extract question-answer pairs.
    Returns a list of dictionaries with 'question', 'answer', 'choices', etc.
    """
    try:
        fence_pattern = r"<selected_chunk_id>(.*?)</selected_chunk_id>"
        match = re.search(fence_pattern, raw_response, flags=re.DOTALL)
        if not match:
            logger.warning("No <selected_chunk_id> found in response; returning empty list.")
            return ""
        chunk_id = match.group(1).strip()
        return chunk_id
    except Exception as e:
        logger.error(f"Failed to parse response: {e}")
        return ""