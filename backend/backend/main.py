import json
import os
import shutil
import threading

import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from flask import Flask, jsonify, request

# Load environment variables from .env file
load_dotenv()

process_lock = threading.Lock()

app = Flask(__name__)

# Manual CORS handling
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


def write_config(additional_instructions="Generate questions to test an undergraduate student"):
    # Write a yaml configuration file
    config = {
        "hf_configuration": {
            "token": os.getenv("HF_TOKEN", ""),
            "hf_organization": os.getenv("HF_ORGANIZATION", ""),
            "private": True,
            "hf_dataset_name": "celarai_demo",
            "concat_if_exist": False,
        },
        "model_list": [
            {
                "model_name": "qwen/qwq-32b",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "api_key": os.getenv("NVIDIA_API_KEY", ""),
                "max_concurrent_requests": 8,
            }
        ],
        "pipeline": {
            "ingestion": {
                "run": True,
                "source_documents_dir": f"{os.path.dirname(os.path.abspath(__file__))}/data/raw",
                "output_dir" : f"{os.path.dirname(os.path.abspath(__file__))}/data/processed",
            },
            "upload_ingest_to_hub": {
                "run": True,
                "source_documents_dir": f"{os.path.dirname(os.path.abspath(__file__))}/data/processed",
            },
            "summarization": {
                "run": True,
                "max_tokens": 16384,
                "token_overlap": 128,
                "encoding_name": "cl100k_base",
            },
            "chunking": {
                "run": True,
                "chunking_configuration": {
                    "chunking_mode": "fast_chunking",
                    "l_max_tokens": 64,
                    "token_overlap": 0,
                    "encoding_name": "cl100k_base"
                }
            },
            "single_shot_question_generation": {
                "run": True,
                "additional_instructions": additional_instructions,
            },
            "multi_hop_question_generation": {
                "run": True,
                "additional_instructions": additional_instructions,
            },
            "lighteval": {
                "run": True,
            }
        }
    }
    print(f"Writing config to {f'{os.path.dirname(os.path.abspath(__file__))}/config.yaml'}")
    with open(f'{os.path.dirname(os.path.abspath(__file__))}/config.yaml', "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print("Configuration file written successfully.")

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the question generation API!"})

@app.route('/process', methods=['POST'])
def process_file():
    # Check if files are present
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': 'No files selected'}), 400
    
    # Get the config JSON from the form data
    config_json = request.form.get('config')
    if not config_json:
        return jsonify({'error': 'No configuration provided'}), 400
    
    try:
        config = json.loads(config_json)
        print(f"Received config: {config}")
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid configuration JSON'}), 400
    
    # Extract configuration parameters
    question_types = config.get('questionTypes', [])
    difficulty = config.get('difficulty', 5)
    custom_instructions = config.get('customInstructions', 'Generate questions to test the corresponding student level')
    
    with process_lock:
        # Prepare data directory
        data_dir = f'{os.path.dirname(os.path.abspath(__file__))}/data/raw'
        # Clear the directory if it exists, else create it
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.makedirs(data_dir, exist_ok=True)

        # Clear processed data directory
        processed_data_dir = f'{os.path.dirname(os.path.abspath(__file__))}/data/processed'
        if os.path.exists(processed_data_dir):
            shutil.rmtree(processed_data_dir)
        os.makedirs(processed_data_dir, exist_ok=True)

        # Save all uploaded files to the directory
        saved_files = []
        for file in files:
            if file.filename != '':
                file_path = os.path.join(data_dir, file.filename)
                file.save(file_path)
                saved_files.append(file.filename)
                print(f"Saved file: {file.filename}")
        
        if not saved_files:
            return jsonify({'error': 'No valid files to process'}), 400

        # Write the configuration file with custom instructions
        if difficulty == 1:
            difficulty_description = "Please do not use difficult words or concepts. Make sure the difficulty is suitable for grade-2 student. Try to ask 4-or-5-word simple questions."
        else:
            difficulty_description = ""
        if difficulty_description:
            write_config(
                additional_instructions=f"Question types: {', '.join(question_types)}.\nDifficulty (out of 10; 1 = Elementary; 5 = Undergraduate; 10 = Expert): {difficulty}.\nDifficulty description: {difficulty_description}\nCustom instructions: {custom_instructions}"
            )
        else:
            write_config(
                additional_instructions=f"Question types: {', '.join(question_types)}.\nDifficulty (out of 10; 1 = Elementary; 5 = Undergraduate; 10 = Expert): {difficulty}.\nCustom instructions: {custom_instructions}"
            )

        try:
            # Run the processing script
            print("Starting question generation pipeline...")
            result = os.system(f'bash {os.path.dirname(os.path.abspath(__file__))}/run.sh')
            
            if result != 0:
                return jsonify({'error': 'Processing pipeline failed'}), 500
            
            # Load the output from huggingface
            dataset = load_dataset(f"{os.getenv('HF_ORGANIZATION', '')}/celarai_demo", name="lighteval", split="train")
            data = dataset.to_pandas().to_dict(orient="records")
            
            # Extract questions from the dataset
            questions = []
            answers = []
            for record in data:
                # Adjust this based on your actual dataset structure
                if 'question' in record:
                    questions.append(record['question'])
                elif 'query' in record:
                    questions.append(record['query'])
                else:
                    # Fallback: use the entire record as a question
                    questions.append(str(record))
                if 'answer' in record:
                    answers.append(record['answer'])
                elif 'ground_truth_answer' in record:
                    answers.append(record['ground_truth_answer'])
                else:
                    answers.append("No answer provided")
            
            response = {
                'success': True,
                'questions': questions,
                'answers': answers,
                'metadata': {
                    'files_processed': saved_files,
                    'question_types': question_types,
                    'difficulty': difficulty,
                    'total_questions': len(questions)
                }
            }
            
            print(f"Successfully generated {len(questions)} questions")
            return jsonify(response)
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return jsonify({
                'error': f'Processing failed: {str(e)}',
                'success': False
            }), 500
    

if __name__ == '__main__':
    app.run(debug=True , port=5001, host='0.0.0.0')