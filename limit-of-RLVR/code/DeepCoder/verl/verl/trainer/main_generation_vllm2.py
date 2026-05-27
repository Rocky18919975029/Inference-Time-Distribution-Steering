# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts using vLLM
"""
import csv
import numpy as np
import hydra
import os
from tabulate import tabulate
from vllm import LLM, SamplingParams
import pandas as pd
from transformers import AutoTokenizer

# VERL specific imports (minimal)
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.hdfs_io import makedirs
from verl.utils import hf_tokenizer # Assuming this provides a HuggingFace compatible tokenizer

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # Often recommended to be 'false' with vLLM or when using tokenizers in multiple processes/threads extensively
# os.environ['TORCH_COMPILE_DISABLE'] = '1'


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Check if output file already exists
    if os.path.exists(config.data.output_path):
        print(f"Output file {config.data.output_path} already exists. Skipping generation and proceeding to evaluation.")
        try:
            dataset = pd.read_parquet(config.data.output_path)
        except Exception as e:
            print(f"Failed to read parquet: {e}. Trying JSON.")
            try:
                import json as js_loader # alias to avoid conflict
                # Adjust path for JSON extension if needed by convention
                json_output_path = config.data.output_path.replace('.parquet', '.json')
                if not os.path.exists(json_output_path) and '.parquet' in config.data.output_path:
                     # if original path was .parquet and .json version doesn't exist, this try block is likely for a malformed parquet
                     # if original path could have been .json, then this would be fine
                     print(f"JSON path {json_output_path} also does not exist or original path was not parquet.")
                     raise FileNotFoundError("Original output path did not lead to a valid file and JSON alternative also not found.")

                with open(json_output_path, 'r') as f:
                    # Reading line-delimited JSON or a list of JSON objects
                    try:
                        dataset_json = js_loader.load(f)
                        dataset = pd.DataFrame(dataset_json)
                    except js_loader.JSONDecodeError:
                         f.seek(0) # Reset file pointer
                         dataset = pd.read_json(f, lines=True if isinstance(js_loader.load(f.readline()), dict) else False) # Basic guess for format
                config.data.output_path = json_output_path # Update path if JSON was loaded
            except Exception as e_json:
                print(f"Failed to read JSON: {e_json}. Trying Polars for Parquet.")
                try:
                    import polars as pl
                    # Ensure path is back to parquet if it was changed
                    parquet_output_path = config.data.output_path.replace('.json', '.parquet')
                    dataset = pl.read_parquet(parquet_output_path).to_pandas()
                    config.data.output_path = parquet_output_path # Update path if Polars parquet was loaded
                except Exception as e_polars:
                    print(f"Failed to read with Polars: {e_polars}. Exiting.")
                    raise Exception("Could not load existing output file.") from e_polars
    else:
        print("Output file not found. Starting generation process.")
        local_model_path = copy_local_path_from_hdfs(config.model.path)
        tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.get('trust_remote_code', True))
        # tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=config.model.get('trust_remote_code', True))


        if config.rollout.temperature == 0.:
            assert config.data.n_samples == 1

        # Read dataset
        try:
            dataset = pd.read_parquet(config.data.path)
            raw_prompts_chat_format = dataset[config.data.prompt_key].tolist()
        except Exception as e:
            print(f"Failed to read parquet dataset: {e}. Trying JSON.")
            try:
                import json as js_loader
                json_data_path = config.data.path.replace('.parquet', '.json')
                with open(json_data_path, 'r') as f:
                    try:
                        json_content = js_loader.load(f)
                        dataset = pd.DataFrame(json_content)
                    except js_loader.JSONDecodeError:
                        f.seek(0)
                        dataset = pd.read_json(f, lines=True)

                raw_prompts_chat_format = dataset[config.data.prompt_key].tolist()
                config.data.path = json_data_path # Update if JSON was loaded
            except Exception as e_json:
                print(f"Failed to read JSON dataset: {e_json}. Exiting.")
                raise

        # Initialize vLLM
        llm = LLM(
            model=local_model_path,
            tokenizer=local_model_path,
            tensor_parallel_size=config.rollout.tensor_model_parallel_size,
            dtype=config.rollout.dtype,
            trust_remote_code=config.model.get('trust_remote_code', True),
            gpu_memory_utilization=config.rollout.get('gpu_memory_utilization', 0.90),
            max_model_len=2048 + config.rollout.response_length,
            seed=config.rollout.seed,
            disable_custom_all_reduce=config.rollout.get('disable_custom_all_reduce', False),
            enforce_eager=config.rollout.get('enforce_eager', False),
            swap_space=96
        )

        stop_tokens = []
        if tokenizer and tokenizer.eos_token:
            stop_tokens.append(tokenizer.eos_token)
        if not stop_tokens:
            pass

        print("stop_tokens = ", stop_tokens)

        sampling_params = SamplingParams(
            n=config.data.n_samples,
            max_tokens=config.rollout.response_length,
            temperature=config.rollout.temperature,
            top_p=config.rollout.top_p,
            frequency_penalty=config.rollout.get('frequency_penalty', 0.0),
            presence_penalty=config.rollout.get('presence_penalty', 0.0),
            stop=stop_tokens if stop_tokens else [],
        )

        num_total_prompts = len(raw_prompts_chat_format)
        batch_size_from_config = config.data.batch_size
        num_batches = (num_total_prompts + batch_size_from_config - 1) // batch_size_from_config

        all_generated_responses_nested = []

        for batch_idx in range(num_batches):
            print(f'Processing batch [{batch_idx+1}/{num_batches}]')
            start_idx = batch_idx * batch_size_from_config
            end_idx = min((batch_idx + 1) * batch_size_from_config, num_total_prompts)
            
            current_batch_chat_formats = raw_prompts_chat_format[start_idx:end_idx]

            if not current_batch_chat_formats:
                continue
            
            try:
                prompt_strings_for_batch = tokenizer.apply_chat_template(
                    current_batch_chat_formats,
                    add_generation_prompt=True,
                    tokenize=False
                )
            except Exception as e:
                 print(f"Error applying chat template to batch: {e}. Trying one by one for this batch.")
                 prompt_strings_for_batch = [
                    tokenizer.apply_chat_template(
                        single_chat_format,
                        add_generation_prompt=True,
                        tokenize=False
                    ) for single_chat_format in current_batch_chat_formats
                ]


            if batch_idx == 0 and prompt_strings_for_batch:
                print(f"Sample formatted prompt for vLLM (first in batch {batch_idx+1}):")
                print(prompt_strings_for_batch[0])
            
            request_outputs = llm.generate(prompt_strings_for_batch, sampling_params)
            
            batch_responses_nested_for_this_batch = []
            for req_output in request_outputs:
                prompt_specific_responses = [comp_output.text for comp_output in req_output.outputs]
                batch_responses_nested_for_this_batch.append(prompt_specific_responses)
                # print(prompt_specific_responses)
            
            all_generated_responses_nested.extend(batch_responses_nested_for_this_batch)

        dataset['responses'] = all_generated_responses_nested

        output_dir = os.path.dirname(config.data.output_path)
        makedirs(output_dir, exist_ok=True)
        dataset.to_parquet(config.data.output_path)
        print(f"Generated responses saved to {config.data.output_path}")

    # --- Evaluation Part ---
    output_dir = os.path.dirname(config.data.output_path)
    
    if 'dataset' not in locals():
        try:
            dataset = pd.read_parquet(config.data.output_path)
        except Exception:
            print(f"Evaluation part: Failed to load {config.data.output_path}. Evaluation might fail.")
            raise
            
    prompts = dataset[config.data.prompt_key]
    responses = dataset['responses']
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    passes = 0
    total = len(dataset)
    total_scores = []
    
    for i in range(total):
        response_lst = responses[i]
        data_source = data_sources[i]
        reward_data = reward_model_data[i]
        
        if not isinstance(reward_data, dict) or 'ground_truth' not in reward_data:
            print(f"Warning: Skipping evaluation for index {i} due to missing or malformed reward_data: {reward_data}")
            total_scores.append([None] * len(response_lst if isinstance(response_lst, list) else [response_lst]))
            continue

        ground_truth = reward_data['ground_truth']
        reward_fn = select_reward_fn(data_source)
        
        score_lst = []
        for r_idx, r in enumerate(response_lst):
            try:
                score = reward_fn(r, ground_truth)
                score_lst.append(score)
            except Exception as e:
                try:
                    score = reward_fn(data_source, r, ground_truth)
                    score_lst.append(score)
                    print(f"Info: Used fallback reward_fn signature for index {i}, response {r_idx}. Error: {e}")
                except Exception as e_fallback:
                    print(f"Error computing reward for index {i}, response {r_idx}: {e_fallback}. Appending default/error score (e.g., 0 or None).")
                    score_lst.append(0.0)

        if not score_lst:
            max_score = 0.0
        else:
            valid_scores = [s for s in score_lst if s is not None]
            if not valid_scores:
                max_score = 0.0
            else:
                max_score = np.max(valid_scores)

        total_scores.append(score_lst)
        if max_score == 1:
            passes += 1

    n_samples = config.data.n_samples
    pass_at_n = passes / total if total > 0 else 0
    
    first_sample_scores = []
    for score_list_for_prompt in total_scores:
        if score_list_for_prompt and score_list_for_prompt[0] is not None:
            first_sample_scores.append(float(score_list_for_prompt[0] == 1))
    pass_at_1 = np.mean(first_sample_scores) if first_sample_scores else 0.0

    print(f"Calculated pass@{n_samples} (pass_at_n): {pass_at_n}")
    print(f"Calculated pass@1 (based on first sample success): {pass_at_1}")


    # Save metrics to CSV
    csv_path = os.path.join(output_dir, 'pass.csv')
    dataset_name = os.path.basename(config.data.path)
    row_data = {
        'model_path': config.model.path,
        'dataset': dataset_name,
        'pass@1': pass_at_1, # Using the refined pass@1 calculation
        f'pass@{n_samples}': pass_at_n
    }

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    table_data = [[k, v] for k, v in row_data.items()]
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))

    processed_total_scores = [[(1.0 if (s == 1 or s is True) else 0.0) if s is not None else 0.0 for s in score_list] for score_list in total_scores]
    
    results_path = os.path.join(output_dir, 'results.json')
    import json as js_dumper # alias
    with open(results_path, 'w') as f:
        js_dumper.dump(processed_total_scores, f, indent=4)
    print(f"Saved detailed scores to {results_path}")

def select_reward_fn(data_source):
    if data_source == 'lighteval/MATH':
        from verl.utils.reward_score import math
        return math.compute_score
    else:
        try:
            from rllm.rewards.rl_reward import rllm_reward_fn
            return rllm_reward_fn
        except ImportError:
            print(f"Warning: rllm_reward_fn not found for data_source '{data_source}'. Using a placeholder.")
            def placeholder_reward_fn(response, ground_truth, data_source_optional=None):
                print(f"Placeholder reward function called for: '{response[:50]}...' (source: {data_source_optional or data_source})")
                return 0.0
            return placeholder_reward_fn


if __name__ == '__main__':
    main()