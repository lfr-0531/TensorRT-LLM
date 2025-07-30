#!/usr/bin/env python3
"""
python longbench_rocket_eval.py --model_path "/home/scratch.trt_llm_data/llm-models/llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k"  --skillm_path ../skillm --output_dir results/vanilla --attention_backend VANILLA --backend pytorch --token_budget 2048 --run_all_tasks
python longbench_rocket_eval.py --model_path "/home/scratch.trt_llm_data/llm-models/llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k"  --skillm_path ../skillm --output_dir results/rocket_vanilla --attention_backend VANILLA --backend pytorch --token_budget 2048 --run_all_tasks --sparse_attn
python longbench_rocket_eval.py --model_path "/home/scratch.trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct"  --skillm_path ../skillm --output_dir results/rocket_vanilla_3.1 --attention_backend VANILLA --backend pytorch --token_budget 2048 --run_all_tasks --sparse_attn

python ../skillm/visualization/longbench_results_summary/long_bench_tasks_summary.py --output_dir /home/scratch.yuhangh_gpu_1/workspace/sparse_attn/trtllm_working/results/rocket_vanilla/longbench_Llama-3-8B-Instruct-Gradient-1048k_rocket_2048_20250728_114705

Usage:
    python longbench_rocket_eval.py --dataset narrativeqa --model_path /path/to/model --token_budget 512 --output_dir results/

    # Run all LongBench tasks
    python longbench_rocket_eval.py --run_all_tasks --model_path /path/to/model --token_budget 512 --output_dir results/
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__),
                             '../skillm'))  # skillm_path
from eval.longbench_utils.metrics import (classification_score, code_sim_score,
                                          count_score, qa_f1_score,
                                          qa_f1_zh_score, retrieval_score,
                                          retrieval_zh_score, rouge_score,
                                          rouge_zh_score)
from pipeline.model_utils import build_chat
from transformers import AutoTokenizer

# Add tensorrt_llm imports
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, RocketSparseAttentionConfig

# Add datasets import for loading skillm data
try:
    from datasets import load_from_disk
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# LongBench datasets as defined in skillm
LONGBENCH_DATASETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
    "samsum", "passage_retrieval_en", "passage_count", "lcc", "repobench-p"
]

# Chat templates mapping
CHAT_TEMPLATES = {
    "llama3.1-8b-instruct": "llama3",
    "llama3-8b-instruct": "llama3",
    "mistral-7b-instruct-v0.2": "mistral",
    "longchat-7b-v1.5-32k": "vicuna"
}


def setup_logging(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(
        output_dir,
        f"longbench_rocket_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ])

    logger = logging.getLogger("longbench_rocket_eval")
    logger.info(f"Logging to {log_file}")
    return logger


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LongBench evaluation with TensorRT-LLM and RocketKV")

    # Model and data arguments
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to model (HF model name or local path)')
    parser.add_argument('--dataset',
                        type=str,
                        choices=LONGBENCH_DATASETS,
                        help='LongBench dataset to evaluate on')
    parser.add_argument('--run_all_tasks',
                        action='store_true',
                        help='Run evaluation on all LongBench tasks')
    parser.add_argument(
        '--skillm_path',
        type=str,
        default='../skillm',
        help='Path to skillm directory for evaluation utilities')

    # Output arguments
    parser.add_argument('--output_dir',
                        type=str,
                        required=True,
                        help='Directory to save results')
    parser.add_argument('--exp_name',
                        type=str,
                        default=None,
                        help='Experiment name (auto-generated if not provided)')

    # Model configuration
    parser.add_argument('--attention_backend',
                        type=str,
                        default='VANILLA',
                        choices=['VANILLA', 'TRTLLM', 'FLASHINFER'],
                        help='Attention backend to use')
    parser.add_argument('--backend',
                        type=str,
                        default='pytorch',
                        choices=['pytorch', 'tensorrt'],
                        help='LLM backend to use')
    parser.add_argument('--chat_template',
                        type=str,
                        default='auto',
                        help='Chat template to use (auto-detect if "auto")')

    # Sequence and batch configuration
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=262144,
                        help='Maximum sequence length')
    parser.add_argument('--max_batch_size',
                        type=int,
                        default=1,
                        help='Maximum batch size')
    parser.add_argument('--max_new_tokens',
                        type=int,
                        default=256,
                        help='Maximum new tokens to generate')
    parser.add_argument(
        '--max_num_tokens',
        type=int,
        default=262144,
        help='Maximum total tokens across all sequences in a batch')
    parser.add_argument('--tensor_parallel_size',
                        type=int,
                        default=1,
                        help='Tensor parallel size')

    # RocketKV configuration
    parser.add_argument('--sparse_attn',
                        action='store_true',
                        help='Use sparse attention')
    parser.add_argument('--token_budget',
                        type=int,
                        default=2048,
                        help='Token budget for RocketKV (prompt_budget)')
    parser.add_argument('--window_size',
                        type=int,
                        default=32,
                        help='Window size for RocketKV')
    parser.add_argument('--kernel_size',
                        type=int,
                        default=63,
                        help='Kernel size for RocketKV')

    # KV cache configuration
    parser.add_argument('--kv_cache_dtype',
                        type=str,
                        default='auto',
                        help='KV cache data type')
    parser.add_argument('--kv_cache_fraction',
                        type=float,
                        default=0.7,
                        help='Fraction of GPU memory for KV cache')

    # Evaluation parameters
    parser.add_argument('--num_samples',
                        type=int,
                        default=None,
                        help='Number of samples to evaluate (None for all)')
    parser.add_argument('--start_idx',
                        type=int,
                        default=0,
                        help='Start index for evaluation')

    # System arguments
    parser.add_argument('--log_level',
                        type=str,
                        default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Validation
    if not args.run_all_tasks and not args.dataset:
        parser.error("Must specify either --dataset or --run_all_tasks")

    return args


def load_longbench_data(dataset: str, skillm_path: str) -> List[Dict[str, Any]]:
    """Load LongBench dataset using HuggingFace datasets format."""
    if not HAS_DATASETS:
        raise ImportError(
            "datasets library is required. Install with: pip install datasets")

    dataset_path = os.path.join(skillm_path, "dataset", "longbench", dataset)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    # Load the dataset using HuggingFace datasets
    dataset_obj = load_from_disk(dataset_path)

    # Convert to list of dictionaries
    data = []
    for item in dataset_obj:
        data.append({
            'input': item.get('input', ''),
            'context': item.get('context', ''),
            'answers': item.get('answers', []),
            'length': item.get('length', 0),
            'all_classes': item.get('all_classes', [])
        })

    return data


def load_evaluation_config(dataset: str, skillm_path: str) -> Dict[str, Any]:
    """Load evaluation configuration for a dataset."""
    config_path = os.path.join(skillm_path, "config", "eval_config",
                               "longbench", f"{dataset}.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Evaluation config not found: {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


def determine_chat_template(model_path: str, chat_template: str) -> str:
    """Determine chat template based on model path."""
    if chat_template != 'auto':
        return chat_template

    model_path_lower = model_path.lower()

    for model_key, template in CHAT_TEMPLATES.items():
        if model_key.replace('-', '').replace('.',
                                              '') in model_path_lower.replace(
                                                  '-', '').replace('.', ''):
            return template

    # Default fallback
    if 'llama' in model_path_lower:
        return 'llama3'
    elif 'mistral' in model_path_lower:
        return 'mistral'
    else:
        return 'none'  # No special formatting


def post_process(pred: str, chat_template: str, dataset: str) -> str:
    """Post-process prediction following skillm's approach."""
    pred = pred.split("</s")[0].strip()
    if chat_template == "qwen":
        pred = pred.split("<|im_end|>")[0]
    elif "llama2" in chat_template.lower():
        pred = (pred.split("(Document")[0].split("\n\nQuestion")[0].split(
            "\n\nAnswer")[0].split("[INST]")[0].split("[/INST]")[0].split(
                "(Passage")[0].strip())
    if dataset == "samsum":
        pred = pred.split("\n")[0].strip()

    return pred


def format_prompt_style(sample: Dict[str, Any], instruction: str,
                        chat_template: str, dataset: str, tokenizer) -> str:
    """Format prompt following skillm's approach exactly."""
    # First format the instruction using the sample data (like skillm does)
    prompt = instruction.format(**sample)

    if dataset not in [
            "trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"
    ]:
        prompt = build_chat(tokenizer, prompt, chat_template)

    return prompt


def initialize_llm(args: argparse.Namespace, logger: logging.Logger) -> LLM:
    """Initialize TensorRT-LLM with RocketKV configuration."""
    logger.info(f"Initializing LLM with model: {args.model_path}")
    logger.info(f"Attention backend: {args.attention_backend}")
    logger.info(f"Token budget: {args.token_budget}")

    # Configure KV cache
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,  # RocketKV doesn't support KV cache reuse
    )

    if args.sparse_attn:
        logger.info(
            f"RocketKV config - window_size: {args.window_size}, "
            f"kernel_size: {args.kernel_size}, prompt_budget: {args.token_budget}"
        )
        # Configure RocketKV sparse attention
        sparse_attention_config = RocketSparseAttentionConfig(
            window_size=args.window_size,
            kernel_size=args.kernel_size,
            prompt_budget=args.token_budget,
        )
    else:
        sparse_attention_config = None

    # Initialize LLM
    llm = LLM(
        model=args.model_path,
        backend=args.backend,
        kv_cache_config=kv_cache_config,
        attn_backend=args.attention_backend,
        sparse_attention_config=sparse_attention_config,
        tensor_parallel_size=args.tensor_parallel_size,
        max_seq_len=args.max_seq_len,
        max_num_tokens=args.max_num_tokens,
        cuda_graph_config=None,
        torch_compile_config=None,
    )

    logger.info("LLM initialized successfully")
    return llm


def run_evaluation(dataset: str, args: argparse.Namespace,
                   logger: logging.Logger) -> Tuple[List[Dict], float]:
    """Run evaluation on a single dataset following skillm's approach."""
    logger.info(f"Starting evaluation on dataset: {dataset}")

    # Load data and config
    data = load_longbench_data(dataset, args.skillm_path)
    eval_config = load_evaluation_config(dataset, args.skillm_path)

    # Filter data if needed
    if args.num_samples:
        end_idx = min(args.start_idx + args.num_samples, len(data))
        data = data[args.start_idx:end_idx]
        logger.info(
            f"Evaluating on {len(data)} samples (indices {args.start_idx}-{end_idx-1})"
        )
    else:
        data = data[args.start_idx:]
        logger.info(
            f"Evaluating on {len(data)} samples (from index {args.start_idx})")

    # Determine chat template
    chat_template = determine_chat_template(args.model_path, args.chat_template)
    logger.info(f"Using chat template: {chat_template}")

    # Initialize LLM
    llm = initialize_llm(args, logger)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Create sampling parameters with proper max_tokens
    max_new_tokens = eval_config['eval_params'].get('max_new_tokens',
                                                    args.max_new_tokens)

    # Prepare prompts following skillm's approach
    instruction = eval_config['eval_params']['instruction']
    prompts = []

    # Set up extra end token ids following skillm's approach (from infllm_utils.py)
    extra_end_token_ids = []
    if chat_template == "llama3":
        try:
            eot_id = tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]
            extra_end_token_ids.append(eot_id)
            logger.info(f"Added llama3 end token: {eot_id}")
        except Exception as e:
            logger.warning(f"Could not add llama3 end token: {e}")

    if chat_template == "qwen":
        try:
            im_end_id = tokenizer.encode("<|im_end|>",
                                         add_special_tokens=False)[0]
            extra_end_token_ids.append(im_end_id)
            logger.info(f"Added qwen end token: {im_end_id}")
        except Exception as e:
            logger.warning(f"Could not add qwen end token: {e}")

    if dataset == "samsum":
        try:
            newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
            extra_end_token_ids.append(newline_id)
            logger.info(f"Added samsum newline token: {newline_id}")
        except Exception as e:
            logger.warning(f"Could not add samsum newline token: {e}")

    if extra_end_token_ids:
        logger.info(f"Using extra end token IDs: {extra_end_token_ids}")

    for sample in data:
        formatted_prompt = format_prompt_style(sample, instruction,
                                               chat_template, dataset,
                                               tokenizer)
        prompts.append(formatted_prompt)

    if len(prompts) == 0:
        logger.warning("No prompts to evaluate")
        return [], 0

    # Run inference
    logger.info("Starting inference...")
    start_time = time.time()

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.8,
        top_p=0.95,
        stop_token_ids=extra_end_token_ids if extra_end_token_ids else None,
    )

    outputs = llm.generate(prompts, sampling_params)

    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.2f} seconds")
    logger.info(
        f"Average time per sample: {inference_time/len(prompts):.3f} seconds")

    # Prepare results with post-processing
    results = []
    for i, (sample, output) in enumerate(zip(data, outputs)):
        # Post-process the prediction following skillm's approach
        prediction = output.outputs[0].text.strip()
        processed_prediction = post_process(prediction, chat_template, dataset)

        result = {
            'sample_id': i + args.start_idx,
            'input': sample.get('input', ''),
            'context': sample.get('context', ''),
            'answers': sample.get('answers', []),
            'all_classes': sample.get('all_classes', []),
            'prediction': processed_prediction,
            'raw_prediction': prediction,  # Keep raw for debugging
            'prompt_length': len(output.prompt_token_ids),
            'output_length': len(output.outputs[0].token_ids),
            'inference_time': getattr(output, 'inference_time', None)
        }
        results.append(result)

    return results, inference_time


def calculate_metrics(dataset: str, predictions: List[str],
                      answers_list: List[List[str]],
                      all_classes_list: List[List[str]]) -> Dict[str, float]:
    """Calculate evaluation metrics for a dataset following skillm's implementation exactly."""

    # Mapping of datasets to their metric functions (exactly as in skillm)
    dataset2metric = {
        "narrativeqa": qa_f1_score,
        "qasper": qa_f1_score,
        "multifieldqa_en": qa_f1_score,
        "multifieldqa_zh": qa_f1_zh_score,
        "hotpotqa": qa_f1_score,
        "2wikimqa": qa_f1_score,
        "musique": qa_f1_score,
        "dureader": rouge_zh_score,
        "gov_report": rouge_score,
        "qmsum": rouge_score,
        "multi_news": rouge_score,
        "vcsum": rouge_zh_score,
        "trec": classification_score,
        "triviaqa": qa_f1_score,
        "samsum": rouge_score,
        "lsht": classification_score,
        "passage_retrieval_en": retrieval_score,
        "passage_count": count_score,
        "passage_retrieval_zh": retrieval_zh_score,
        "lcc": code_sim_score,
        "repobench-p": code_sim_score,
    }

    if dataset not in dataset2metric:
        # Fallback to simple exact match with cleaning
        total_score = 0
        for pred, answers in zip(predictions, answers_list):
            cleaned_pred = pred.lstrip('\n').split('\n')[0].strip()
            score = max([
                1.0 if cleaned_pred.lower() == ans.strip().lower() else 0.0
                for ans in answers
            ])
            total_score += score
        return {"exact_match": round(100 * total_score / len(predictions), 2)}

    metric_func = dataset2metric[dataset]
    total_score = 0.0
    raw_results = []

    # Follow skillm's scorer function exactly
    for pred, ground_truths, all_classes in zip(predictions, answers_list,
                                                all_classes_list):
        score = 0.0

        # Apply the same prediction cleaning as skillm
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            pred = pred.lstrip('\n').split('\n')[0]

        # For code datasets, apply additional cleaning
        if dataset in ["lcc", "repobench-p"]:
            # This cleaning is done inside code_sim_score, but let's also apply it here for consistency
            all_lines = pred.lstrip('\n').split('\n')
            for line in all_lines:
                if ('`' not in line) and ('#' not in line) and ('//'
                                                                not in line):
                    pred = line
                    break

        # Calculate max score across all reference answers (exactly as in skillm)
        for ground_truth in ground_truths:
            score = max(
                score, metric_func(pred, ground_truth, all_classes=all_classes))

        total_score += score
        raw_results.append({'answers': pred, 'score': score})

    final_score = round(100 * total_score / len(predictions), 2)
    return {metric_func.__name__: final_score}


def save_results(results: List[Dict], dataset: str, args: argparse.Namespace,
                 inference_time: float, output_dir: str,
                 logger: logging.Logger):
    """Save evaluation results in format compatible with skillm summary script."""
    os.makedirs(output_dir, exist_ok=True)

    # Extract predictions, answers, and all_classes for evaluation
    predictions = [r['prediction'] for r in results]
    answers_list = [r['answers'] for r in results]
    all_classes_list = [r.get('all_classes', []) for r in results]

    # Calculate metrics
    processed_results = calculate_metrics(dataset, predictions, answers_list,
                                          all_classes_list)
    logger.info(f"Evaluation metrics: {processed_results}")

    # Save detailed results for manual inspection
    results_file = os.path.join(output_dir, f"{dataset}_results.jsonl")
    with open(results_file, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

    # Save prediction results in skillm format for evaluation
    pred_dir = os.path.join(output_dir, "pred", args.attention_backend)
    os.makedirs(pred_dir, exist_ok=True)
    pred_file = os.path.join(pred_dir,
                             f"{dataset}_rocket_{args.token_budget}.jsonl")

    with open(pred_file, 'w', encoding='utf-8') as f:
        for result in results:
            pred_data = {
                "pred": result['prediction'],
                "answers": result['answers'],
                "all_classes": result.get('all_classes', []),
                "length": result.get('prompt_length', 0)
            }
            json.dump(pred_data, f, ensure_ascii=False)
            f.write('\n')

    # Create the config structure expected by skillm summary script
    config = {
        'pipeline_params': {
            'model_name': args.model_path,
            'method': args.attention_backend,
            'token_budget': args.token_budget,
            'max_seq_len': args.max_seq_len,
            'max_new_tokens': args.max_new_tokens,
            'window_size': args.window_size,
            'kernel_size': args.kernel_size
        },
        'eval_params': {
            'dataset': dataset,
            'num_samples': len(results)
        },
        'eval_results': {
            'processed_results': processed_results
        },
        'management': {
            'output_folder_dir': output_dir,
            'sub_dir': {
                'input_config': 'input_config/',
                'raw_results': 'raw_results.json',
                'result_vis': 'result_vis.png',
                'output_config': 'output_config.json'
            },
            'exp_desc':
            f'{dataset}_{os.path.basename(args.model_path)}_{args.attention_backend}_{args.token_budget}',
            'total_inference_time': inference_time,
            'avg_inference_time': inference_time / len(results),
            'evaluation_timestamp': datetime.now().isoformat()
        }
    }

    # Save raw results following skillm's format
    raw_results = []
    for i, (pred, ground_truths, all_classes) in enumerate(
            zip(predictions, answers_list, all_classes_list)):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            pred_cleaned = pred.lstrip('\n').split('\n')[0]
        else:
            pred_cleaned = pred

        # Get the individual score for this prediction
        list(processed_results.keys())[0]
        try:
            dataset2metric = {
                "narrativeqa": qa_f1_score,
                "qasper": qa_f1_score,
                "multifieldqa_en": qa_f1_score,
                "multifieldqa_zh": qa_f1_zh_score,
                "hotpotqa": qa_f1_score,
                "2wikimqa": qa_f1_score,
                "musique": qa_f1_score,
                "dureader": rouge_zh_score,
                "gov_report": rouge_score,
                "qmsum": rouge_score,
                "multi_news": rouge_score,
                "vcsum": rouge_zh_score,
                "trec": classification_score,
                "triviaqa": qa_f1_score,
                "samsum": rouge_score,
                "lsht": classification_score,
                "passage_retrieval_en": retrieval_score,
                "passage_count": count_score,
                "passage_retrieval_zh": retrieval_zh_score,
                "lcc": code_sim_score,
                "repobench-p": code_sim_score,
            }

            if dataset in dataset2metric:
                metric_func = dataset2metric[dataset]
                for ground_truth in ground_truths:
                    score = max(
                        score,
                        metric_func(pred_cleaned,
                                    ground_truth,
                                    all_classes=all_classes))
        except Exception:
            # Fallback scoring
            score = max([
                1.0 if pred_cleaned.lower() == ans.strip().lower() else 0.0
                for ans in ground_truths
            ])

        raw_results.append({
            'answers': ground_truths,
            'pred': pred_cleaned,
            'score': score
        })

    raw_results_file = os.path.join(output_dir, 'raw_results.json')
    with open(raw_results_file, 'w', encoding='utf-8') as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    # Save output_config.json in the format expected by skillm summary script
    output_config_file = os.path.join(output_dir, 'output_config.json')
    with open(output_config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Save summary for our own tracking
    summary_file = os.path.join(output_dir, f"{dataset}_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {results_file}")
    logger.info(
        f"SkillM-compatible output_config.json saved to {output_config_file}")
    logger.info(f"Summary saved to {summary_file}")
    logger.info(f"Prediction file saved to {pred_file}")


def main():
    """Main evaluation function."""
    args = parse_arguments()

    # Setup experiment name
    if not args.exp_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = os.path.basename(args.model_path).replace('/', '_')
        args.exp_name = f"longbench_{model_name}_rocket_{args.token_budget}_{timestamp}"

    # Setup output directory
    output_dir = os.path.join(args.output_dir, args.exp_name)
    logger = setup_logging(output_dir, args.log_level)

    logger.info("=" * 80)
    logger.info("LongBench Evaluation with TensorRT-LLM and RocketKV")
    logger.info("=" * 80)
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Output directory: {output_dir}")

    # Save configuration
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Configuration saved to {config_file}")

    # Determine datasets to evaluate
    if args.run_all_tasks:
        datasets = LONGBENCH_DATASETS
        logger.info(f"Running evaluation on {len(datasets)} LongBench datasets")
    else:
        datasets = [args.dataset]
        logger.info(f"Running evaluation on dataset: {args.dataset}")

    # Run evaluations
    all_results = {}
    total_start_time = time.time()

    for dataset in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating dataset: {dataset}")
        logger.info(f"{'='*60}")

        dataset_output_dir = os.path.join(output_dir, dataset)
        results, inference_time = run_evaluation(dataset, args, logger)
        if len(results) == 0:
            logger.warning(f"No results for dataset: {dataset}")
            continue
        save_results(results, dataset, args, inference_time, dataset_output_dir,
                     logger)
        all_results[dataset] = {
            'num_samples': len(results),
            'inference_time': inference_time,
            'output_dir': dataset_output_dir
        }
        logger.info(f"Dataset {dataset} completed successfully")

    total_time = time.time() - total_start_time

    # Save overall summary
    overall_summary = {
        'experiment_name':
        args.exp_name,
        'total_evaluation_time':
        total_time,
        'evaluated_datasets':
        list(all_results.keys()),
        'successful_datasets':
        [d for d, r in all_results.items() if 'error' not in r],
        'failed_datasets': [d for d, r in all_results.items() if 'error' in r],
        'results_by_dataset':
        all_results,
        'configuration':
        vars(args)
    }

    overall_summary_file = os.path.join(output_dir, "overall_summary.json")
    with open(overall_summary_file, 'w') as f:
        json.dump(overall_summary, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION COMPLETED")
    logger.info(f"{'='*80}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(
        f"Successful datasets: {len(overall_summary['successful_datasets'])}")
    logger.info(f"Failed datasets: {len(overall_summary['failed_datasets'])}")
    logger.info(f"Overall summary saved to: {overall_summary_file}")

    if overall_summary['failed_datasets']:
        logger.warning(f"Failed datasets: {overall_summary['failed_datasets']}")
        return 1

    return 0


if __name__ == '__main__':
    main()
