import json
import os.path as osp
from tqdm import tqdm
from time import time
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from collections import defaultdict

from core.args import dataclass_from_dict
from core.transforms.video_transform import get_video_transform
from apps.plm.generate import PackedCausalTransformerGeneratorArgs, PackedCausalTransformerGenerator, load_consolidated_model_and_tokenizer

from dataset_config import EGO_BLIND_DATASET_CONFIG

def store_egoblind_format(predictions_path: Path) -> dict:
    """
    Convert PLM model predictions to EgoBlind evaluation format.

    This script transforms PLM prediction JSON files into the JSONL format 
    required by EgoBlind evaluation scripts, extracting question IDs and 
    generated answers.

    The result file is created at the same location as the input file, with 
    "_egoblind_format.jsonl" appended to the original filename.
    """
 
    with open(predictions_path, 'r') as f:
        data = json.load(f)
    
    egoblind_formatted_data = []
    for pred in data: 
        egoblind_formatted_data.append({
            "question_id": pred["question_id"],
            "pred": pred["generated_answer"],
        })

    pred_out_path = predictions_path.parent / f"{predictions_path.stem}_egoblind_format.jsonl"
    with open(pred_out_path, 'w') as f:
        for item in egoblind_formatted_data:
            f.write(json.dumps(item) + "\n")

def load_annotations(anno_path: Path) -> list[dict]:
    """
    Load ground truth annotations from a CSV file.

    Returns:
        dict: A dictionary with video names as keys. Each video contains:
            - 'questions': List of questions for this video
            - 'answers': List of answer lists (each question has up to 4 answer variants)
            - 'timestamps': List of timestamps (one per question)

    Structure:
        {
            'video_idx_1': {
                'questions': ['What am I holding?', 'Where is the door?'],
                'answers': [
                    ['A cup', 'A white cup', 'Cup', None],  # 4 variants for Q1
                    ['On your left', 'To the left']          # 2 variants for Q2
                ],
                'timestamps': [5.0, 15.0]
            },
            'video_idx_2': {
                ...
            }
        }

    Notes:
        - Each question may have 1-4 answer variants (some may be None/null which are ignored)
        - timestamps align with questions (same length)
        - answers is a list of lists where answers[i] corresponds to questions[i]
    """

    def _get_video_path(video_name: str) -> str:
        return str(EGO_BLIND_DATASET_CONFIG['test_dir'] / "video" / f"{video_name}.mp4")

    gt_annotations = defaultdict(lambda: defaultdict(list))
    df = pd.read_csv(anno_path, header=0)

    for _, row in df.iterrows():
        video_name = str(f"{row["video_name"]:05d}")
        
        gt_annotations[video_name]['question_ids'].append(row["question_id"])
        gt_annotations[video_name]['questions'].append(str(row["question"]))
        
        answer_variants = []
        # There can be up to 4 answer variants
        for i in range(4): 
            if str(row[f"answer{i}"]) != 'nan':
                answer_variants.append(str(row[f"answer{i}"]))
        
        gt_annotations[video_name]['gt_answers'].append(answer_variants)
        gt_annotations[video_name]['start_times'].append(float(0)) # For the model, context is from start to the mentioned time-stamp
        gt_annotations[video_name]['end_times'].append(float(row["start-time/s"]))
        gt_annotations[video_name]['video_path'] = _get_video_path(video_name) # HACK: Sorry, but there should be a better way to do this :)

    return gt_annotations

def create_json_response(generated_answers: list[str], meta_data: dict) -> str: 
    out = []
    for idx, response in enumerate(generated_answers):
        out.append({
            "video_id": meta_data["video_id"],
            "video_path": meta_data["video_path"],
            "system_prompt": meta_data["system_prompt"],
            "start_time": meta_data["start_times"][idx],
            "end_time": meta_data["end_times"][idx],
            "timestamp": meta_data["end_times"][idx],
            "question_id": meta_data["question_ids"][idx],
            "question": meta_data["questions"][idx],
            "generated_answer": response,
            "ground_truth": meta_data["gt_answers"][idx],
        })
    return out

def get_system_prompt(type: str) -> str:
    if type == "normal_vqa": 
        return ""
    elif type == "blind_aware": 
        return (
                "I will provide you with a video each time and one question; "
                "These questions are all questions raised by the blind person in the video from his own first-person perspective in the current scene. "
                "Your task is to answer the blind person's question which was posed in the last frame of the video. "
                "The answer needs to be based on the content of the picture and the objective characteristics that the blind person cannot see. "
                "If the question cannot be answered, you can say I don't know. "
                "Do not include Chinese characters in your response. The question is: "
            )

def append_system_prompt(question: str, type: str) -> str:
    system_prompt = get_system_prompt(type)
    if system_prompt:
        return system_prompt + question
    else:
        return question

def generate_answer(
    input_dict: dict,
    prompt_type: str,
    max_frames: int=4,
    temperature: int=0.0,
    top_p: float=None,
    top_k: float=None,
) -> list[dict]:
    """
    Generate answers for the given video and questions using the Perception-LM model.
    Args:
        input_dict (dict): Dictionary containing:
            - 'video_path' (str): Path to the video file.
            - 'questions' (list[str]): List of questions to answer.
            - 'start_times' (list[float]): List of start times for context.
            - 'end_times' (list[float]): List of end times for context.
            - 'gt_answers' (list[list[str]]): List of ground truth answer variants for each question.
        prompt_type (str): Type of prompt to use ('normal_vqa' or 'blind_aware').
        max_frames (int): Number of frames to sample from the video.
        temperature (int): Sampling temperature for generation.
        top_p (float): Top-p sampling parameter.
        top_k (float): Top-k sampling parameter.    
    Returns:
        list[dict]: List of dictionaries containing generated answers and metadata.
    """
    
    # Input prep
    video_path = input_dict['video_path']
    questions = input_dict['questions']
    start_times = input_dict['start_times']
    end_times = input_dict['end_times']
    
    assert len(questions) == len(start_times) == len(end_times), "Length of questions, start_times, and end_times must be the same."
    
    # Load model and tokenizer
    transform = get_video_transform(image_res=model.vision_model.image_size)
    
    # Create generator
    gen_cfg = dataclass_from_dict(
        PackedCausalTransformerGeneratorArgs,
        {"temperature": temperature, "top_p": top_p, "top_k": top_k},
        strict=False,
    )
    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

    # Generate answers for each question
    generated_answers = []
    for question, start_time, end_time in zip(questions, start_times, end_times):
        prompts = []
        video_info = (video_path, max_frames, start_time, end_time, None) # path, max_frames, start, end, bbox_map
        try:
            frames, _ = transform(video_info)

            # Append system prompt based on prompt type
            question = append_system_prompt(question, prompt_type)
            prompts.append((question, frames))

            # Run generation
            answer, _, _ = generator.generate(prompts)
            generated_answers.append(answer[0])  # answer is a list of generated answers
        
        except RuntimeError as e:
            if "Invalid data found when processing input" in str(e):
                print(f"Skipping corrupted video: {video_info[0]}")
                return None  # or appropriate default value

    # Prepare JSON output for all questions for this one video
    meta_data = {
        "video_id": osp.basename(video_path).split('.')[0],
        "video_path": video_path,
        "question_ids": input_dict['question_ids'],
        "system_prompt": get_system_prompt(prompt_type),
        "questions": questions,
        "generated_answers": generated_answers,
        "gt_answers": input_dict['gt_answers'],
        "start_times": start_times,
        "end_times": end_times,
    }
    json_output = create_json_response(generated_answers, meta_data)
    
    return json_output


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate predictions for EgoBlind Test Release set using Perception-LM.")
    parser.add_argument("--model_ckpt", type=str, required=True, default=None, help="Path to the model checkpoint.")
    parser.add_argument("--experiment_name", type=str, default="egoblind_test_release_generation", help="Name of the experiment.")
    parser.add_argument("--prompt_type", type=str, choices=["normal_vqa", "blind_aware"], required=True, help="Type of prompt to use for generation.")
    parser.add_argument("--egoblind_format", action='store_true', help="If set, convert output to EgoBlind format after generation.")
    args = parser.parse_args()

    test_dir = EGO_BLIND_DATASET_CONFIG['test_dir']
    videos_dir = test_dir / "video"
    gt_annotations_path = test_dir / "test_half_release.csv"
    
    predictions_output_dir = Path("/projects/torresani-lab/ajay/perception_models/predictions")
    predictions_output_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = Path("/projects/torresani-lab/ajay/perception_models/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_str = "Predictions for EgoBlind Test Release set using Perception-LM\n"
    summary_str += "-------------------------------------------------------\n\n"
    summary_str += f"Experiment Name: {args.experiment_name}\n"
    summary_str += f"Videos dir: {videos_dir}\n"
    summary_str += f"System Prompt Type: {args.prompt_type}\n"
    summary_str += f"System Prompt: {get_system_prompt(args.prompt_type)}\n"
    summary_str += f"GT Annotations path: {gt_annotations_path}\n"

    annotations = load_annotations(gt_annotations_path)
    summary_str += f"Number of videos: {len(annotations)}\n"

    model_ckpt = args.model_ckpt
    
    max_frames = 32
    summary_str += f"Using model checkpoint: {model_ckpt}\n"
    summary_str += f"Max frames per video: {max_frames}\n"
    model, tokenizer, config = load_consolidated_model_and_tokenizer(model_ckpt)

    all_outputs = []
    inference_times = []
    for video_id in tqdm(sorted(annotations.keys()), desc="Generating predictions", total=len(annotations)):
        input_dict = annotations[video_id]

        start = time()
        out = generate_answer(input_dict, prompt_type=args.prompt_type, max_frames=max_frames)
        if out is None:
            continue  # Skip corrupted video
        inference_times.append(time() - start)
        
        all_outputs.extend(out)
    
    # Save predictions with versioning
    pred_out_file = predictions_output_dir / f"{args.experiment_name}.json"
    if pred_out_file.exists():
        existing = list(predictions_output_dir.glob(f"{args.experiment_name}*.json"))
        idx = len(existing)
        pred_out_file = predictions_output_dir / f"{args.experiment_name}_v{idx}.json"
    
    summary_str += f"Saving predictions to: {pred_out_file}\n"
    # Save all outputs to JSON file
    with open(pred_out_file, "w") as f:
        json.dump(all_outputs, f, indent=4)
    # If requested, convert to EgoBlind format
    if args.egoblind_format:
        summary_str += "Converting predictions to EgoBlind format...\n"
        store_egoblind_format(pred_out_file)
        summary_str += "Conversion completed.\n"

    # Calculate inference statistics
    inference_times = np.array(inference_times)
    total_time = np.sum(inference_times) / 3600.0  # in hours
    avg_time = np.mean(inference_times)
    median_time = np.median(inference_times)

    summary_str += f"\n----------------Inference time statistics:-------------\n"
    summary_str += f"Total inference time: {total_time:.2f} hours\n"
    summary_str += f"Average inference time per video: {avg_time:.2f} seconds\n"
    summary_str += f"Median inference time per video: {median_time:.2f} seconds\n"
    summary_str += f"-------------------------------------------------------\n"

    summary_str += "Done generating predictions.\n"
    
    # Save summary with versioning
    summary_file = logs_dir / f"{args.experiment_name}.txt"
    if summary_file.exists():
        existing = list(logs_dir.glob(f"{args.experiment_name}*.txt"))
        summary_file = logs_dir / f"{args.experiment_name}_v{len(existing)}.txt"
    
    summary_str += f"Saving summary to: {summary_file}\n"
    with open(summary_file, "w") as f:
        f.write(summary_str)

    print("\nDone.", flush=True)