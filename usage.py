import json
import os.path as osp
from tqdm import tqdm
from time import time
from pathlib import Path

import numpy as np
import pandas as pd
from collections import defaultdict

from core.args import dataclass_from_dict
from core.transforms.video_transform import get_video_transform
from apps.plm.generate import PackedCausalTransformerGeneratorArgs, PackedCausalTransformerGenerator, load_consolidated_model_and_tokenizer

from dataset_config import EGO_BLIND_DATASET_CONFIG

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
        
        # The videos are stored in subdirectories named split_0, split_1, ..., split_9
        videos_dir = EGO_BLIND_DATASET_CONFIG['test_dir'] / "Videos"
        for i in range(10):
            split_path = videos_dir / f"split_{i}" / f"{video_name}.mp4"
            if split_path.exists():
                return str(split_path)
        
        # Possibly incorrect video name
        return None

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
            "start_time": meta_data["start_times"][idx],
            "end_time": meta_data["end_times"][idx],
            "timestamp": meta_data["end_times"][idx],
            "question_id": meta_data["question_ids"][idx],
            "question": meta_data["questions"][idx],
            "generated_answer": response,
            "ground_truth": meta_data["gt_answers"][idx],
        })
    return out


def generate_answer(
    input_dict: dict,
    number_of_frames: int=4,
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
        number_of_frames (int): Number of frames to sample from the video.
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
        video_info = (video_path, number_of_frames, start_time, end_time, None) # path, max_frames, start, end, bbox_map
        try:
            frames, _ = transform(video_info)
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
        "questions": questions,
        "generated_answers": generated_answers,
        "gt_answers": input_dict['gt_answers'],
        "start_times": start_times,
        "end_times": end_times,
    }
    json_output = create_json_response(generated_answers, meta_data)
    
    return json_output


if __name__ == "__main__":

    test_dir = EGO_BLIND_DATASET_CONFIG['test_dir']
    videos_dir = test_dir / "Videos"
    gt_annotations_path = test_dir / "test_half_release.csv"

    annotations = load_annotations(gt_annotations_path)

    model_ckpt = "facebook/Perception-LM-8B" 
    model, tokenizer, config = load_consolidated_model_and_tokenizer(model_ckpt)

    all_outputs = []
    for video_id in tqdm(sorted(annotations.keys()), desc="Generating predictions", total=len(annotations)):
        input_dict = annotations[video_id]
        out = generate_answer(input_dict, number_of_frames=32)
        print(out)
        break
