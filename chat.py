from core.args import dataclass_from_dict
from core.transforms.video_transform import get_video_transform
from apps.plm.generate import PackedCausalTransformerGeneratorArgs, PackedCausalTransformerGenerator, load_consolidated_model_and_tokenizer

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

model_ckpt = "/projects/torresani-lab/ajay/perception_models/checkpoints/finetune_egoblind/checkpoints/0000000500"
# model_ckpt = "facebook/Perception-LM-8B"

# Load model and tokenizer
model, tokenizer, config = load_consolidated_model_and_tokenizer(model_ckpt)
transform = get_video_transform(image_res=model.vision_model.image_size)

# Create generator
gen_cfg = dataclass_from_dict(
    PackedCausalTransformerGeneratorArgs,
    {"temperature": 0.7, "top_p": 0.9, "top_k": None},
    strict=False,
)
generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

video_path = "/projects/torresani-lab/ajay/datasets/egoblind/train/video/00129.mp4"

max_frames = 32
start_time = None
end_time = None
prompt_type = "blind_aware"  # options: normal_vqa, blind_aware

system_prompt = get_system_prompt("blind_aware")

video_info = (video_path, max_frames, start_time, end_time, None) # path, max_frames, start, end, bbox_map
frames, _ = transform(video_info)

while True: 
    
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    
    prompt = system_prompt + user_input
    
    # Run generation
    answer, _, _ = generator.generate([(prompt, frames)])

    print("User: ", user_input)
    print("Assistant: ", answer[0])
    print("=" * 50)