from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import torch
import soundfile as sf

# 加载模型和处理器
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="auto"
)
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

# 构造对话，包括文本 prompt 和图片输入（本地路径或 URL 均可）
conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant that can understand images and answer questions."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "../img/BlueUp1.jpg"},  # 本地路径或 URL
            {"type": "text", "text": "Descirbe objects and their relative locations in details: "}
        ],
    },
]

# 如果不涉及音频，设为 False
USE_AUDIO_IN_VIDEO = False

# 准备推理输入
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(
    text=text,
    audio=audios,
    images=images,
    videos=videos,
    return_tensors="pt",
    padding=True,
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
)
inputs = inputs.to(model.device).to(model.dtype)

# 推理（仅生成文字，无需语音时可不保存 audio）
text_ids, _ = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
output_text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output_text)
