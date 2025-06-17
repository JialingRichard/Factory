from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import requests
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

model_id = "google/gemma-3-4b-pt"

# 下载图片并压缩到 224x224
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image = image.resize((224, 224))  # ✅ 压缩图像

# int4 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 加载模型（int4）
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
).eval()

# 加载 processor
processor = AutoProcessor.from_pretrained(model_id)

# 构造输入
prompt = "<start_of_image> who are you?"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
input_len = model_inputs["input_ids"].shape[-1]

# 生成文本
with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print("📝 Generated text:")
print(decoded)

# sleep(500)  # 等待500秒钟，确保输出可见
from time import sleep
sleep(500)  # 等待500秒钟，确保输出可见