import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- 可选：强制启用 Hugging Face Hub 的并行下载器 ---
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # 启用并行加速器（需 pip install hf-transfer）
os.environ["HF_HUB_PROGRESS"] = "1"            # 强制显示下载进度条

# --- 设置模型名称 ---
model_name = "roberta-large-uncased"

print("正在加载 tokenizer 和模型（BERT），这应该显示进度条...")

# 下载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 下载模型
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

print("模型和 tokenizer 加载完毕。")

# --- 简单推理测试 ---
text = "This is a great movie!"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
pred = torch.argmax(logits, dim=-1).item()

print(f"文本分类结果（0或1）: {pred}")
