# import torch
# import time
# import sys
# import importlib.util
# import soundfile as sf

# from awq.models.base import BaseAWQForCausalLM
# from transformers import AutoProcessor
# from transformers import Qwen2_5OmniProcessor
# from qwen_omni_utils import process_mm_info
# from huggingface_hub import hf_hub_download

# from modeling_qwen2_5_omni_low_VRAM_mode import (
#     Qwen2_5OmniDecoderLayer
# )
# from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration

# def replace_transformers_module():
#     original_mod_name = 'transformers.models.qwen2_5_omni.modeling_qwen2_5_omni'
    
#     new_mod_path = 'modeling_qwen2_5_omni_low_VRAM_mode.py'

#     if original_mod_name in sys.modules:
#         del sys.modules[original_mod_name]

#     spec = importlib.util.spec_from_file_location(original_mod_name, new_mod_path)
#     new_mod = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(new_mod)

#     sys.modules[original_mod_name] = new_mod

# replace_transformers_module()

# class Qwen2_5_OmniAWQForConditionalGeneration(BaseAWQForCausalLM):
#     layer_type = "Qwen2_5OmniDecoderLayer"
#     max_seq_len_key = "max_position_embeddings"
#     modules_to_not_convert = ["visual"]
#     @staticmethod
#     def get_model_layers(model: "Qwen2_5OmniForConditionalGeneration"):
#         return model.thinker.model.layers

#     @staticmethod
#     def get_act_for_scaling(module: "Qwen2_5OmniForConditionalGeneration"):
#         return dict(is_scalable=False)

#     @staticmethod
#     def move_embed(model: "Qwen2_5OmniForConditionalGeneration", device: str):
#         model.thinker.model.embed_tokens = model.thinker.model.embed_tokens.to(device)
#         model.thinker.visual = model.thinker.visual.to(device)
#         model.thinker.audio_tower = model.thinker.audio_tower.to(device)
        
#         model.thinker.visual.rotary_pos_emb = model.thinker.visual.rotary_pos_emb.to(device)
#         model.thinker.model.rotary_emb = model.thinker.model.rotary_emb.to(device)
        
#         for layer in model.thinker.model.layers:
#             layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)
        
#     @staticmethod
#     def get_layers_for_scaling(
#         module: "Qwen2_5OmniDecoderLayer", input_feat, module_kwargs
#     ):
#         layers = []

#         # attention input
#         layers.append(
#             dict(
#                 prev_op=module.input_layernorm,
#                 layers=[
#                     module.self_attn.q_proj,
#                     module.self_attn.k_proj,
#                     module.self_attn.v_proj,
#                 ],
#                 inp=input_feat["self_attn.q_proj"],
#                 module2inspect=module.self_attn,
#                 kwargs=module_kwargs,
#             )
#         )

#         # attention out
#         # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
#         if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
#             layers.append(
#                 dict(
#                     prev_op=module.self_attn.v_proj,
#                     layers=[module.self_attn.o_proj],
#                     inp=input_feat["self_attn.o_proj"],
#                 )
#             )

#         # linear 1
#         layers.append(
#             dict(
#                 prev_op=module.post_attention_layernorm,
#                 layers=[module.mlp.gate_proj, module.mlp.up_proj],
#                 inp=input_feat["mlp.gate_proj"],
#                 module2inspect=module.mlp,
#             )
#         )

#         # linear 2
#         layers.append(
#             dict(
#                 prev_op=module.mlp.up_proj,
#                 layers=[module.mlp.down_proj],
#                 inp=input_feat["mlp.down_proj"],
#             )
#         )

#         return layers

# device_map = {
#     "thinker.model": "cuda", 
#     "thinker.lm_head": "cuda", 
#     "thinker.visual": "cuda",  
#     "thinker.audio_tower": "cpu",  
#     "talker": "cpu",  
#     "token2wav": "cpu",  
# }
# device = 'cuda'

# model_path = "Qwen/Qwen2.5-Omni-7B-AWQ"

# model = Qwen2_5_OmniAWQForConditionalGeneration.from_quantized(  
#                                             model_path, 
#                                             model_type="qwen2_5_omni",
#                                             device_map=device_map, 
#                                             torch_dtype=torch.float16,   
#                                             attn_implementation="flash_attention_2"
#                                         )

# # spk_path = model_path + '/spk_dict.pt' # use this line if you load model from local
# spk_path = hf_hub_download(repo_id=model_path, filename='spk_dict.pt')

# model.model.load_speakers(spk_path)

# # model.model.thinker.model.embed_tokens = model.model.thinker.model.embed_tokens.to(device)
# # model.model.thinker.visual = model.model.thinker.visual.to(device)
# # model.model.thinker.audio_tower = model.model.thinker.audio_tower.to(device)
# # model.model.thinker.visual.rotary_pos_emb = model.model.thinker.visual.rotary_pos_emb.to(device)
# # model.model.thinker.model.rotary_emb = model.model.thinker.model.rotary_emb.to(device)

# # for layer in model.model.thinker.model.layers:
# #     layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)


# processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

# # def video_inference(video_path, prompt, sys_prompt):
# #     messages = [
# #         {"role": "system", "content": [
# #                 {"type": "text", "text": sys_prompt},
# #             ]},
# #         {"role": "user", "content": [
# #                 {"type": "video", "video": video_path},
# #             ]
# #         },
# #     ]
# #     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# #     audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
# #     inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
# #     inputs = inputs.to('cuda')
    

# #     output = model.generate(**inputs, use_audio_in_video=True, return_audio=True)
# #     text = processor.batch_decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
# #     audio = output[2]
# #     return text, audio


# # video_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"
# # system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

# # torch.cuda.reset_peak_memory_stats()
# # start = time.time()
# # response, audio  = video_inference(video_path, prompt=None, sys_prompt=system_prompt)
# # end = time.time()
# # peak_memory = torch.cuda.max_memory_allocated()

# # audio_file_path = "./output_audio_awq.wav"
# # sf.write(
# #     audio_file_path,
# #     audio.reshape(-1).detach().cpu().numpy(),
# #     samplerate=24000,
# # )

# # print(response[0])
# # print(f"Total Inference Time: {end-start:.2f} s.")
# # print(f"Peak GPU Memory Used: {peak_memory / 1024 / 1024:.2f} MB")

# # model.disable_talker()
# from PIL import Image
# img_path = "../../img/BlueUp1.jpg"
# image = Image.open(img_path).convert("RGB")
# image = image.resize((224, 224))  # 或者 (384, 384)
# image.save("resized.jpg")

# conversation = [
#     {
#         "role": "system",
#         "content": [{"type": "text", "text": "You are a helpful assistant that can understand images and answer questions."}],
#     },
#     {
#         "role": "user",
       
# "content": [
#             {"type": "image", "image": "resized.jpg"},
#             {"type": "text", "text": "Descirbe objects and their relative locations in details: "}
#         ],
#     },
# ]


# # 如果不涉及音频，设为 False
# USE_AUDIO_IN_VIDEO = False

# # 准备推理输入
# text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
# audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
# inputs = processor(
#     text=text,
#     audio=audios,
#     images=images,
#     videos=videos,
#     return_tensors="pt",
#     padding=True,
#     use_audio_in_video=USE_AUDIO_IN_VIDEO,
# )

# device = next(model.parameters()).device
# dtype = next(model.parameters()).dtype

# inputs = inputs.to(device=device, dtype=dtype)

# # inputs = inputs.to(model.device).to(model.dtype)

# # 推理（仅生成文字，无需语音时可不保存 audio）
# # text_ids, _ = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

# text_ids = model.generate(
#     **inputs,
#     use_audio_in_video=False,           # 不启用音频
#     return_audio=False, # 不返回音频
#     max_new_tokens=20,                  # 限制输出长度
#     do_sample=True,                     # 采样生成（非贪心）
#     temperature=0.7,                    # 控制多样性
#     top_p=0.9,                          # nucleus sampling
#     repetition_penalty=1.1             # 减少重复输出
# )


# output_text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print(output_text)



import torch
import os
from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor

# 自定义替换原始 Qwen 模型实现为低 VRAM 版本
# 确保你有 modeling_qwen2_5_omni_low_VRAM_mode.py 在当前目录下
from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration

# 替换 transformers 中的原始模块为自定义 low VRAM 实现
import sys
import importlib.util

def replace_transformers_module():
    original_mod_name = 'transformers.models.qwen2_5_omni.modeling_qwen2_5_omni'
    new_mod_path = str(Path(__file__).parent / "modeling_qwen2_5_omni_low_VRAM_mode.py")

    if original_mod_name in sys.modules:
        del sys.modules[original_mod_name]

    spec = importlib.util.spec_from_file_location(original_mod_name, new_mod_path)
    new_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(new_mod)

    sys.modules[original_mod_name] = new_mod

replace_transformers_module()

# 自定义 device_map（显存优化）
device_map = {
    "thinker.model": "cuda",
    "thinker.lm_head": "cuda",
    "thinker.visual": "cuda",     # 图像编码器放 GPU
    "thinker.audio_tower": "cpu", # 音频编码器放 CPU
    "talker": None,               # 屏蔽 talker（TTS 模块）
    "token2wav": None             # 屏蔽语音合成模块
}

# 加载 AWQ 量化模型（INT4）
model_id = "Qwen/Qwen2.5-Omni-7B-AWQ"

# 初始化处理器
processor = AutoProcessor.from_pretrained(model_id)

# 加载模型并屏蔽 talker 和 token2wav 模块
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_id,
    device_map=device_map,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)

# 强制移除 talker 和 token2wav 模块引用（防止内部调用 .to(device)）
model.talker = None
model.token2wav = None

# 加载 speaker 字典（如果存在，否则跳过）
try:
    spk_dict_path = hf_hub_download(repo_id=model_id, filename="spk_dict.pt")
    model.load_speakers(spk_dict_path)
except Exception as e:
    print("No speaker dict found or failed to load (normal for non-TTS mode).")

print("Model loaded successfully.")

# 准备图像输入
img_path = "resized.jpg"  # 示例图片路径
image = Image.open(img_path).convert("RGB").resize((224, 224))
image.save(img_path)

# 构造对话历史（系统提示词不含语音相关描述）
conversation = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant that can understand images and answer questions."}],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": "Describe objects and their relative locations in details:"}
        ],
    },
]

# 多模态预处理
USE_AUDIO_IN_VIDEO = False
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = [], [image], []

inputs = processor(
    text=text,
    audio=audios,
    images=images,
    videos=videos,
    return_tensors="pt",
    padding=True,
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
)

# 获取设备信息并移动张量
device = next(model.parameters()).device
dtype = next(model.parameters()).dtype
inputs = inputs.to(device=device, dtype=dtype)

# 推理参数设置
output = model.generate(
    **inputs,
    use_audio_in_video=False,
    return_audio=False,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

# 解码输出
response = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("\n\n🤖 Model Response:\n")
print(response)