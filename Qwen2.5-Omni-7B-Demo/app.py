import gradio as gr
from http import HTTPStatus
import uuid
from gradio_client import utils as client_utils
import gradio.processing_utils as processing_utils
import base64
from openai import OpenAI
import soundfile as sf
import numpy as np
import io
import os
import modelscope_studio.components.base as ms
import modelscope_studio.components.antd as antd
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider

# Voice settings
VOICE_LIST = ['Cherry', 'Ethan', 'Serena', 'Chelsie']
DEFAULT_VOICE = 'Cherry'

# OSS_ACCESS_KEY_ID and OSS_ACCESS_KEY_SECRET。
auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())

endpoint = os.getenv("OSS_ENDPOINT")

region = os.getenv("OSS_REGION")

bucket_name = os.getenv("OSS_BUCKET_NAME")

bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)

default_system_prompt = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'

API_KEY = os.environ['API_KEY']

client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

is_modelscope_studio = os.getenv('MODELSCOPE_ENVIRONMENT') == 'studio'


def get_text(text: str, cn_text: str):
    if is_modelscope_studio:
        return cn_text
    return text


def encode_file_to_base64(file_path):
    with open(file_path, "rb") as file:
        mime_type = client_utils.get_mimetype(file_path)
        bae64_data = base64.b64encode(file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{bae64_data}"


def file_path_to_oss_url(file_path: str):
    if file_path.startswith("http"):
        return file_path
    ext = file_path.split('.')[-1]
    object_name = f'studio-temp/Qwen2.5-Omni-Demo/{uuid.uuid4()}.{ext}'
    response = bucket.put_object_from_file(object_name, file_path)
    file_url = file_path
    if response.status == HTTPStatus.OK:
        file_url = bucket.sign_url('GET',
                                   object_name,
                                   60 * 60,
                                   slash_safe=True)
    return file_url


def format_history(history: list, system_prompt: str, oss_cache):
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    for item in history:
        if isinstance(item["content"], str):
            messages.append({"role": item['role'], "content": item['content']})
        elif item["role"] == "user" and (isinstance(item["content"], list) or
                                         isinstance(item["content"], tuple)):
            file_path = item["content"][0]

            file_url = oss_cache.get(file_path,
                                     file_path_to_oss_url(file_path))
            oss_cache[file_path] = file_url

            file_url = file_url if file_url.startswith(
                "http") else encode_file_to_base64(file_path=file_path)

            mime_type = client_utils.get_mimetype(file_path)
            ext = file_path.split('.')[-1]

            if mime_type.startswith("image"):
                messages.append({
                    "role":
                    item['role'],
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": file_url
                        }
                    }]
                })
            elif mime_type.startswith("video"):
                messages.append({
                    "role":
                    item['role'],
                    "content": [{
                        "type": "video_url",
                        "video_url": {
                            "url": file_url
                        }
                    }]
                })
            elif mime_type.startswith("audio"):
                messages.append({
                    "role":
                    item['role'],
                    "content": [{
                        "type": "input_audio",
                        "input_audio": {
                            "data": file_url,
                            "format": ext
                        }
                    }]
                })
    return messages


def predict(messages, voice=DEFAULT_VOICE):
    print('predict history: ', messages)
    completion = client.chat.completions.create(
        model="qwen-omni-turbo",
        messages=messages,
        modalities=["text", "audio"],
        audio={
            "voice": voice,
            "format": "wav"
        },
        stream=True,
        stream_options={"include_usage": True})

    response_text = ""
    audio_str = ""
    for chunk in completion:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(
                    delta,
                    'audio') and delta.audio and delta.audio.get("transcript"):
                response_text += delta.audio.get("transcript")
            if hasattr(delta,
                       'audio') and delta.audio and delta.audio.get("data"):
                audio_str += delta.audio.get("data")
            yield {"type": "text", "data": response_text}
    pcm_bytes = base64.b64decode(audio_str)
    audio_np = np.frombuffer(pcm_bytes, dtype=np.int16)
    wav_io = io.BytesIO()
    sf.write(wav_io, audio_np, samplerate=24000, format="WAV")
    wav_io.seek(0)
    wav_bytes = wav_io.getvalue()
    audio_path = processing_utils.save_bytes_to_cache(
        wav_bytes, "audio.wav", cache_dir=demo.GRADIO_CACHE)
    yield {"type": "audio", "data": audio_path}


def media_predict(audio, video, history, system_prompt, state_value,
                  voice_choice):
    files = [audio, video]
    for f in files:
        if f:
            history.append({"role": "user", "content": (f, )})

    formatted_history = format_history(history=history,
                                       system_prompt=system_prompt,
                                       oss_cache=state_value["oss_cache"])

    # First yield
    yield (
        None,  # microphone
        None,  # webcam
        history,  # media_chatbot
        gr.update(visible=False),  # submit_btn
        gr.update(visible=True),  # stop_btn
        state_value  # state
    )

    history.append({"role": "assistant", "content": ""})

    for chunk in predict(formatted_history, voice_choice):
        if chunk["type"] == "text":
            history[-1]["content"] = chunk["data"]
            yield (
                None,  # microphone
                None,  # webcam
                history,  # media_chatbot
                gr.update(visible=False),  # submit_btn
                gr.update(visible=True),  # stop_btn
                state_value  # state
            )
        if chunk["type"] == "audio":
            history.append({
                "role": "assistant",
                "content": gr.Audio(chunk["data"])
            })

    # Final yield
    yield (
        None,  # microphone
        None,  # webcam
        history,  # media_chatbot
        gr.update(visible=True),  # submit_btn
        gr.update(visible=False),  # stop_btn
        state_value  # state
    )


def chat_predict(text, audio, image, video, history, system_prompt,
                 state_value, voice_choice):
    # Process text input
    if text:
        history.append({"role": "user", "content": text})

    # Process audio input
    if audio:
        history.append({"role": "user", "content": (audio, )})

    # Process image input
    if image:
        history.append({"role": "user", "content": (image, )})

    # Process video input
    if video:
        history.append({"role": "user", "content": (video, )})

    formatted_history = format_history(history=history,
                                       system_prompt=system_prompt,
                                       oss_cache=state_value["oss_cache"])

    yield None, None, None, None, history, state_value

    history.append({"role": "assistant", "content": ""})
    for chunk in predict(formatted_history, voice_choice):
        if chunk["type"] == "text":
            history[-1]["content"] = chunk["data"]
            yield gr.skip(), gr.skip(), gr.skip(), gr.skip(
            ), history, state_value
        if chunk["type"] == "audio":
            history.append({
                "role": "assistant",
                "content": gr.Audio(chunk["data"])
            })
    yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history, state_value


with gr.Blocks() as demo, ms.Application(), antd.ConfigProvider():
    state = gr.State({"oss_cache": {}})

    with gr.Sidebar(open=False):
        system_prompt_textbox = gr.Textbox(label="System Prompt",
                                           value=default_system_prompt)
        voice_choice = gr.Dropdown(label="Voice Choice",
                                   choices=VOICE_LIST,
                                   value=DEFAULT_VOICE)
    with antd.Flex(gap="small", justify="center", align="center"):
        antd.Image('./logo-1.png', preview=False, width=67, height=67)
        with antd.Flex(vertical=True, gap="small", align="center"):
            antd.Typography.Title("Qwen2.5-Omni Demo",
                                  level=1,
                                  elem_style=dict(margin=0, fontSize=28))
            with antd.Flex(vertical=True, gap="small"):
                antd.Typography.Text(get_text("🎯 Instructions for use:",
                                              "🎯 使用说明："),
                                     strong=True)
                antd.Typography.Text(
                    get_text(
                        "1️⃣ Click the Audio Record button or the Camera Record button.",
                        "1️⃣ 点击音频录制按钮，或摄像头-录制按钮"))
                antd.Typography.Text(
                    get_text("2️⃣ Input audio or video.", "2️⃣ 输入音频或者视频"))
                antd.Typography.Text(
                    get_text(
                        "3️⃣ Click the submit button and wait for the model's response.",
                        "3️⃣ 点击提交并等待模型的回答"))
        antd.Image('./logo-2.png',
                   preview=False,
                   width=80,
                   height=80,
                   elem_style=dict(marginTop=5))
    with gr.Tabs():
        with gr.Tab("Online"):
            with gr.Row():
                with gr.Column(scale=1):
                    microphone = gr.Audio(sources=['microphone'],
                                          format="wav",
                                          type="filepath")
                    webcam = gr.Video(sources=['webcam'],
                                      format="mp4",
                                      height=400,
                                      include_audio=True)
                    submit_btn = gr.Button(get_text("Submit", "提交"),
                                           variant="primary")
                    stop_btn = gr.Button(get_text("Stop", "停止"), visible=False)
                    clear_btn = gr.Button(get_text("Clear History", "清除历史"))
                with gr.Column(scale=2):
                    media_chatbot = gr.Chatbot(height=650, type="messages")

                def clear_history():
                    return [], gr.update(value=None), gr.update(value=None)

                submit_event = submit_btn.click(fn=media_predict,
                                                inputs=[
                                                    microphone, webcam,
                                                    media_chatbot,
                                                    system_prompt_textbox,
                                                    state, voice_choice
                                                ],
                                                outputs=[
                                                    microphone, webcam,
                                                    media_chatbot, submit_btn,
                                                    stop_btn, state
                                                ])
                stop_btn.click(
                    fn=lambda:
                    (gr.update(visible=True), gr.update(visible=False)),
                    inputs=None,
                    outputs=[submit_btn, stop_btn],
                    cancels=[submit_event],
                    queue=False)
                clear_btn.click(fn=clear_history,
                                inputs=None,
                                outputs=[media_chatbot, microphone, webcam])

        with gr.Tab("Offline"):
            chatbot = gr.Chatbot(type="messages", height=650)

            # Media upload section in one row
            with gr.Row(equal_height=True):
                audio_input = gr.Audio(sources=["upload"],
                                       type="filepath",
                                       label="Upload Audio",
                                       elem_classes="media-upload",
                                       scale=1)
                image_input = gr.Image(sources=["upload"],
                                       type="filepath",
                                       label="Upload Image",
                                       elem_classes="media-upload",
                                       scale=1)
                video_input = gr.Video(sources=["upload"],
                                       label="Upload Video",
                                       elem_classes="media-upload",
                                       scale=1)

            # Text input section
            text_input = gr.Textbox(show_label=False,
                                    placeholder="Enter text here...")

            # Control buttons
            with gr.Row():
                submit_btn = gr.Button(get_text("Submit", "提交"),
                                       variant="primary",
                                       size="lg")
                stop_btn = gr.Button(get_text("Stop", "停止"),
                                     visible=False,
                                     size="lg")
                clear_btn = gr.Button(get_text("Clear History", "清除历史"),
                                      size="lg")

            def clear_chat_history():
                return [], gr.update(value=None), gr.update(
                    value=None), gr.update(value=None), gr.update(value=None)

            submit_event = gr.on(
                triggers=[submit_btn.click, text_input.submit],
                fn=chat_predict,
                inputs=[
                    text_input, audio_input, image_input, video_input, chatbot,
                    system_prompt_textbox, state, voice_choice
                ],
                outputs=[
                    text_input, audio_input, image_input, video_input, chatbot,
                    state
                ])

            stop_btn.click(fn=lambda:
                           (gr.update(visible=True), gr.update(visible=False)),
                           inputs=None,
                           outputs=[submit_btn, stop_btn],
                           cancels=[submit_event],
                           queue=False)

            clear_btn.click(fn=clear_chat_history,
                            inputs=None,
                            outputs=[
                                chatbot, text_input, audio_input, image_input,
                                video_input
                            ])

            # Add some custom CSS to improve the layout
            gr.HTML("""
                <style>
                    .media-upload {
                        margin: 10px;
                        min-height: 160px;
                    }
                    .media-upload > .wrap {
                        border: 2px dashed #ccc;
                        border-radius: 8px;
                        padding: 10px;
                        height: 100%;
                    }
                    .media-upload:hover > .wrap {
                        border-color: #666;
                    }
                    /* Make upload areas equal width */
                    .media-upload {
                        flex: 1;
                        min-width: 0;
                    }
                </style>
            """)

demo.queue(default_concurrency_limit=100, max_size=100).launch(max_threads=100,
                                                               ssr_mode=False)
