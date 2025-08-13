import json
import os
import sys
import threading
import time

import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse

parser = argparse.ArgumentParser(description="IndexTTS WebUI")
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr

from indextts.infer_indextts2 import IndexTTS2
from indextts import infer_indextts2
from tools.i18n.i18n import I18nAuto
from modelscope.hub import api

i18n = I18nAuto(language="zh_CN")
MODE = 'local'
tts = IndexTTS2(model_dir=cmd_args.model_dir, cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"), is_fp16=False,
                use_deepspeed=False)

os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70
with open("tests/cases.jsonl", "r", encoding="utf-8") as f:
    example_cases = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_vector"):
            example["use_emotion_vector"] = True
            emo_vector = example.get("emo_vector")
        else:
            example["use_emotion_vector"] = False
            emo_vector = [0.0] * 8
        example_cases.append([os.path.join("tests", example.get("prompt_audio", "sample_prompt.wav")),
                              example.get("text"),
                              # ["普通推理", "批次推理"][example.get("infer_mode", 0)],
                              example.get("use_emotion_reference", False),
                              example.get("emo_ref_path", None),
                              example.get("emo_weight", 0.5),
                              example.get("use_emotion_vector", False),
                              *emo_vector,
                              example.get("use_emotion_text", False),
                              example.get("emo_text", ""),
                              example.get("use_duration", False),
                              example.get("duration", "1.0"),
                              ])


def validate_duration(duration):
    try:
        duration_float = float(duration)
        if duration_float <= 0:
            return None, "时长必须为正数"
        return duration_float, None
    except ValueError:
        return None, "请输入有效的浮点数"


def toggle_emotion_controls(use_emotion_reference):
    if use_emotion_reference:
        return (gr.update(visible=True), gr.update(visible=True),
                gr.update(value=False),
                gr.update(value=False))
    else:
        return (gr.update(visible=False), gr.update(visible=False),
                gr.update(),
                gr.update()
                )


def toggle_emotion_vector(use_emotion_vector):
    if use_emotion_vector:
        return (gr.update(visible=True),
                gr.update(value=False),
                gr.update(value=False))
    else:
        return (gr.update(visible=False),
                gr.update(),
                gr.update())


def toggle_emotion_text(use_emo_text):
    if use_emo_text:
        return (gr.update(visible=True),
                gr.update(value=False),
                gr.update(value=False))
    else:
        return (gr.update(visible=False),
                gr.update(),
                gr.update())


def toggle_duration(use_duration):
    if use_duration:
        return gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)


def gen_single(prompt, text,
               emo_ref_path, emo_weight,
               use_duration, duration,
               use_emotion_reference, use_emotion_vector,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               use_emo_text, emo_text, emo_text_weight,
               max_text_tokens_per_sentence=120, sentences_bucket_max_size=4,
               *args, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    # set gradio progress
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    emo_weight = emo_weight if (use_emotion_reference and emo_ref_path) else 1.0
    vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8] if use_emotion_vector else None
    if use_emo_text and not emo_text:
        emo_text = text

    # 新增：duration类型检查
    if use_duration:
        duration_checked, err = validate_duration(duration)
        if err:
            # 返回Gradio报错弹窗
            gr.Warning(err)
        duration = duration_checked

        # if infer_mode == "普通推理":
    output = tts.infer(spk_audio_prompt=prompt, text=text,
                       output_path=output_path,
                       emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                       emo_vector=vec,
                       use_emo_text=use_emo_text, emo_text=emo_text, emo_text_weight=float(emo_text_weight),
                       use_speed=use_duration, target_dur=float(duration),
                       verbose=cmd_args.verbose,
                       max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                       **kwargs)
    # else:
    #     # 批次推理
    #     output = tts.infer_fast(prompt, text, output_path, verbose=cmd_args.verbose,
    #         max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
    #         sentences_bucket_max_size=(sentences_bucket_max_size),
    #         **kwargs)
    return gr.update(value=output, visible=True)


def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button


with gr.Blocks(title="IndexTTS2 Demo") as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
    <h2><center>(情感丰富且时长可控的自回归零样本TTS系统)</h2>
<p align="center">
<a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>
</p>
    ''')
    with gr.Tab("音频生成"):
        with gr.Row():
            os.makedirs("prompts", exist_ok=True)
            prompt_audio = gr.Audio(label="参考音频", key="prompt_audio",
                                    sources=["upload", "microphone"], type="filepath")
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column():
                input_text_single = gr.TextArea(label="文本", key="input_text_single", placeholder="请输入目标文本",
                                                info="当前模型版本{}".format(tts.model_version or "1.0"))
                # infer_mode = gr.Radio(choices=["普通推理", "批次推理"], label="推理模式",info="批次推理：更适合长句，性能翻倍",value="普通推理")
                # infer_mode = gr.Radio(choices=["普通推理"], label="推理模式",info="批次推理：更适合长句，性能翻倍",value="普通推理")
                gen_button = gr.Button("生成语音", key="gen_button", interactive=True)
            output_audio = gr.Audio(label="生成结果", visible=True, key="output_audio")
        with gr.Accordion("功能设置"):
            # 时长控制部分
            with gr.Row():
                with gr.Column():
                    use_duration = gr.Checkbox(label="使用时长控制", value=False,interactive=True)
                    duration_tip = gr.Textbox(show_label=False, value="建议时长：", interactive=False, visible=False)
                duration = gr.Textbox(label="合成时长(秒)", value="1.0", visible=False)

            # 情感控制选项部分
            with gr.Row():
                gr.Markdown("### 情感控制方式")

            with gr.Row():
                use_emotion_reference = gr.Checkbox(label="使用情感参考音频", value=True)
                use_emotion_vector = gr.Checkbox(label="使用情感向量控制", value=False)
                use_emotion_text = gr.Checkbox(label="使用情感描述文本控制", value=False)
        # 情感参考音频部分
        with gr.Group(visible=True) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(label="上传情感参考音频", type="filepath")

            with gr.Row():
                emo_weight = gr.Slider(label="情感权重", minimum=0.0, maximum=1.6, value=0.5, step=0.01)

        # 情感向量控制部分
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label="喜", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec2 = gr.Slider(label="怒", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec3 = gr.Slider(label="哀", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec4 = gr.Slider(label="惧", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label="厌恶", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec6 = gr.Slider(label="低落", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec7 = gr.Slider(label="惊喜", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec8 = gr.Slider(label="平静", minimum=0.0, maximum=1.4, value=0.0, step=0.05)

        with gr.Group(visible=False) as emo_text_group:
            with gr.Row():
                emo_text = gr.Textbox(label="情感描述文本", placeholder="请输入情感描述文本", value="",
                                      info="例如：高兴，愤怒，悲伤等;不填写时，将使用目标文本作为情感描述文本")
                emo_text_weight = gr.Slider(label="情感描述文本权重", minimum=0.0, maximum=1.0, value=1.0, step=0.01)

        with gr.Accordion("高级生成参数设置", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        "**GPT2 采样设置** _参数会影响音频多样性和生成速度详见[Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)_")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info="是否进行采样")
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0,
                                                       minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0,
                                                   maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50,
                                               maximum=tts.cfg.gpt.max_mel_tokens, step=10,
                                               info="生成Token最大数量，过小导致音频被截断", key="max_mel_tokens")
                    # with gr.Row():
                    #     typical_sampling = gr.Checkbox(label="typical_sampling", value=False, info="不建议使用")
                    #     typical_mass = gr.Slider(label="typical_mass", value=0.9, minimum=0.0, maximum=1.0, step=0.1)
                with gr.Column(scale=2):
                    gr.Markdown("**分句设置** _参数会影响音频质量和生成速度_")
                    with gr.Row():
                        max_text_tokens_per_sentence = gr.Slider(
                            label="分句最大Token数", value=120, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2,
                            key="max_text_tokens_per_sentence",
                            info="建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高",
                        )
                        sentences_bucket_max_size = gr.Slider(
                            label="分句分桶的最大容量（批次推理生效）", value=4, minimum=1, maximum=16, step=1,
                            key="sentences_bucket_max_size",
                            info="建议2-8之间，值越大，一批次推理包含的分句数越多，过大可能导致内存溢出",
                        )
                    with gr.Accordion("预览分句结果", open=True) as sentences_settings:
                        sentences_preview = gr.Dataframe(
                            headers=["序号", "分句内容", "Token数"],
                            key="sentences_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                # typical_sampling, typical_mass,
            ]

        # if len(example_cases) > 0:
        #     gr.Examples(
        #         examples=example_cases,
        #         inputs=[prompt_audio,
        #                 input_text_single,
        #                 infer_mode,
        #                 use_emotion_reference,
        #                 emo_upload,
        #                 emo_weight,
        #                 use_emotion_vector,
        #                 vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
        #                 use_emotion_text,
        #                 emo_text,
        #                 use_duration,
        #                 duration
        #                 ],
        #     )


    def on_input_text_change(text, max_tokens_per_sentence):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            sentences = tts.tokenizer.split_sentences(text_tokens_list,
                                                      max_tokens_per_sentence=int(max_tokens_per_sentence))
            data = []
            for i, s in enumerate(sentences):
                sentence_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, sentence_str, tokens_count])
            min_dur, max_dur = infer_indextts2.get_text_tts_dur(text)
            # 判断文本长度：70个中文字符或70个英文单词
            import re
            def count_chinese(text):
                return len(re.findall(r'[\u4e00-\u9fff]', text))
            def count_english_words(text):
                return len(re.findall(r'[a-zA-Z]+', text))
            chinese_count = count_chinese(text)
            english_word_count = count_english_words(text)
            if chinese_count >= MAX_LENGTH_TO_USE_SPEED or english_word_count >= MAX_LENGTH_TO_USE_SPEED:
                return {
                    sentences_preview: gr.update(value=data, visible=True, type="array"),
                    duration_tip: gr.update(value=f"建议时长：{min_dur:.2f} - {max_dur:.2f} 秒"),
                    duration: gr.update(value=(min_dur + max_dur) / 2.0),
                    use_duration: gr.update(value=False, interactive=False)
                }
            else:
                return {
                    sentences_preview: gr.update(value=data, visible=True, type="array"),
                    duration_tip: gr.update(value=f"建议时长：{min_dur:.2f} - {max_dur:.2f} 秒"),
                    duration: gr.update(value=(min_dur + max_dur) / 2.0),
                    use_duration: gr.update(interactive=True)
                }
        else:
            df = pd.DataFrame([], columns=["序号", "分句内容", "Token数"])
            return {
                sentences_preview: gr.update(value=df),
                duration_tip: gr.update(value="建议时长："),
                duration: gr.update(value=1.0),
                use_duration: gr.update(interactive=True)
            }


    # 事件绑定
    def on_infer_mode_change(infer_mode="普通推理"):
        if infer_mode == "批次推理":
            return gr.update(value=False, interactive=False)
        else:
            return gr.update(interactive=True)


    # infer_mode.change(
    #     on_infer_mode_change,
    #     inputs=[infer_mode],
    #     outputs=[use_duration]
    # )
    use_emotion_reference.change(
        toggle_emotion_controls,
        inputs=[use_emotion_reference],
        outputs=[
            emotion_reference_group,
            emo_weight,
            use_emotion_vector,
            use_emotion_text,
        ]
    )

    use_emotion_vector.change(
        toggle_emotion_vector,
        inputs=[use_emotion_vector],
        outputs=[
            emotion_vector_group,
            use_emotion_reference,
            use_emotion_text,
        ]
    )
    use_emotion_text.change(
        toggle_emotion_text,
        inputs=[use_emotion_text],
        outputs=[
            emo_text_group,
            use_emotion_reference,
            use_emotion_vector,
        ]
    )

    use_duration.change(toggle_duration, inputs=[use_duration], outputs=[duration, duration_tip])

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_sentence],
        outputs=[sentences_preview, duration_tip, duration, use_duration]
    )
    max_text_tokens_per_sentence.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_sentence],
        outputs=[sentences_preview, duration_tip, duration, use_duration]
    )
    prompt_audio.upload(update_prompt_audio,
                        inputs=[],
                        outputs=[gen_button])

    gen_button.click(gen_single,
                     inputs=[prompt_audio, input_text_single, emo_upload, emo_weight,
                             use_duration, duration, use_emotion_reference, use_emotion_vector,
                             vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                             use_emotion_text, emo_text, emo_text_weight,
                             max_text_tokens_per_sentence, sentences_bucket_max_size,
                             *advanced_params,
                             ],
                     outputs=[output_audio])

if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)
