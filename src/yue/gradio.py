import gradio as gr
import os
from .common import create_args
from .infer_stage1 import main as stage1_main
from .infer_stage2 import main as stage2_main
from .infer_postprocess import main as postprocess_main
from pathlib import Path
import torch
import json
import random


def generate_music(
    genre_txt,
    lyrics_txt,
    stage1_model,
    stage2_model,
    max_new_tokens,
    run_n_segments,
    stage1_cache_size,
    stage2_cache_size,
    use_audio_prompt,
    audio_prompt_file,
    prompt_start_time,
    prompt_end_time,
    use_dual_tracks_prompt,
    vocal_track_prompt_path,
    instrumental_track_prompt_path,
    output_dir,
    keep_intermediate,
    cuda_idx,
    seed,
    rescale,
    stage1_cache_mode,
    stage2_cache_mode,
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_dir = Path(output_dir).absolute().as_posix()

    # Handle audio prompt path
    audio_prompt_path = ""
    if use_audio_prompt and audio_prompt_file is not None:
        audio_prompt_path = audio_prompt_file.name

    # Create arguments using the create_args function
    args, _ = create_args(
        genre_txt=genre_txt,
        lyrics_txt=lyrics_txt,
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        max_new_tokens=max_new_tokens,
        run_n_segments=run_n_segments,
        stage1_cache_size=stage1_cache_size,
        stage2_cache_size=stage2_cache_size,
        use_audio_prompt=use_audio_prompt,
        audio_prompt_path=audio_prompt_path,
        prompt_start_time=prompt_start_time,
        prompt_end_time=prompt_end_time,
        use_dual_tracks_prompt=use_dual_tracks_prompt,
        vocal_track_prompt_path=vocal_track_prompt_path,
        instrumental_track_prompt_path=instrumental_track_prompt_path,
        output_dir=output_dir,
        keep_intermediate=keep_intermediate,
        rescale=rescale,
        cuda_idx=int(cuda_idx),
        seed=int(seed),
        stage1_cache_mode=stage1_cache_mode,
        stage2_cache_mode=stage2_cache_mode,
    )

    # Generate music
    print("Starting stage 1...")
    stage1_main(args)
    print("Starting stage 2...")
    stage2_main(args)
    print("Starting postprocessing...")
    output_audio = postprocess_main(args)

    # Return the generated audio files
    return output_audio


def load_tags():
    try:
        tags_file = Path(__file__).parent.parent / "top_200_tags.json"
        with open(tags_file, "r", encoding="utf-8") as f:
            tags_data = json.load(f)
            # Combine all values from each category and remove duplicates
            all_tags = []
            for category_tags in tags_data.values():
                all_tags.extend(tag.lower() for tag in category_tags)
            # Remove duplicates while preserving order
            return list(dict.fromkeys(all_tags))
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback tags if file is not found or invalid
        return [
            "pop",
            "rock",
            "jazz",
            "classical",
            "electronic",
            "hip hop",
            "r&b",
            "country",
            "folk",
            "metal",
            "indie",
            "blues",
        ]


def get_random_tags(n=12):
    tags = load_tags()
    n = min(n, len(tags))  # Ensure we don't try to sample more than available
    return random.sample(tags, n)


def toggle_tag(current_tags, tag_to_toggle):
    if not current_tags:
        return tag_to_toggle, gr.update(variant="primary")

    current_tag_list = current_tags.split()
    if tag_to_toggle in current_tag_list:
        # Remove the tag if it exists
        current_tag_list = [t for t in current_tag_list if t != tag_to_toggle]
        new_variant = "secondary"
    else:
        # Add the tag if it doesn't exist
        current_tag_list.append(tag_to_toggle)
        new_variant = "primary"

    return " ".join(current_tag_list), gr.update(variant=new_variant)


def refresh_tag_buttons():
    tags = get_random_tags()
    return [
        gr.update(value=tag, variant="secondary") for tag in tags
    ] + [  # for buttons
        gr.update(value=tag) for tag in tags
    ]  # for hidden textboxes


def clear_tags():
    return "", *[gr.update(variant="secondary") for _ in range(12)]


# Create the Gradio interface
with gr.Blocks(
    theme=gr.themes.Glass(
        primary_hue="green",
        secondary_hue="violet",
        neutral_hue="slate",
    )
) as demo:
    gr.Markdown(
        """
        # YuE Music Generation Interface
        Generate music with lyrics and genre tags using the YuE model.

        <div style="display:flex;justify-content:center;column-gap:4px;">
            <a href="https://github.com/multimodal-art-projection/YuE">
                <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
            </a> 
            <a href="https://x.com/bdsqlsz">
                <img src="https://img.shields.io/twitter/follow/bdsqlsz">
            </a>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Input Parameters")
                with gr.Row():
                    genre_txt = gr.Textbox(
                        label="Genre Tags", placeholder="Enter genre tags here", scale=4
                    )
                with gr.Row():
                    gr.Markdown("### Suggested Tags")
                    refresh_tags = gr.Button(
                        "🎲",
                        size="sm",
                        variant="primary",
                        scale=0.1,
                        min_width=1,
                    )
                    clear_tags_btn = gr.Button(
                        "🗑️",
                        size="sm",
                        variant="primary",
                        scale=0.1,
                        min_width=1,
                    )
                with gr.Row():
                    tag_buttons = []
                    tag_values = []
                    for tag in get_random_tags():
                        tag_value = gr.Textbox(value=tag, visible=False)
                        btn = gr.Button(tag, size="md", variant="secondary")
                        btn.click(
                            fn=toggle_tag,
                            inputs=[genre_txt, tag_value],
                            outputs=[genre_txt, btn],
                        )
                        tag_buttons.append(btn)
                        tag_values.append(tag_value)
                lyrics_txt = gr.Textbox(
                    label="Lyrics",
                    placeholder="Enter lyrics here",
                    lines=10,
                )

            with gr.Group():
                gr.Markdown("### Model Settings")
                with gr.Row():
                    stage1_model = gr.Dropdown(
                        label="Stage 1 Model",
                        choices=[
                            "m-a-p/YuE-s1-7B-anneal-en-cot",
                            "m-a-p/YuE-s1-7B-anneal-en-icl",
                            "m-a-p/YuE-s1-7B-anneal-jp-kr-cot",
                            "m-a-p/YuE-s1-7B-anneal-jp-kr-icl",
                            "m-a-p/YuE-s1-7B-anneal-zh-cot",
                            "m-a-p/YuE-s1-7B-anneal-zh-icl",
                        ],
                        value="m-a-p/YuE-s1-7B-anneal-en-cot",
                    )
                    stage1_cache_mode = gr.Radio(
                        label="stage 1 cache mode",
                        choices=["FP16", "Q8", "Q6", "Q4"],
                        value="FP16",
                        info="Recommended value depends on your GPU memory.",
                    )
                with gr.Row():
                    stage2_model = gr.Dropdown(
                        label="Stage 2 Model",
                        choices=[
                            "m-a-p/YuE-s2-1B-general",
                        ],
                        value="m-a-p/YuE-s2-1B-general",
                    )
                    stage2_cache_mode = gr.Radio(
                        label="stage 2 cache mode",
                        choices=["FP16", "Q8", "Q6", "Q4"],
                        value="FP16",
                        info="Recommended value depends on your GPU memory.",
                    )
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Generation Settings")
                with gr.Row():
                    max_new_tokens = gr.Slider(
                        label="Max New Tokens",
                        minimum=1,
                        maximum=16384,
                        value=3000,
                        step=100,
                        info="The maximum number of tokens to generate.",
                    )
                    run_n_segments = gr.Slider(
                        label="Number of Segments",
                        minimum=1,
                        maximum=5,
                        value=2,
                        step=1,
                        info="The number of segments to process during the generation.",
                    )
                with gr.Row():
                    stage1_cache_size = gr.Number(
                        label="stage 1 cache size",
                        value=16384,
                        minimum=1024,
                        maximum=81920,
                        info="Recommended value depends on your GPU memory.",
                        step=1024,
                    )
                    stage2_cache_size = gr.Number(
                        label="stage 2 cache size",
                        value=(
                            round(
                                torch.cuda.get_device_properties(0).total_memory
                                / (1024 * 1024 * 1024)
                            )
                            - 1
                        )
                        * 1024,
                        minimum=1024,
                        maximum=81920,
                        info="Recommended value depends on your GPU memory.",
                        step=1024,
                    )
                    cuda_idx = gr.Radio(
                        label="CUDA Index",
                        choices=[str(i) for i in range(torch.cuda.device_count())],
                        value="0",
                        type="index",
                    )

                with gr.Row():
                    seed = gr.Slider(
                        label="Seed",
                        value=42,
                        minimum=0,
                        maximum=999999999999,
                        step=1,
                        interactive=True,
                        info="Random seed for reproducibility.",
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        keep_intermediate = gr.Checkbox(
                            label="Keep Intermediate Files", value=True
                        )
                        use_audio_prompt = gr.Checkbox(
                            label="Use Audio Prompt", value=False
                        )

                    with gr.Column(scale=1):
                        rescale = gr.Checkbox(label="Rescale Audio", value=False)
                        use_dual_tracks_prompt = gr.Checkbox(
                            label="Use Dual Tracks Prompt", value=False
                        )

            with gr.Group():
                gr.Markdown("### Audio Prompt Settings")
                audio_prompt_file = gr.Audio(
                    label="Audio Prompt File", type="filepath", visible=False
                )
                with gr.Row():
                    vocal_track_prompt_path = gr.Audio(
                        label="Vocal Track Prompt File",
                        type="filepath",
                        visible=False,
                        min_width=80,
                        scale=0.5,
                    )
                    instrumental_track_prompt_path = gr.Audio(
                        label="Instrumental Track Prompt File",
                        type="filepath",
                        visible=False,
                        min_width=80,
                        scale=0.5,
                    )
                with gr.Row():
                    prompt_start_time = gr.Number(
                        label="Prompt Start Time (s)",
                        value=0.0,
                        visible=False,
                        interactive=True,
                    )
                    prompt_end_time = gr.Number(
                        label="Prompt End Time (s)",
                        value=30.0,
                        visible=False,
                        interactive=True,
                    )

            output_dir = gr.Textbox(label="Output Directory", value="./output")

            generate_btn = gr.Button(
                "Generate Music 🎵",
                variant="primary",
                scale=2,
                size="lg",
            )
            output_audio = gr.Audio(label="Generated Music")

    with gr.Row():
        gr.Examples(
            examples=[
                [
                    "m-a-p/YuE-s1-7B-anneal-en-cot",
                    "inspiring female uplifting pop airy vocal electronic bright vocal vocal",
                    """[verse]
Staring at the sunset, colors paint the sky
Thoughts of you keep swirling, can't deny
I know I let you down, I made mistakes
But I'm here to mend the heart I didn't break

[chorus]
Every road you take, I'll be one step behind
Every dream you chase, I'm reaching for the light
You can't fight this feeling now
I won't back down
You know you can't deny it now
I won't back down

[verse]
They might say I'm foolish, chasing after you
But they don't feel this love the way we do
My heart beats only for you, can't you see?
I won't let you slip away from me

[chorus]
Every road you take, I'll be one step behind
Every dream you chase, I'm reaching for the light
You can't fight this feeling now
I won't back down
You know you can't deny it now
I won't back down

[bridge]
No, I won't back down, won't turn around
Until you're back where you belong
I'll cross the oceans wide, stand by your side
Together we are strong

[outro]
Every road you take, I'll be one step behind
Every dream you chase, love's the tie that binds
You can't fight this feeling now
I won't back down""",
                ],
                [
                    "m-a-p/YuE-s1-7B-anneal-zh-cot",
                    "儿童 female clear vocal 动画 high-pitched vocal 声乐 有趣 打击乐器 欢乐",
                    """[verse]
疙疙瘩瘩，松松垮垮，
歪歪扭扭，坑坑洼洼。
大脑，今天再次启动咯！
行嘞~！
我是疙疙瘩瘩的大脑，
时而迷糊时而明晰。
我是疙疙瘩瘩的大脑，
哟呵！搞清楚了！
我是松松垮垮的大脑，
有时愉快有时烦闷。
我是松松垮垮的大脑，
哎~实在太烦啦！
疙疙瘩瘩，松松垮垮，
歪歪扭扭，坑坑洼洼。
哟~真不轻松呀！


[chorus]
我是晃晃的大脑，
思一思后做判定。
我是晃晃的大脑，
嘿！就选定！
我是凸凹的大脑，
壮壮身躯随我令。
我是凸凹的大脑，
哟！要警醒！
皱皱巴巴，软软绵绵，
弯弯曲曲，凹凸不平。
大脑，今天也多谢你啦！


[end]
没关系！""",
                ],
            ],
            inputs=[stage1_model, genre_txt, lyrics_txt],
            cache_examples=False,
            examples_per_page=100,
        )

    # Show/hide audio prompt options based on checkbox
    def update_prompt_times(audio_data):
        if audio_data is None:
            return gr.update(value=0.0), gr.update(value=30.0)
        return gr.update(value=0.0), gr.update(
            value=audio_data[1]
        )  # audio_data[1] contains duration

    use_audio_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[use_audio_prompt],
        outputs=[audio_prompt_file],
    )
    use_audio_prompt.change(
        fn=lambda x: gr.update(
            choices=(
                [m for m in stage1_model.choices if "icl" in m[0].lower()]
                if x
                else stage1_model.choices
            ),
            value=(
                stage1_model.value.replace("cot", "icl") if x else stage1_model.value
            ),
        ),
        inputs=[use_audio_prompt],
        outputs=[stage1_model],
    )
    use_audio_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[use_audio_prompt or use_dual_tracks_prompt],
        outputs=[prompt_start_time],
    )

    use_audio_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[use_audio_prompt or use_dual_tracks_prompt],
        outputs=[prompt_end_time],
    )

    use_dual_tracks_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[use_dual_tracks_prompt],
        outputs=[vocal_track_prompt_path],
    )

    use_dual_tracks_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[use_dual_tracks_prompt],
        outputs=[instrumental_track_prompt_path],
    )

    use_dual_tracks_prompt.change(
        fn=lambda x: gr.update(
            choices=(
                [m for m in stage1_model.choices if "icl" in m[0].lower()]
                if x
                else stage1_model.choices
            ),
            value=(
                stage1_model.value.replace("cot", "icl") if x else stage1_model.value
            ),
        ),
        inputs=[use_dual_tracks_prompt],
        outputs=[stage1_model],
    )
    use_dual_tracks_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[use_dual_tracks_prompt or use_audio_prompt],
        outputs=[prompt_start_time],
    )
    use_dual_tracks_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[use_dual_tracks_prompt or use_audio_prompt],
        outputs=[prompt_end_time],
    )

    # Update prompt times when audio file changes
    audio_prompt_file.change(
        fn=update_prompt_times,
        inputs=[audio_prompt_file],
        outputs=[prompt_start_time, prompt_end_time],
    )

    # Connect the generate button to the generate_music function
    generate_btn.click(
        fn=generate_music,
        inputs=[
            genre_txt,
            lyrics_txt,
            stage1_model,
            stage2_model,
            max_new_tokens,
            run_n_segments,
            stage1_cache_size,
            stage2_cache_size,
            use_audio_prompt,
            audio_prompt_file,
            prompt_start_time,
            prompt_end_time,
            use_dual_tracks_prompt,
            vocal_track_prompt_path,
            instrumental_track_prompt_path,
            output_dir,
            keep_intermediate,
            cuda_idx,
            seed,
            rescale,
            stage1_cache_mode,
            stage2_cache_mode,
        ],
        outputs=[output_audio],
    )

    # Connect refresh button to tag buttons
    refresh_tags.click(
        fn=refresh_tag_buttons,
        outputs=[*tag_buttons, *tag_values],
    )

    # Connect clear button to clear tags function
    clear_tags_btn.click(
        fn=clear_tags,
        outputs=[genre_txt, *tag_buttons],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
