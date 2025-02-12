import argparse

from exllamav2 import (
    ExLlamaV2Cache,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_Q6,
    ExLlamaV2Cache_Q8,
)
from transformers import LogitsProcessor
import torch
import random
import numpy as np
from pathlib import Path


def create_args(
    genre_txt: str = (Path(__file__) / "../../prompt_egs/genre.txt").as_posix(),
    lyrics_txt: str = (Path(__file__) / "../../prompt_egs/lyrics.txt").as_posix(),
    stage1_model: str = "m-a-p/YuE-s1-7B-anneal-en-cot",
    stage2_model="m-a-p/YuE-s2-1B-general",
    max_new_tokens: int = 3000,
    repetition_penalty: float = 1.1,
    run_n_segments: int = 2,
    stage1_cache_size: int = 16384,
    stage2_cache_size: int = 8192,
    use_audio_prompt: bool = False,
    audio_prompt_path: str = "",
    prompt_start_time: float = 0.0,
    prompt_end_time: float = 30.0,
    use_dual_tracks_prompt: bool = False,
    vocal_track_prompt_path: str = "",
    instrumental_track_prompt_path: str = "",
    output_dir: str = "./output",
    keep_intermediate: bool = False,
    disable_offload_model: bool = True,
    cuda_idx: int = 0,
    seed: int = 42,
    basic_model_config: str = (
        Path(__file__) / "../../../xcodec_mini_infer/final_ckpt/config.yaml"
    ).as_posix(),
    resume_path: str = (
        Path(__file__) / "../../../xcodec_mini_infer/final_ckpt/ckpt_00360000.pth"
    ).as_posix(),
    config_path: str = (
        Path(__file__) / "../../../xcodec_mini_infer/decoders/config.yaml"
    ).as_posix(),
    vocal_decoder_path: str = (
        Path(__file__) / "../../../xcodec_mini_infer/decoders/decoder_131000.pth"
    ).as_posix(),
    inst_decoder_path: str = (
        Path(__file__) / "../../../xcodec_mini_infer/decoders/decoder_151000.pth"
    ).as_posix(),
    rescale: bool = False,
    stage1_cache_mode: str = "FP16",
    stage2_cache_mode: str = "FP16",
) -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    # Model Configuration:
    parser.add_argument(
        "--stage1_model",
        type=str,
        default="m-a-p/YuE-s1-7B-anneal-en-cot",
        help="The model checkpoint path or identifier for the Stage 1 model.",
    )
    parser.add_argument(
        "--stage2_model",
        type=str,
        default="m-a-p/YuE-s2-1B-general",
        help="The model checkpoint path or identifier for the Stage 2 model.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=3000,
        help="The maximum number of new tokens to generate in one pass during text generation.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="repetition_penalty ranges from 1.0 to 2.0 (or higher in some cases). It controls the diversity and coherence of the audio tokens generated. The higher the value, the greater the discouragement of repetition. Setting value to 1.0 means no penalty.",
    )
    parser.add_argument(
        "--run_n_segments",
        type=int,
        default=2,
        help="The number of segments to process during the generation.",
    )
    parser.add_argument(
        "--stage1_use_exl2",
        action="store_true",
        help="Use exllamav2 to load and run stage 1 model.",
    )
    parser.add_argument(
        "--stage2_use_exl2",
        action="store_true",
        help="Use exllamav2 to load and run stage 2 model.",
    )
    parser.add_argument(
        "--stage2_batch_size",
        type=int,
        default=4,
        help="The non-exl2 batch size used in Stage 2 inference.",
    )
    parser.add_argument(
        "--stage1_cache_size",
        type=int,
        default=16384,
        help="The cache size used in Stage 1 inference.",
    )
    parser.add_argument(
        "--stage2_cache_size",
        type=int,
        default=8192,
        help="The exl2 cache size used in Stage 2 inference.",
    )
    parser.add_argument(
        "--stage1_cache_mode",
        type=str,
        default="FP16",
        help="The cache mode used in Stage 1 inference (FP16, Q8, Q6, Q4). Quantized k/v cache will save VRAM at the cost of some speed and precision.",
    )
    parser.add_argument(
        "--stage2_cache_mode",
        type=str,
        default="FP16",
        help="The cache mode used in Stage 2 inference (FP16, Q8, Q6, Q4). Quantized k/v cache will save VRAM at the cost of some speed and precision.",
    )
    parser.add_argument(
        "--stage1_no_guidance",
        action="store_true",
        help="Disable classifier-free guidance for stage 1",
    )
    # Prompt
    parser.add_argument(
        "--genre_txt",
        type=str,
        required=True,
        help="The file path to a text file containing genre tags that describe the musical style or characteristics (e.g., instrumental, genre, mood, vocal timbre, vocal gender). This is used as part of the generation prompt.",
    )
    parser.add_argument(
        "--lyrics_txt",
        type=str,
        required=True,
        help="The file path to a text file containing the lyrics for the music generation. These lyrics will be processed and split into structured segments to guide the generation process.",
    )
    parser.add_argument(
        "--use_audio_prompt",
        action="store_true",
        help="If set, the model will use an audio file as a prompt during generation. The audio file should be specified using --audio_prompt_path.",
    )
    parser.add_argument(
        "--audio_prompt_path",
        type=str,
        default="",
        help="The file path to an audio file to use as a reference prompt when --use_audio_prompt is enabled.",
    )
    parser.add_argument(
        "--prompt_start_time",
        type=float,
        default=0.0,
        help="The start time in seconds to extract the audio prompt from the given audio file.",
    )
    parser.add_argument(
        "--prompt_end_time",
        type=float,
        default=30.0,
        help="The end time in seconds to extract the audio prompt from the given audio file.",
    )
    parser.add_argument(
        "--use_dual_tracks_prompt",
        action="store_true",
        help="If set, the model will use dual tracks as a prompt during generation. The vocal and instrumental files should be specified using --vocal_track_prompt_path and --instrumental_track_prompt_path.",
    )
    parser.add_argument(
        "--vocal_track_prompt_path",
        type=str,
        default="",
        help="The file path to a vocal track file to use as a reference prompt when --use_dual_tracks_prompt is enabled.",
    )
    parser.add_argument(
        "--instrumental_track_prompt_path",
        type=str,
        default="",
        help="The file path to an instrumental track file to use as a reference prompt when --use_dual_tracks_prompt is enabled.",
    )
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The directory where generated outputs will be saved.",
    )
    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="If set, intermediate outputs will be saved during processing.",
    )
    parser.add_argument(
        "--disable_offload_model",
        action="store_true",
        help="If set, the model will not be offloaded from the GPU to CPU after Stage 1 inference.",
    )
    parser.add_argument("--cuda_idx", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="An integer value to reproduce generation.",
    )
    # Config for xcodec and upsampler
    parser.add_argument(
        "--basic_model_config",
        default="./xcodec_mini_infer/final_ckpt/config.yaml",
        help="YAML files for xcodec configurations.",
    )
    parser.add_argument(
        "--resume_path",
        default="./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth",
        help="Path to the xcodec checkpoint.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./xcodec_mini_infer/decoders/config.yaml",
        help="Path to Vocos config file.",
    )
    parser.add_argument(
        "--vocal_decoder_path",
        type=str,
        default="./xcodec_mini_infer/decoders/decoder_131000.pth",
        help="Path to Vocos decoder weights.",
    )
    parser.add_argument(
        "--inst_decoder_path",
        type=str,
        default="./xcodec_mini_infer/decoders/decoder_151000.pth",
        help="Path to Vocos decoder weights.",
    )
    parser.add_argument(
        "-r", "--rescale", action="store_true", help="Rescale output to avoid clipping."
    )

    args = parser.parse_args(
        [
            "--genre_txt",
            genre_txt,
            "--lyrics_txt",
            lyrics_txt,
            "--stage1_model",
            stage1_model,
            "--stage2_model",
            stage2_model,
            "--max_new_tokens",
            str(max_new_tokens),
            "--repetition_penalty",
            str(repetition_penalty),
            "--run_n_segments",
            str(run_n_segments),
            "--output_dir",
            output_dir,
            "--cuda_idx",
            str(cuda_idx),
            "--basic_model_config",
            basic_model_config,
            "--resume_path",
            resume_path,
            "--config_path",
            config_path,
            "--vocal_decoder_path",
            vocal_decoder_path,
            "--inst_decoder_path",
            inst_decoder_path,
            "--seed",
            str(seed),
        ]
    )

    if use_audio_prompt:
        args.use_audio_prompt = True
        args.audio_prompt_path = audio_prompt_path
        args.prompt_start_time = prompt_start_time
        args.prompt_end_time = prompt_end_time

    if use_dual_tracks_prompt:
        args.use_dual_tracks_prompt = True
        args.vocal_track_prompt_path = vocal_track_prompt_path
        args.instrumental_track_prompt_path = instrumental_track_prompt_path

    args.keep_intermediate = keep_intermediate

    args.disable_offload_model = disable_offload_model

    args.rescale = rescale

    args.stage1_use_exl2 = True

    args.stage2_use_exl2 = True

    args.stage1_cache_size = stage1_cache_size

    args.stage2_cache_size = stage2_cache_size

    args.stage1_cache_mode = stage1_cache_mode

    args.stage2_cache_mode = stage2_cache_mode

    return args, parser


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cache_class(cache_mode: str):
    match cache_mode:
        case "Q4":
            return ExLlamaV2Cache_Q4
        case "Q6":
            return ExLlamaV2Cache_Q6
        case "Q8":
            return ExLlamaV2Cache_Q8
        case _:
            return ExLlamaV2Cache


class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores
