#!/usr/bin/env python3
"""Bengali ASR inference using fine-tuned Whisper Medium (float16 + SDPA).

Handles both short clips and long-form audio (via 30s chunked pipeline).
"""

import os
import sys

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# cuDNN 9.x lacks fp16 conv1d kernels for some GPUs; the fallback path works fine
torch.backends.cudnn.enabled = False


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.isfile(audio_path):
        print(f"Error: file not found: {audio_path}")
        sys.exit(1)

    # Load processor and model in float16 with SDPA attention
    processor = WhisperProcessor.from_pretrained(MODEL_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(DEVICE)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="bn", task="transcribe"
    )
    model.generation_config.forced_decoder_ids = forced_decoder_ids

    # Use chunked pipeline for robust long-form transcription
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        device=DEVICE,
        torch_dtype=torch.float16,
    )

    result = pipe(
        audio_path,
        generate_kwargs={"forced_decoder_ids": forced_decoder_ids},
    )
    print(result["text"])


if __name__ == "__main__":
    main()
