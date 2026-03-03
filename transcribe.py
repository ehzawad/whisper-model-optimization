#!/usr/bin/env python3
"""Bengali ASR inference using fine-tuned Whisper Medium (float16 + SDPA)."""

import os
import sys

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

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

    # Load and resample audio to 16 kHz
    audio, _ = librosa.load(audio_path, sr=16000)
    input_features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE, dtype=torch.float16)

    # Force Bengali transcription (set on generation_config for older checkpoints)
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="bn", task="transcribe"
    )

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(transcription)


if __name__ == "__main__":
    main()
