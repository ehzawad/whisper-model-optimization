# whisper-model-optimization

Fine-tuned Whisper Medium for Bengali speech recognition, powered by faster-whisper (CTranslate2).

## Quick Start

```bash
pip install -r requirements.txt

# Convert HF model to CTranslate2 format (one-time):
ct2-transformers-converter --model . --output_dir ct2_model_fp16 \
    --copy_files tokenizer.json preprocessor_config.json --quantization float16
```

### Transcribe audio

```bash
python transcribe_fw.py audio.wav
python transcribe_fw.py long_meeting.wav
python transcribe_fw.py meeting_22_11_23/
python transcribe_fw.py file1.wav file2.wav --json
python transcribe_fw.py audio.wav --device cpu --compute-type int8
```

**Options:**

```
--json              JSON output with per-file RTF and timing
--compute-type T    float16 (default), int8_float16, int8, float32
--beam-size N       Beam size for decoding (default: 1)
--batch-size N      Batch size for batched inference (default: 8)
--no-vad            Disable Silero VAD filtering
--ct2-model PATH    Path to CTranslate2 model dir (default: ct2_model_fp16/)
--cpu-threads N     CPU threads (default: 4)
```

### Python API

```python
from transcribe_fw import transcribe_file, transcribe_audio, get_model

text = transcribe_file("audio.wav")

model = get_model()  # loaded once, reused
text = transcribe_file("audio.wav", model=model)
```

### Live streaming ASR

```bash
uvicorn streaming.server:app --host 0.0.0.0 --port 8000

# Python client
python clients/test_client.py audio.wav

# Browser client
open http://localhost:8000
```

## Benchmarks: Naive HF Pipeline vs faster-whisper

**Task:** Bengali speech-to-text on a 947.3s (15m47s) meeting recording
**Model:** Fine-tuned Whisper Medium for Bengali (float16, CTranslate2 4.7.1)

### Reproduce

```bash
AUDIO=meeting_22_11_23/single_audio/2023-11-22T07_37_42.518693Z_65f437bd-53d0-4ff2-a667-9b5a6935c52d.wav

# naive HF baseline (transformers.pipeline, batch_size=1, sequential decoding)
python transcribe_naive.py $AUDIO --chunk-length 15
python transcribe_naive.py $AUDIO --chunk-length 30

# optimized HF (SDPA + batched chunking + cuDNN benchmark + auto batch_size)
python transcribe.py $AUDIO --chunk-length 15
python transcribe.py $AUDIO --chunk-length 30

# faster-whisper (CTranslate2 + Silero VAD + batched inference)
python transcribe_fw.py $AUDIO --chunk-length 15
python transcribe_fw.py $AUDIO --chunk-length 30

# faster-whisper default (chunk_length=None → 30s from model FeatureExtractor)
python transcribe_fw.py $AUDIO
```

### Tesla T4 (16GB VRAM)

| Approach | Chunk | Inference | RTF | Throughput | Speedup |
|---|---|---|---|---|---|
| Naive HF pipeline | 15s | 290.5s | 0.307 | 3.3x | baseline (1.0x) |
| Naive HF pipeline | 30s | 283.2s | 0.299 | 3.3x | 1.0x |
| Optimized HF (SDPA+batch) | 15s | 23.7s | 0.025 | 39.9x | 12.3x |
| **Optimized HF (SDPA+batch)** | **30s** | **15.5s** | **0.016** | **61.2x** | **18.7x** |
| faster-whisper | 15s | 25.8s | 0.027 | 36.6x | 11.3x |
| **faster-whisper** | **30s** | **18.2s** | **0.019** | **52.2x** | **16.0x** |

### RTX 2050 (4GB VRAM)

| Approach | Chunk | Inference | RTF | Throughput | Speedup |
|---|---|---|---|---|---|
| Naive HF pipeline | 15s | 501.6s | 0.530 | 1.9x | baseline (1.0x) |
| Naive HF pipeline | 30s | 497.9s | 0.526 | 1.9x | 1.0x |
| **faster-whisper** | **15s** | **55.7s** | **0.059** | **17.0x** | **9.0x** |
| **faster-whisper** | **30s** | **39.4s** | **0.042** | **24.0x** | **12.6x** |

### Key observations

- **Naive HF pipeline** uses `transformers.pipeline(chunk_length_s=N)` with sequential decoding, deprecated `forced_decoder_ids`, and `batch_size=1`. Chunk size has negligible impact (~2%) because the bottleneck is sequential per-chunk decoding.
- **Optimized HF** adds fp16 + SDPA attention + cuDNN benchmark mode + batched chunk processing (auto `batch_size=41` on T4's 16GB). This is why it's competitive with — and even slightly faster than — faster-whisper on T4: the large VRAM allows processing 41 chunks in parallel.
- **faster-whisper** wins on VRAM-constrained GPUs (RTX 2050, 4GB) where the optimized HF batch size drops to ~3. CTranslate2's C++ kernels + Silero VAD silence-skipping dominate at low batch sizes.
- **15s vs 30s chunks:** Consistently ~40–50% slower across all approaches because more chunks = more decoder passes (64 vs 32). Each chunk incurs fixed encoder + decoder overhead.
- **T4 vs RTX 2050:** T4's 320 GB/s memory bandwidth + 16GB VRAM delivers dramatically higher throughput, especially for the optimized HF approach which scales with batch size.

## Architecture

```
transcribe_fw.py           # faster-whisper (CTranslate2) inference
transcribe_naive.py        # Naive HF pipeline baseline (for benchmarking)
transcribe.py              # HF pipeline (used by streaming server)
ct2_model_fp16/            # CTranslate2 converted model (float16, 1.5GB)
streaming/
  server.py                # FastAPI + WebSocket server (multi-session)
  asr_engine.py            # ASRSession + BatchScheduler
  vad.py                   # Silero VAD wrapper
  hallucination.py         # Bengali hallucination filter
  audio_utils.py           # PCM16/float32 conversion
clients/
  test_client.py           # Python WebSocket client
  index.html               # Browser mic capture client
```
