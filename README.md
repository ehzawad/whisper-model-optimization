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

## Benchmarks

**Audio:** `2023-11-22T07_37_42...9b5a6935c52d.wav` — 947.3s (15m47s) Bengali meeting recording

```bash
AUDIO=meeting_22_11_23/single_audio/2023-11-22T07_37_42.518693Z_65f437bd-53d0-4ff2-a667-9b5a6935c52d.wav
python transcribe_naive.py $AUDIO                        # naive HF baseline
python transcribe.py $AUDIO                              # optimized HF (SDPA+batch)
python transcribe_fw.py $AUDIO                           # faster-whisper (CTranslate2)
```

### Tesla T4 (16GB VRAM)

| Approach | Chunk | Inference | RTF | Realtime | vs Naive |
|---|---|---|---|---|---|
| Naive HF pipeline | 15s | 290.5s | 0.307 | 3.3x | baseline |
| Naive HF pipeline | 30s | 283.2s | 0.299 | 3.3x | 1.0x |
| Optimized HF (SDPA+batch) | 15s | 23.7s | 0.025 | 39.9x | 12.3x |
| **Optimized HF (SDPA+batch)** | **30s** | **15.5s** | **0.016** | **61.2x** | **18.7x** |
| faster-whisper | 15s | 25.8s | 0.027 | 36.6x | 11.3x |
| **faster-whisper** | **30s** | **18.2s** | **0.019** | **52.2x** | **16.0x** |

### RTX 2050 (4GB VRAM)

| Approach | Chunk | Inference | RTF | Realtime | vs Naive |
|---|---|---|---|---|---|
| Naive HF pipeline | 30s | 494.3s | 0.522 | 1.9x | baseline |
| Optimized HF (SDPA+batch) | 30s | 76.6s | 0.081 | 12.4x | 6.5x |
| **faster-whisper** | **30s** | **37.9s** | **0.040** | **25.0x** | **13.0x** |

### Key observations

- **Naive HF pipeline** — `transformers.pipeline(chunk_length_s=N)`, sequential per-chunk decoding, `batch_size=1`. Chunk size barely matters (~2%).
- **Optimized HF** — fp16 + SDPA + batched chunks. Competitive with faster-whisper on T4 (batch_size=41), but falls behind on low-VRAM GPUs (batch_size=2).
- **faster-whisper** — CTranslate2 C++ kernels + Silero VAD. Wins on VRAM-constrained GPUs. Half the VRAM (~800MB vs ~1700MB).
- **15s vs 30s chunks** — ~40-50% slower (64 vs 32 decoder passes).
- **T4 vs RTX 2050** — T4's 320 GB/s bandwidth + 16GB VRAM delivers dramatically higher throughput.

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
