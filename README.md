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

## Benchmark: Naive HF Pipeline vs faster-whisper

**Audio:** 947.3s (15m47s) Bengali meeting | **GPU:** RTX 2050 (4GB VRAM)

| Approach | Chunk | Inference | RTF | Throughput | Speedup |
|---|---|---|---|---|---|
| Naive HF pipeline | 15s | 501.6s | 0.530 | 1.9x | baseline (1.0x) |
| Naive HF pipeline | 30s | 497.9s | 0.526 | 1.9x | 1.0x |
| **faster-whisper** | **15s** | **55.7s** | **0.059** | **17.0x** | **9.0x** |
| **faster-whisper** | **30s** | **39.4s** | **0.042** | **24.0x** | **12.6x** |

15s chunks are slower because more chunks = more decoder passes (63 vs 32 chunks). Each chunk incurs fixed encoder + decoder overhead regardless of speech content.

## Architecture

```
transcribe_fw.py           # faster-whisper (CTranslate2) inference
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
