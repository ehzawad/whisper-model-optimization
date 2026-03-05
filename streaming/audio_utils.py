"""Audio format conversion utilities for the streaming ASR pipeline."""

import numpy as np

WHISPER_SAMPLE_RATE = 16000


def pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    """Convert raw PCM 16-bit signed little-endian bytes to float32 in [-1, 1]."""
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] numpy array to PCM 16-bit LE bytes."""
    return (audio * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()


def resample_if_needed(
    audio: np.ndarray, orig_sr: int, target_sr: int = WHISPER_SAMPLE_RATE
) -> np.ndarray:
    """Resample audio array if sample rates differ. Uses librosa."""
    if orig_sr == target_sr:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def read_wav_file(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file and return (float32 numpy array, sample_rate)."""
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32")
    # Convert stereo to mono if needed
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr
