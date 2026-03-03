"""Silero VAD wrapper for streaming voice activity detection."""

import numpy as np
import torch
from silero_vad import load_silero_vad, VADIterator


class SileroVADWrapper:
    """Streaming VAD using Silero VAD model.

    Processes audio in 512-sample (32ms) windows at 16kHz.
    Tracks speech state and exposes boolean queries for the ASR engine.
    """

    WINDOW_SIZE = 512  # 32ms at 16kHz
    SAMPLE_RATE = 16000

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 30,
    ):
        self.model = load_silero_vad()
        self.iterator = VADIterator(
            self.model,
            threshold=threshold,
            sampling_rate=self.SAMPLE_RATE,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.is_speaking = False
        self._buffer = np.array([], dtype=np.float32)

    def feed(self, audio_chunk: np.ndarray) -> list[dict]:
        """Feed audio samples (float32, 16kHz) and return speech events.

        Returns list of event dicts:
          {"event": "speech_start", "sample_offset": int}
          {"event": "speech_end", "sample_offset": int}
        """
        self._buffer = np.concatenate([self._buffer, audio_chunk])
        events = []

        offset = 0
        while offset + self.WINDOW_SIZE <= len(self._buffer):
            window = self._buffer[offset : offset + self.WINDOW_SIZE]
            tensor = torch.from_numpy(window)

            speech_dict = self.iterator(tensor, return_seconds=False)

            if speech_dict:
                if "start" in speech_dict:
                    self.is_speaking = True
                    events.append(
                        {"event": "speech_start", "sample_offset": speech_dict["start"]}
                    )
                elif "end" in speech_dict:
                    self.is_speaking = False
                    events.append(
                        {"event": "speech_end", "sample_offset": speech_dict["end"]}
                    )

            offset += self.WINDOW_SIZE

        # Keep unprocessed remainder
        self._buffer = self._buffer[offset:]
        return events

    def reset(self):
        """Reset VAD state for a new audio stream."""
        self.iterator.reset_states()
        self.is_speaking = False
        self._buffer = np.array([], dtype=np.float32)
