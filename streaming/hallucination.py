"""Hallucination filter for Bengali Whisper output."""

from dataclasses import dataclass

# Known Bengali hallucination phrases Whisper produces on silence/noise.
# These are empirically common across Bengali Whisper models fine-tuned on YouTube data.
BENGALI_HALLUCINATION_PHRASES = [
    "সাবস্ক্রাইব করুন",
    "ধন্যবাদ সবাইকে",
    "আমার চ্যানেলটি সাবস্ক্রাইব",
    "লাইক কমেন্ট শেয়ার",
    "ভিডিওটি ভালো লাগলে",
    "দেখতে থাকুন",
    "পরবর্তী ভিডিওতে",
    "আসসালামু আলাইকুম",
    "...",
]


@dataclass
class SegmentScore:
    """Filtered segment with reason if rejected."""

    text: str
    accepted: bool
    reason: str = ""


class HallucinationFilter:
    """Filter hallucinated Whisper segments for Bengali.

    Heuristics:
    1. no_speech_prob threshold (Whisper thinks it's silence)
    2. avg_logprob threshold (low model confidence)
    3. Known hallucination phrase matching
    4. Repetition detection (same word N times)
    5. Empty text
    """

    def __init__(
        self,
        no_speech_prob_threshold: float = 0.6,
        avg_logprob_threshold: float = -1.0,
        repetition_threshold: int = 3,
        custom_phrases: list[str] | None = None,
    ):
        self.no_speech_prob_threshold = no_speech_prob_threshold
        self.avg_logprob_threshold = avg_logprob_threshold
        self.repetition_threshold = repetition_threshold

        self.hallucination_phrases = set(BENGALI_HALLUCINATION_PHRASES)
        if custom_phrases:
            self.hallucination_phrases.update(custom_phrases)

    def check_segment(
        self,
        text: str,
        no_speech_prob: float,
        avg_logprob: float,
    ) -> SegmentScore:
        """Evaluate a single transcription segment."""
        text_stripped = text.strip()

        # 1. No-speech probability
        if no_speech_prob > self.no_speech_prob_threshold:
            return SegmentScore(
                text=text_stripped,
                accepted=False,
                reason=f"no_speech_prob={no_speech_prob:.2f}",
            )

        # 2. Average log probability
        if avg_logprob < self.avg_logprob_threshold:
            return SegmentScore(
                text=text_stripped,
                accepted=False,
                reason=f"avg_logprob={avg_logprob:.2f}",
            )

        # 3. Known hallucination phrase
        for phrase in self.hallucination_phrases:
            if phrase in text_stripped:
                return SegmentScore(
                    text=text_stripped,
                    accepted=False,
                    reason=f"hallucination_phrase: '{phrase}'",
                )

        # 4. Repetition detection
        words = text_stripped.split()
        if len(words) >= self.repetition_threshold:
            for i in range(len(words) - self.repetition_threshold + 1):
                window = words[i : i + self.repetition_threshold]
                if len(set(window)) == 1:
                    return SegmentScore(
                        text=text_stripped,
                        accepted=False,
                        reason=f"repetition: '{window[0]}' x{self.repetition_threshold}",
                    )

        # 5. Empty text
        if not text_stripped:
            return SegmentScore(text="", accepted=False, reason="empty_text")

        return SegmentScore(text=text_stripped, accepted=True)
