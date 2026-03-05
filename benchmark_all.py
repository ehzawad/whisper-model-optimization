#!/usr/bin/env python3
"""Comprehensive benchmark: 4 inference methods × all audio files.

Measures per-file inference time, VRAM usage, and cross-method transcript similarity.
"""

import gc
import glob
import json
import os
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import (
    BitsAndBytesConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(MODEL_DIR, "meeting_22_11_23")
SINGLE_AUDIO = os.path.join(
    AUDIO_DIR,
    "single_audio",
    "2023-11-22T07_37_42.518693Z_65f437bd-53d0-4ff2-a667-9b5a6935c52d.wav",
)
OUTPUT_DIR = os.path.join(MODEL_DIR, "benchmark_results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.enabled = False


def get_vram_mib():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_vram_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def load_method(method_name):
    """Load model+pipeline for a given method. Returns (pipe, forced_decoder_ids)."""
    processor = WhisperProcessor.from_pretrained(MODEL_DIR)
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="bn", task="transcribe"
    )

    if method_name == "fp16":
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_DIR, torch_dtype=torch.float16, attn_implementation="sdpa"
        ).to(DEVICE)

    elif method_name == "bnb_int8":
        qconfig = BitsAndBytesConfig(load_in_8bit=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            quantization_config=qconfig,
            attn_implementation="sdpa",
            device_map="auto",
        )

    elif method_name == "bnb_int4":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4"
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            quantization_config=qconfig,
            attn_implementation="sdpa",
            device_map="auto",
        )

    elif method_name == "quanto_int8":
        from transformers import QuantoConfig

        qconfig = QuantoConfig(weights="int8")
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            quantization_config=qconfig,
            attn_implementation="sdpa",
        ).to(DEVICE)

    model.generation_config.forced_decoder_ids = forced_decoder_ids

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        device=model.device if method_name in ("bnb_int8", "bnb_int4") else DEVICE,
        torch_dtype=torch.float16,
    )

    return pipe, forced_decoder_ids


def transcribe_file(pipe, forced_decoder_ids, audio_path):
    """Transcribe a single file, return (text, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = pipe(
        audio_path,
        generate_kwargs={"forced_decoder_ids": forced_decoder_ids},
    )
    elapsed = time.perf_counter() - t0
    return result["text"].strip(), elapsed


def compute_similarity(text_a, text_b):
    """Return (word_similarity, char_similarity)."""
    if not text_a and not text_b:
        return 1.0, 1.0
    if not text_a or not text_b:
        return 0.0, 0.0
    words_a, words_b = text_a.split(), text_b.split()
    word_sim = SequenceMatcher(None, words_a, words_b).ratio()
    char_sim = SequenceMatcher(None, text_a, text_b).ratio()
    return word_sim, char_sim


def get_speaker(filename):
    base = os.path.basename(filename)
    for sp in ["Sazzad", "saiful", "Sunny"]:
        if sp in base:
            return sp
    return "unknown"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all individual audio files
    individual_wavs = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
    print(f"Found {len(individual_wavs)} individual clips + 1 single long audio")

    # Get durations
    durations = {}
    for wav in individual_wavs:
        durations[wav] = librosa.get_duration(path=wav)
    durations[SINGLE_AUDIO] = librosa.get_duration(path=SINGLE_AUDIO)

    methods = ["fp16", "bnb_int8", "bnb_int4", "quanto_int8"]
    all_results = {}  # method -> {filepath: {text, time, duration}}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"METHOD: {method}")
        print(f"{'='*60}")

        reset_vram_stats()

        t_load_start = time.perf_counter()
        pipe, forced_decoder_ids = load_method(method)
        t_load = time.perf_counter() - t_load_start
        vram_after_load = get_vram_mib()

        print(f"  Model load time: {t_load:.1f}s")
        print(f"  VRAM after load: {vram_after_load:.0f} MiB")

        # Warmup pass
        if torch.cuda.is_available():
            reset_vram_stats()
        _ = transcribe_file(pipe, forced_decoder_ids, individual_wavs[0])

        reset_vram_stats()
        method_results = {}

        # Transcribe all individual clips
        total_inference_time = 0.0
        for i, wav in enumerate(individual_wavs):
            text, elapsed = transcribe_file(pipe, forced_decoder_ids, wav)
            method_results[wav] = {
                "text": text,
                "time": elapsed,
                "duration": durations[wav],
            }
            total_inference_time += elapsed
            if (i + 1) % 50 == 0 or i == 0:
                print(
                    f"  [{i+1}/{len(individual_wavs)}] "
                    f"last={elapsed:.2f}s  cumulative={total_inference_time:.1f}s"
                )

        # Transcribe the single long audio
        print(f"  Transcribing single long audio ({durations[SINGLE_AUDIO]:.0f}s)...")
        text, elapsed = transcribe_file(pipe, forced_decoder_ids, SINGLE_AUDIO)
        method_results[SINGLE_AUDIO] = {
            "text": text,
            "time": elapsed,
            "duration": durations[SINGLE_AUDIO],
        }

        vram_peak = get_vram_mib()
        total_audio_s = sum(durations[w] for w in individual_wavs)

        print(f"  Individual clips: {total_inference_time:.1f}s inference "
              f"for {total_audio_s:.0f}s audio (RTF={total_inference_time/total_audio_s:.3f})")
        print(f"  Long audio: {elapsed:.1f}s inference "
              f"for {durations[SINGLE_AUDIO]:.0f}s audio (RTF={elapsed/durations[SINGLE_AUDIO]:.3f})")
        print(f"  VRAM peak: {vram_peak:.0f} MiB")

        all_results[method] = {
            "results": method_results,
            "load_time": t_load,
            "vram_peak": vram_peak,
            "total_clips_inference": total_inference_time,
            "long_audio_inference": elapsed,
        }

        # Save per-method transcripts
        with open(os.path.join(OUTPUT_DIR, f"transcripts_{method}.json"), "w") as f:
            serializable = {}
            for k, v in method_results.items():
                serializable[os.path.basename(k)] = v
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        # Unload
        del pipe
        del _
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === ANALYSIS ===
    print(f"\n\n{'='*70}")
    print("COMPREHENSIVE ANALYSIS")
    print(f"{'='*70}")

    # 1. Per-method summary
    ref_method = "fp16"
    total_audio_clips = sum(durations[w] for w in individual_wavs)

    print("\n## 1. Overall Performance\n")
    print(f"{'Method':<14} {'Load(s)':<9} {'Clips(s)':<10} {'Long(s)':<9} "
          f"{'ClipRTF':<9} {'LongRTF':<9} {'VRAM(MiB)':<10}")
    print("-" * 70)
    for m in methods:
        r = all_results[m]
        clip_rtf = r["total_clips_inference"] / total_audio_clips
        long_rtf = r["long_audio_inference"] / durations[SINGLE_AUDIO]
        print(
            f"{m:<14} {r['load_time']:<9.1f} {r['total_clips_inference']:<10.1f} "
            f"{r['long_audio_inference']:<9.1f} {clip_rtf:<9.3f} {long_rtf:<9.3f} "
            f"{r['vram_peak']:<10.0f}"
        )

    # 2. Per-file timing statistics
    print("\n## 2. Per-Clip Inference Time Statistics (seconds)\n")
    print(f"{'Method':<14} {'Mean':<8} {'Median':<8} {'P95':<8} {'Max':<8} {'Min':<8} {'Std':<8}")
    print("-" * 62)
    for m in methods:
        times = [all_results[m]["results"][w]["time"] for w in individual_wavs]
        arr = np.array(times)
        print(
            f"{m:<14} {arr.mean():<8.3f} {np.median(arr):<8.3f} "
            f"{np.percentile(arr, 95):<8.3f} {arr.max():<8.3f} "
            f"{arr.min():<8.3f} {arr.std():<8.3f}"
        )

    # 3. Per-speaker breakdown
    print("\n## 3. Per-Speaker Average Inference Time (seconds)\n")
    speakers = ["Sazzad", "saiful", "Sunny"]
    print(f"{'Method':<14}", end="")
    for sp in speakers:
        print(f" {sp:<16}", end="")
    print()
    print("-" * 62)
    for m in methods:
        print(f"{m:<14}", end="")
        for sp in speakers:
            sp_wavs = [w for w in individual_wavs if sp in os.path.basename(w)]
            sp_times = [all_results[m]["results"][w]["time"] for w in sp_wavs]
            avg = np.mean(sp_times) if sp_times else 0
            print(f" {avg:<16.3f}", end="")
        print()

    # 4. Cross-method similarity (using fp16 as reference)
    print(f"\n## 4. Transcript Similarity vs fp16 Reference\n")
    print(f"{'Method':<14} {'WordSim%':<10} {'CharSim%':<10} {'MeanLen':<10} {'EmptyClips':<12}")
    print("-" * 56)
    for m in methods:
        word_sims, char_sims, lengths, empties = [], [], [], 0
        for w in individual_wavs:
            ref_text = all_results[ref_method]["results"][w]["text"]
            cmp_text = all_results[m]["results"][w]["text"]
            if not cmp_text:
                empties += 1
            ws, cs = compute_similarity(ref_text, cmp_text)
            word_sims.append(ws)
            char_sims.append(cs)
            lengths.append(len(cmp_text))

        # Also for long audio
        ref_long = all_results[ref_method]["results"][SINGLE_AUDIO]["text"]
        cmp_long = all_results[m]["results"][SINGLE_AUDIO]["text"]
        ws_long, cs_long = compute_similarity(ref_long, cmp_long)

        print(
            f"{m:<14} {np.mean(word_sims)*100:<10.1f} {np.mean(char_sims)*100:<10.1f} "
            f"{np.mean(lengths):<10.1f} {empties:<12}"
        )

    # Long audio similarity
    print(f"\n## 5. Long Audio (15.8 min) Similarity vs fp16\n")
    print(f"{'Method':<14} {'WordSim%':<10} {'CharSim%':<10} {'Length':<10}")
    print("-" * 44)
    for m in methods:
        ref_long = all_results[ref_method]["results"][SINGLE_AUDIO]["text"]
        cmp_long = all_results[m]["results"][SINGLE_AUDIO]["text"]
        ws, cs = compute_similarity(ref_long, cmp_long)
        print(f"{m:<14} {ws*100:<10.1f} {cs*100:<10.1f} {len(cmp_long):<10}")

    # 6. Pairwise similarity matrix (all methods)
    print(f"\n## 6. Pairwise Word Similarity Matrix (avg across all clips)\n")
    print(f"{'':14}", end="")
    for m in methods:
        print(f" {m:<14}", end="")
    print()
    for m1 in methods:
        print(f"{m1:<14}", end="")
        for m2 in methods:
            sims = []
            for w in individual_wavs:
                t1 = all_results[m1]["results"][w]["text"]
                t2 = all_results[m2]["results"][w]["text"]
                ws, _ = compute_similarity(t1, t2)
                sims.append(ws)
            print(f" {np.mean(sims)*100:>12.1f}%", end="")
        print()

    # 7. Speed vs accuracy scatter data
    print(f"\n## 7. Speed-Accuracy Trade-off Summary\n")
    print(f"{'Method':<14} {'TotalTime(s)':<14} {'RelSpeed':<10} {'AvgWordSim%':<12} {'VRAM(MiB)':<10}")
    print("-" * 60)
    fp16_total = (all_results["fp16"]["total_clips_inference"]
                  + all_results["fp16"]["long_audio_inference"])
    for m in methods:
        total_t = (all_results[m]["total_clips_inference"]
                   + all_results[m]["long_audio_inference"])
        rel_speed = fp16_total / total_t
        word_sims = []
        for w in individual_wavs:
            ref_text = all_results[ref_method]["results"][w]["text"]
            cmp_text = all_results[m]["results"][w]["text"]
            ws, _ = compute_similarity(ref_text, cmp_text)
            word_sims.append(ws)
        avg_ws = np.mean(word_sims) * 100
        print(
            f"{m:<14} {total_t:<14.1f} {rel_speed:<10.2f}x {avg_ws:<12.1f} "
            f"{all_results[m]['vram_peak']:<10.0f}"
        )

    # 8. Worst-case clips per method (highest divergence from fp16)
    print(f"\n## 8. Top 5 Most Divergent Clips per Method (vs fp16)\n")
    for m in methods:
        if m == ref_method:
            continue
        divergences = []
        for w in individual_wavs:
            ref_text = all_results[ref_method]["results"][w]["text"]
            cmp_text = all_results[m]["results"][w]["text"]
            ws, cs = compute_similarity(ref_text, cmp_text)
            divergences.append((os.path.basename(w), ws, cs, ref_text, cmp_text))
        divergences.sort(key=lambda x: x[1])

        print(f"\n  {m}:")
        for fname, ws, cs, ref_t, cmp_t in divergences[:5]:
            print(f"    {fname}: word={ws*100:.1f}% char={cs*100:.1f}%")
            print(f"      fp16:  {ref_t[:80]}...")
            print(f"      {m}: {cmp_t[:80]}...")

    # Save full results as JSON
    summary = {
        "metadata": {
            "num_clips": len(individual_wavs),
            "total_clip_audio_s": total_audio_clips,
            "long_audio_s": durations[SINGLE_AUDIO],
            "device": str(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"),
        },
        "methods": {},
    }
    for m in methods:
        r = all_results[m]
        clip_times = [r["results"][w]["time"] for w in individual_wavs]
        word_sims = []
        for w in individual_wavs:
            ref_text = all_results[ref_method]["results"][w]["text"]
            cmp_text = all_results[m]["results"][w]["text"]
            ws, _ = compute_similarity(ref_text, cmp_text)
            word_sims.append(ws)

        summary["methods"][m] = {
            "load_time_s": r["load_time"],
            "vram_peak_mib": r["vram_peak"],
            "clips_total_inference_s": r["total_clips_inference"],
            "long_audio_inference_s": r["long_audio_inference"],
            "clip_rtf": r["total_clips_inference"] / total_audio_clips,
            "long_rtf": r["long_audio_inference"] / durations[SINGLE_AUDIO],
            "clip_mean_time_s": float(np.mean(clip_times)),
            "clip_median_time_s": float(np.median(clip_times)),
            "clip_p95_time_s": float(np.percentile(clip_times, 95)),
            "avg_word_similarity_vs_fp16": float(np.mean(word_sims)),
        }

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n\nResults saved to {OUTPUT_DIR}/")
    print("  - summary.json (aggregate metrics)")
    print("  - transcripts_<method>.json (per-file transcripts + timing)")


if __name__ == "__main__":
    main()
