#!/usr/bin/env python3
"""Enhanced benchmark: original 4 methods + torch.compile + whisper-large-v3-turbo.

Runs all methods across all 254 clips + the single long audio.
Measures per-file inference time, VRAM, and cross-method transcript similarity.
"""

import gc
import glob
import json
import os
import time
from difflib import SequenceMatcher

import librosa
import numpy as np
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
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

TURBO_MODEL_ID = "openai/whisper-large-v3-turbo"


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
    """Load model+pipeline for a given method. Returns (pipe, gen_kwargs)."""

    if method_name in ("fp16", "fp16_compiled"):
        processor = WhisperProcessor.from_pretrained(MODEL_DIR)
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="bn", task="transcribe"
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_DIR, torch_dtype=torch.float16, attn_implementation="sdpa"
        ).to(DEVICE)
        model.generation_config.forced_decoder_ids = forced_decoder_ids

        if method_name == "fp16_compiled":
            model.forward = torch.compile(model.forward, mode="default")

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            device=DEVICE,
            torch_dtype=torch.float16,
        )
        gen_kwargs = {"forced_decoder_ids": forced_decoder_ids}
        return pipe, gen_kwargs

    elif method_name == "turbo_fp16":
        processor = AutoProcessor.from_pretrained(TURBO_MODEL_ID)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            TURBO_MODEL_ID,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        ).to(DEVICE)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            device=DEVICE,
            torch_dtype=torch.float16,
        )
        gen_kwargs = {"language": "bn", "task": "transcribe"}
        return pipe, gen_kwargs

    elif method_name == "turbo_compiled":
        processor = AutoProcessor.from_pretrained(TURBO_MODEL_ID)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            TURBO_MODEL_ID,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        ).to(DEVICE)
        model.forward = torch.compile(model.forward, mode="default")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            device=DEVICE,
            torch_dtype=torch.float16,
        )
        gen_kwargs = {"language": "bn", "task": "transcribe"}
        return pipe, gen_kwargs

    else:
        raise ValueError(f"Unknown method: {method_name}")


def transcribe_file(pipe, gen_kwargs, audio_path):
    t0 = time.perf_counter()
    result = pipe(audio_path, generate_kwargs=gen_kwargs)
    elapsed = time.perf_counter() - t0
    return result["text"].strip(), elapsed


def compute_similarity(text_a, text_b):
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

    individual_wavs = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
    print(f"Found {len(individual_wavs)} individual clips + 1 single long audio\n")

    durations = {}
    for wav in individual_wavs:
        durations[wav] = librosa.get_duration(path=wav)
    durations[SINGLE_AUDIO] = librosa.get_duration(path=SINGLE_AUDIO)
    total_audio_clips = sum(durations[w] for w in individual_wavs)

    methods = [
        "fp16",
        "fp16_compiled",
        "turbo_fp16",
        "turbo_compiled",
    ]

    all_results = {}

    for method in methods:
        print(f"\n{'='*60}", flush=True)
        print(f"METHOD: {method}", flush=True)
        print(f"{'='*60}", flush=True)

        reset_vram_stats()

        t_load_start = time.perf_counter()
        try:
            pipe, gen_kwargs = load_method(method)
        except Exception as e:
            print(f"  FAILED to load: {e}", flush=True)
            continue
        t_load = time.perf_counter() - t_load_start
        vram_after_load = get_vram_mib()

        print(f"  Model load time: {t_load:.1f}s", flush=True)
        print(f"  VRAM after load: {vram_after_load:.0f} MiB", flush=True)

        try:
            # Warmup (important for torch.compile)
            if torch.cuda.is_available():
                reset_vram_stats()

            warmup_count = 3 if "compiled" in method else 1
            print(f"  Warmup ({warmup_count} passes)...", flush=True)
            for _ in range(warmup_count):
                _ = transcribe_file(pipe, gen_kwargs, individual_wavs[0])

            reset_vram_stats()
            method_results = {}

            # Transcribe all individual clips
            total_inference_time = 0.0
            for i, wav in enumerate(individual_wavs):
                text, elapsed = transcribe_file(pipe, gen_kwargs, wav)
                method_results[wav] = {
                    "text": text,
                    "time": elapsed,
                    "duration": durations[wav],
                }
                total_inference_time += elapsed
                if (i + 1) % 50 == 0 or i == 0:
                    print(
                        f"  [{i+1}/{len(individual_wavs)}] "
                        f"last={elapsed:.2f}s  cumulative={total_inference_time:.1f}s",
                        flush=True,
                    )

            # Transcribe single long audio
            print(f"  Transcribing long audio ({durations[SINGLE_AUDIO]:.0f}s)...", flush=True)
            text, elapsed = transcribe_file(pipe, gen_kwargs, SINGLE_AUDIO)
            method_results[SINGLE_AUDIO] = {
                "text": text,
                "time": elapsed,
                "duration": durations[SINGLE_AUDIO],
            }

            vram_peak = get_vram_mib()

            print(
                f"  Clips: {total_inference_time:.1f}s for {total_audio_clips:.0f}s audio "
                f"(RTF={total_inference_time/total_audio_clips:.3f})",
                flush=True,
            )
            print(
                f"  Long:  {elapsed:.1f}s for {durations[SINGLE_AUDIO]:.0f}s audio "
                f"(RTF={elapsed/durations[SINGLE_AUDIO]:.3f})",
                flush=True,
            )
            print(f"  VRAM peak: {vram_peak:.0f} MiB", flush=True)

        except Exception as e:
            print(f"  FAILED during inference: {e}", flush=True)
            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        all_results[method] = {
            "results": method_results,
            "load_time": t_load,
            "vram_peak": vram_peak,
            "total_clips_inference": total_inference_time,
            "long_audio_inference": elapsed,
        }

        # Save transcripts
        with open(os.path.join(OUTPUT_DIR, f"transcripts_{method}.json"), "w") as f:
            serializable = {}
            for k, v in method_results.items():
                serializable[os.path.basename(k)] = v
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === COMPREHENSIVE REPORT ===
    active_methods = [m for m in methods if m in all_results]
    ref_method = "fp16"

    print(f"\n\n{'#'*70}")
    print("# COMPREHENSIVE BENCHMARK REPORT")
    print(f"# {len(individual_wavs)} clips ({total_audio_clips:.0f}s) + 1 long audio ({durations[SINGLE_AUDIO]:.0f}s)")
    print(f"# GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'#'*70}")

    # Table 1: Overall performance
    print("\n## 1. Overall Performance\n")
    header = f"{'Method':<16} {'Load(s)':<9} {'Clips(s)':<10} {'Long(s)':<9} {'ClipRTF':<9} {'LongRTF':<9} {'VRAM(MiB)':<10}"
    print(header)
    print("-" * len(header))
    fp16_clip = all_results[ref_method]["total_clips_inference"]
    fp16_long = all_results[ref_method]["long_audio_inference"]
    for m in active_methods:
        r = all_results[m]
        clip_rtf = r["total_clips_inference"] / total_audio_clips
        long_rtf = r["long_audio_inference"] / durations[SINGLE_AUDIO]
        print(
            f"{m:<16} {r['load_time']:<9.1f} {r['total_clips_inference']:<10.1f} "
            f"{r['long_audio_inference']:<9.1f} {clip_rtf:<9.3f} {long_rtf:<9.3f} "
            f"{r['vram_peak']:<10.0f}"
        )

    # Table 2: Relative speed
    print("\n## 2. Relative Speed (vs fp16 baseline)\n")
    print(f"{'Method':<16} {'ClipSpeedup':<14} {'LongSpeedup':<14} {'VRAM vs fp16':<14}")
    print("-" * 58)
    fp16_vram = all_results[ref_method]["vram_peak"]
    for m in active_methods:
        r = all_results[m]
        clip_speedup = fp16_clip / r["total_clips_inference"]
        long_speedup = fp16_long / r["long_audio_inference"]
        vram_delta = ((r["vram_peak"] - fp16_vram) / fp16_vram) * 100
        print(
            f"{m:<16} {clip_speedup:<14.2f}x {long_speedup:<14.2f}x "
            f"{vram_delta:>+12.1f}%"
        )

    # Table 3: Per-clip timing stats
    print("\n## 3. Per-Clip Inference Time Statistics (seconds)\n")
    print(f"{'Method':<16} {'Mean':<8} {'Median':<8} {'P95':<8} {'Max':<8} {'Min':<8}")
    print("-" * 56)
    for m in active_methods:
        times = [all_results[m]["results"][w]["time"] for w in individual_wavs]
        arr = np.array(times)
        print(
            f"{m:<16} {arr.mean():<8.3f} {np.median(arr):<8.3f} "
            f"{np.percentile(arr, 95):<8.3f} {arr.max():<8.3f} {arr.min():<8.3f}"
        )

    # Table 4: Per-speaker breakdown
    print("\n## 4. Per-Speaker Mean Inference Time (seconds)\n")
    speakers = ["Sazzad", "saiful", "Sunny"]
    print(f"{'Method':<16}", end="")
    for sp in speakers:
        print(f" {sp:<14}", end="")
    print()
    print("-" * 58)
    for m in active_methods:
        print(f"{m:<16}", end="")
        for sp in speakers:
            sp_wavs = [w for w in individual_wavs if sp in os.path.basename(w)]
            sp_times = [all_results[m]["results"][w]["time"] for w in sp_wavs]
            print(f" {np.mean(sp_times):<14.3f}", end="")
        print()

    # Table 5: Transcript similarity vs fp16
    print(f"\n## 5. Transcript Similarity vs fp16 (all 254 clips)\n")
    print(f"{'Method':<16} {'WordSim%':<10} {'CharSim%':<10} {'MeanLen':<10} {'EmptyClips':<12}")
    print("-" * 58)
    for m in active_methods:
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
        print(
            f"{m:<16} {np.mean(word_sims)*100:<10.1f} {np.mean(char_sims)*100:<10.1f} "
            f"{np.mean(lengths):<10.1f} {empties:<12}"
        )

    # Table 6: Long audio similarity
    print(f"\n## 6. Long Audio (15.8 min) Similarity vs fp16\n")
    print(f"{'Method':<16} {'WordSim%':<10} {'CharSim%':<10} {'Length':<10}")
    print("-" * 46)
    for m in active_methods:
        ref_long = all_results[ref_method]["results"][SINGLE_AUDIO]["text"]
        cmp_long = all_results[m]["results"][SINGLE_AUDIO]["text"]
        ws, cs = compute_similarity(ref_long, cmp_long)
        print(f"{m:<16} {ws*100:<10.1f} {cs*100:<10.1f} {len(cmp_long):<10}")

    # Table 7: Full pairwise similarity matrix
    print(f"\n## 7. Pairwise Word Similarity Matrix (avg across 254 clips)\n")
    print(f"{'':16}", end="")
    for m in active_methods:
        print(f" {m[:13]:<14}", end="")
    print()
    for m1 in active_methods:
        print(f"{m1:<16}", end="")
        for m2 in active_methods:
            sims = []
            for w in individual_wavs:
                t1 = all_results[m1]["results"][w]["text"]
                t2 = all_results[m2]["results"][w]["text"]
                ws, _ = compute_similarity(t1, t2)
                sims.append(ws)
            print(f" {np.mean(sims)*100:>12.1f}%", end="")
        print()

    # Table 8: Speed-accuracy tradeoff
    print(f"\n## 8. Speed-Accuracy Trade-off Summary\n")
    print(f"{'Method':<16} {'TotalTime(s)':<14} {'RelSpeed':<10} {'AvgWordSim%':<12} {'VRAM(MiB)':<10} {'Model':<12}")
    print("-" * 74)
    fp16_total = fp16_clip + fp16_long
    for m in active_methods:
        r = all_results[m]
        total_t = r["total_clips_inference"] + r["long_audio_inference"]
        rel_speed = fp16_total / total_t
        word_sims = []
        for w in individual_wavs:
            ref_text = all_results[ref_method]["results"][w]["text"]
            cmp_text = all_results[m]["results"][w]["text"]
            ws, _ = compute_similarity(ref_text, cmp_text)
            word_sims.append(ws)
        model_type = "turbo-v3" if "turbo" in m else "ft-medium"
        print(
            f"{m:<16} {total_t:<14.1f} {rel_speed:<10.2f}x {np.mean(word_sims)*100:<12.1f} "
            f"{r['vram_peak']:<10.0f} {model_type:<12}"
        )

    # Table 9: Worst divergent clips
    print(f"\n## 9. Top 5 Most Divergent Clips per Method (vs fp16)\n")
    for m in active_methods:
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
            if ref_t:
                print(f"      fp16: {ref_t[:70]}...")
            if cmp_t:
                print(f"      {m[:8]}: {cmp_t[:70]}...")

    # Table 10: Per-method efficiency (time per second of audio)
    print(f"\n## 10. Efficiency: Seconds of Compute per Second of Audio\n")
    print(f"{'Method':<16} {'Clips':<10} {'Long':<10} {'Overall':<10}")
    print("-" * 46)
    for m in active_methods:
        r = all_results[m]
        clip_eff = r["total_clips_inference"] / total_audio_clips
        long_eff = r["long_audio_inference"] / durations[SINGLE_AUDIO]
        total_compute = r["total_clips_inference"] + r["long_audio_inference"]
        total_audio = total_audio_clips + durations[SINGLE_AUDIO]
        overall_eff = total_compute / total_audio
        print(f"{m:<16} {clip_eff:<10.4f} {long_eff:<10.4f} {overall_eff:<10.4f}")

    # Save summary JSON
    summary = {
        "metadata": {
            "num_clips": len(individual_wavs),
            "total_clip_audio_s": total_audio_clips,
            "long_audio_s": durations[SINGLE_AUDIO],
            "speakers": {"Sazzad": 90, "saiful": 69, "Sunny": 95},
            "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
            "torch_version": torch.__version__,
        },
        "methods": {},
    }
    for m in active_methods:
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

    with open(os.path.join(OUTPUT_DIR, "enhanced_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
