# Speaker diarization candidates on the 54 s shop clip, 2026-11-26 (Mac M2 Pro, CPU)

Question: which `audio.diarization` candidate gives speaker slots that keep the enrolled person apart from the
other voices, live? The owner's decisions require this
([realtime-attribution-decisions.md](../realtime-attribution-decisions.md), decision 3: a word belongs to the slot
active when it was said).

There is no diarization truth for this clip. The owner's word labels stand in for it (see "Proxy" below). This is
one clip, so nothing here is a held-out accuracy, and no setting was tuned on it. Every candidate ran with its
published default settings and a 0.5 activity threshold.

## Candidates (state of the art as of 2026-11, with sources)

| candidate (registry `audio.diarization`) | live? | published evidence | licence |
|---|---|---|---|
| `nemotron3`: nvidia/Nemotron-3-Diarization, 8 slots | live (0.08-0.32-1.04 s buffers) | NVIDIA, arXiv [2611.12246](https://arxiv.org/abs/2611.12246): lowest DER on DIHARD III, CALLHOME, AliMeeting, AMI, NOTSOFAR-1 at 30.4 / 1.04 / 0.32 s, better than Sortformer-v2.1 and offline DiariZen (authors' numbers; training includes 21 languages). Independent: [Voice Arena Diarization-Bench v1](https://voicearena.com/diarization-bench-v1-launch) (2026-09-16, English only, batch): 14.7 DER, rank 1 | weights OpenMDW-1.1 (declared) |
| `sortformer.v2_1`: nvidia/diar_streaming_sortformer_4spk-v2.1, 4 slots | live | [model card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) (it names Nemotron-3 as its successor); arXiv [2507.18446](https://arxiv.org/abs/2507.18446); Voice Arena (v2 checkpoint): 24.6 DER | NVIDIA Open Model License; NeMo Apache-2.0 |
| `pyannoteai.live1.hosted`: pyannoteAI Live-1, 8 speakers, WebSocket | live, hosted | [docs](https://docs.pyannote.ai/tutorials/streaming-real-time), [launch post](https://www.pyannote.ai/blog/introducing-live-1-streaming-diarization) (July 2026, "sub-300 ms"); no published DER found | pyannoteAI terms; audio leaves the device (EEA servers); 0.198 EUR/h (Developer plan) |
| `pyannoteai.precision3.offline`: Precision-3 | **reference, not live** | [pyannoteAI benchmarks](https://www.pyannote.ai/blog/precision-3-benchmarks): 14.35 average DER on 15 sets (their numbers); Voice Arena: 20.6, rank 3 | pyannoteAI terms; 0.112 EUR/h |
| `diarizen.large_s80_v2`: BUT-FIT DiariZen-Large-s80-v2 | **reference, not live** | [GitHub](https://github.com/BUTSpeechFIT/DiariZen): AliMeeting-far 10.8, AMI-SDM 13.9, MSDWild 15.8, RAMC 11.0, DIHARD3 14.5 DER; Voice Arena: 19.3, rank 2 (best open model after Nemotron-3) | **weights CC BY-NC 4.0** (opt-in `allow_noncommercial`); code MIT; WeSpeaker ResNet34 embedding and PLDA CC BY 4.0 |
| `pyannote.community1` (local) / `pyannoteai.community1.offline` (hosted) | **reference, not live** | [model card](https://huggingface.co/pyannote/speaker-diarization-community-1): better than pyannote 3.1, worse than Precision-2 on all 12 listed sets; Voice Arena: 30.6 | CC BY 4.0, **gated** on Hugging Face; hosted 0.035 EUR/h |

Checked and not added: FASTDIAR (arXiv [2610.02941](https://arxiv.org/abs/2610.02941), Oct 2026, streaming, no
weights or code published); diart / pyannote segmentation-3.0 online (2021-2024 systems, beaten by the above);
moss_transcribe_diarize in Transformers 5.23 (ASR and diarization in one model, offline). The
[Benchmarking Diarization Models](https://arxiv.org/abs/2509.26177) study (2025) also ranks DiariZen first among
open systems and pyannoteAI best overall.

## Proxy

- Person's words (owner): ends 12.04, 14.0, 19.74, 20.14, 21.46, 22.30, 28.10, 38.20, 46.64, 47.74 s. Spans are the
  Whisper-turbo word times of the earlier run (`final_turbo`) with those ends; 14.0 s was never transcribed, so
  13.6-14.0 s is assumed.
- Other voices' words: 12.52-12.64, 20.14-20.34, 20.34-20.74, 27.06-27.44, 49.44-49.86 s, plus the stretch 49-52 s
  (another person).
- A slot is *active over a word* when it is active for at least 50 % of the word's span (30 % for the 49-52 s stretch).
- **Best slot** is the slot active over most of the person's words. **P** is how many of the person's 10 words it holds;
  **O** is how many of the 5 other-voice words it holds too; **O only** is how many it holds with no other slot
  active; **R** is its share of 49-52 s. A pure person slot is P 10/10, O 0/5, R 0. (The strict test "a person word
  with a slot no other word has" gives 0/10 for every candidate: 12.64 and 27.44 s overlap the person.)

## Results

| candidate, mode | best slot: P / O / O only / R | slots over 19.4-20.8 s (person 19.38-20.14, others 20.14-20.74) | live lag median / p95 | compute / audio | peak RSS |
|---|---|---|---|---|---|
| nemotron3 ultra_low_latency (80 ms look-ahead) | 5/10, 2/5, 1/5, 0.07 | 2, 0+2 (19.47-19.68), 0 (19.68-20.10), none, 2 (20.25-20.80) | 0.39 / 2.86 s | 0.88 | 1.55 GB |
| nemotron3 very_low_latency (160 ms) | 4/10, 2/5, 1/5, 0.06 | like low_latency, switch at 20.11 / 20.26 | 0.45 / 0.56 s | 0.45 | 1.53 GB |
| nemotron3 low_latency (320 ms) | 4/10, 2/5, 1/5, 0.00 | 2, 0+2 (19.44-19.66), 0 (19.66-20.10), none, 2 (20.26-20.80) | 0.65 / 0.77 s | 0.37 | 1.54 GB |
| sortformer.v2_1 ultra_low_latency (80 ms) | 7/10, 3/5, 2/5, 0.27 | 0 (with 1 at times), no switch | 29.7 / 65.8 s (fell behind) | 2.29 | 1.69 GB |
| sortformer.v2_1 low_latency (560 ms) | 10/10, 5/5, 4/5, 0.95 | 0+1 (19.40-19.84), 0 (19.84-20.80) | 17.1 / 23.2 s (fell behind) | 1.42 | 1.72 GB |
| pyannoteai.live1.hosted | 6/10, 3/5, 2/5, 0.00 | 1+2 (19.40-19.72), 1 (19.72-20.80) | 0.71 / 0.82 s (first record 5.5 s after opening) | hosted | 51 MB (client) |
| **pyannoteai.precision3.offline** | **10/10, 2/5, 0/5, 0.00** | 0+1 (19.40-19.68), **1 (19.68-20.14)**, none, 0 (20.20-20.80) | offline: 12.7 s for the whole clip, upload included | hosted | 70 MB (client) |
| pyannoteai.community1.offline | 10/10, 5/5, 4/5, 1.00 | 0, no switch | offline: 19.4 s | hosted | 62 MB (client) |
| **diarizen.large_s80_v2** | **10/10, 2/5, 0/5, 0.00** | 0+1 (19.40-19.73), **1 (19.73-20.13)**, 0 (20.13-20.80) | offline | 1.22 | 10.9 GB |

Per word (slots active over ≥ 30 % of the span; person's words first, then other voices'):

| candidate | 12.04 | 14.0 | 19.74 | 20.14 | 21.46 | 22.30 | 28.10 | 38.20 | 46.64 | 47.74 | other 12.64 | 20.34 | 20.74 | 27.44 | 49.86 | 49-52 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nemotron3 ultra_low | - | 1 | 2+0 | 0 | 1 | 1 | 0+2+1 | 2 | 2 | 2 | 0 | 2 (45 %) | 2 | 0+2 | 1 | 1 (54 %) |
| nemotron3 low | 0 | 1+0 (32 %) | 0+2 | 0 | 1 | 1 | 0+1 | 2 | 2 | 2 | 0 | 2 (40 %) | 2 | 0+2 | 1 | 1 (80 %) |
| sortformer low | 0 | 0 | 0+1 | 0 | 0 | 0 | 0+1 | 0 | 0 | 0 | 0 | 0 | 0 | 0+1 | 0 | 0 (95 %) |
| Live-1 | 1 | 1 | 1+2 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1+2 | 1 | 1 | 0 | 0 | 0 (83 %) |
| Precision-3 | 1 | 1 | 1+0 | 1 | 1 | 1 | 0+1 | 1 | 1 | 1 | 0+1 | 0 (70 %) | 0 | 0+1 | 0 | 0 (87 %) |
| DiariZen | 1 | 0+1 (57 %) | 1+0 | 1 | 1 | 1 | 0+1 | 1 | 1 | 1 | 0+1 | 0 | 0 | 0+1 | 0 | 0 (89 %) |

(very_low_latency matches low except 12.04 "-", 14.0 1+0 (30 %) and 22.30 1+0. Sortformer ultra_low: person 1+0, 0, 0+1, 0, 0, 0,
0+1, 0, 1, 1; others 1+0, 0, 0, 0+1, 1; 49-52 s 1 (49 %). Community-1: every word in slot 0, with slot 1 too at 14.0, 19.74, 28.10
and 27.44.)

## What this says

- **No live candidate kept the person in one slot of their own on this clip.** Nemotron-3 used three slots, and each
  held both the person and other voices. Live-1 put the person in slot 1 until 23 s and in slot 0 after that, and the
  other voices went into the same two slots. That is the same split pyannoteAI Precision-2 made earlier
  ([overlap-2026-11-26.md](overlap-2026-11-26.md)): an excited voice early on and a calm one later. Sortformer
  heard about one speaker. With decision 3 as written ("a word belongs to the slot active at its capture time"),
  each of these live models puts other voices' words (at least two of 12.64, 20.34, 20.74, 27.44 s) in a slot
  that also holds the person's words.
- **The two offline references agree with the owner.** Precision-3 and DiariZen each held all 10 of the person's
  words in one slot. The other voices' words in that slot (12.64 and 27.44 s) were said over the person, with both
  slots active, and 49-52 s stayed out of it. In the 1 s exchange at 19.4-20.8 s both had both slots active over the person's
  first word, the person's slot alone over the second, and the other slot from 20.13-20.20 s: the owner's boundary is
  20.14 s.
- Nemotron-3 also switched inside that exchange (its slot 0 for the person until 20.10 s, slot 2 for the other voice
  from 20.25 s). Locally that is right, but across the clip slot 2 is later the person's (38-48 s): its slots are
  consistent for a few seconds, not for the session. Live-1 and Sortformer did not switch there.
- Speed on this Mac CPU, with other agents' jobs running at the same time (load average 4-6): Nemotron-3 stayed
  within real time at 160 and 320 ms look-ahead (compute 0.45 and 0.37 of the audio). At 80 ms it fell behind at
  times (p95 lag 2.9 s). Sortformer-v2.1 is slower than real time on CPU in both modes. Its published real-time
  factors are GPU numbers.
- A live default therefore needs a check beyond diarization: the slot-identity step of decision 3 (each slot's
  accumulated sound against the few-shot voice, updated over time) has to split a slot once it is seen to hold
  two voices. Choosing slots alone will not do it here. The offline references are what the verification lane can
  revise against (decision 1).

## Correctness checks

- Sortformer adapter vs NeMo 3.0.0's own `forward_streaming` over the whole recording (first 20 s of the clip): every
  frame probability is identical (maximum absolute difference 0.0) in both modes. `tests/test_sortformer.py` checks
  that the adapter's chunk framing is the one NeMo's loader uses, with a model double.
- The Live-1 adapter commits a chunk only up to the server's `last_predicted_timestamp` (no earlier event follows),
  sends exactly 100 ms float32 messages, and pads the last one with silence (`tests/test_slot_sources.py`).
- Weight files were hashed after download and match the publishers' LFS SHA256: Sortformer .nemo `8abd3283...`,
  DiariZen `689a10f0...`, WeSpeaker ResNet34 `366edf44...`, PLDA `9b77bcd8...` / `325f1ce8...`.

## Cost

pyannoteAI, free trial, list prices (pyannote.ai/pricing, Developer plan, 2026-11-26): one Live-1 stream of
54.4 s (0.0030 EUR), one Precision-3 job of 54.3 s (0.0017 EUR), one Community-1 job of 54.3 s (0.0005 EUR).
The credit actually used was not checked.

## Reproduce

`scripts/qa_diarization.py run OUT.json MODULE:FACTORY '{kwargs}' --wav try-it/testbed/a.wav --pace` in the
candidate's environment (local models through `try-it/guard.sh`), then `scripts/qa_diarization.py score OUT.json`.
Factories: `gamst.providers.diarization:NemotronDiarization.from_local`,
`gamst.providers.sortformer:SortformerDiarization.from_local`, `gamst.providers.slot_sources:hosted_live`,
`:hosted_offline` and `:local_offline`. Environments: `nemotron3` (Transformers 5.23.0, torch 2.8.0); `sortformer`
(nemo_toolkit[asr] 3.0.0, torch 2.8.0, Python 3.11); `diarizen` (DiariZen @ 844f555 with its pyannote-audio 3.1.1
fork, torch 2.1.1, numpy 1.26.4, transformers 4.46.3, speechbrain 1.1.1, Python 3.11). The hosted ones run in any
environment with websockets 17.4.

The first runs' raw records were lost when the session restarted. The numbers above are from those runs, written
down at the time. The script was rewritten from them and rerun: Precision-3 and Sortformer low_latency gave the
same per-word slots, P/O counts and switch times. Speed depends on what else runs: the Sortformer rerun, under
`guard.sh` with nothing else heavy, took 1.12 x the audio (lag median 2.9 s, p95 8.3 s), where the first run took 1.42 x;
it is still slower than real time. Records of the reruns are in `try-it/testbed/diarization/`.
