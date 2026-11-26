# Handoff: feat/sota-first-rule (worktree .local-run/wt-sota-first, from integration 6298a8a)

Owner decision 31: "가장 SOTA가 우선이어야지, 더 약한 모델들이 투표로 이기면 안 된다." The three-recogniser majority of the
verification lane is replaced by a primary-first rule (`decide: "primary"`); `majority` stays an option.

## Status
- [x] Worktree, scope (word_verification, its tests, recipe parameters, docs; no count_video / partials)
- [x] Scribe v3 Live per-word confidence checked with the real API: committed words carry `logprob` (probe below)
- [ ] Rule + plumbing + tests
- [ ] Four clips x2 with the rule, scored against owner labels
- [ ] Docs: catch-up-verification, decision-log (31), lessons

## Scribe v3 Live probe (raw WebSocket, vad commits, keyterm = the counted word; one pass per clip)
`committed_transcript_with_timestamps.words[]` = `{text, start, end, type, speaker_id, logprob, characters, channel_index}`.
Counted-word logprob (p = exp): 뽀또 24.16 (Scribe's invented word / another voice) -1.108 (0.33); 피파 clip 41.76 (the
person, true) -3.843 (0.02). Full table in docs/catch-up-verification.md once written.
