# Where attribution fails (2026-11-26)

`scripts/qa_attribution_bottleneck.py`: CAM++ embeddings of the owner-labelled words (person / another voice), scored against
A = the voice from another recording (what the runs used) and B = the person's other labelled words in the same clip
(leave-one-out; an upper bound for a same-recording few-shot). Windows of 0.6 s and 1.5 s around each word.

| clip | words person / other | A: person ≥ 0.5 (1.5 s) | B: person ≥ 0.5 (1.5 s) | A: AUC person vs other | B: AUC |
|---|---|---|---|---|---|
| 뽀또 | 9 / 6 | 44 % | 100 % | 0.81 | 0.89 |
| 몇기야 | 6 / 2 | 0 % | 83 % | 0.42 | 0.00 (the two other-voice words score higher) |
| 시발 | 7 / 0 | 0 % | 71 % | – | – |
| 달 | 5 words / 0 | 0 % | 100 % | – | – |

1. **Main bottleneck: the voice reference.** With a reference from the same recording the person's words move from mostly
   below 0.5 to mostly above it. This is an input/structure problem, fixable without a new model.
2. **Second bottleneck: deciding one short word at a time.** Even with the same-recording reference, other voices near the
   person score close to (뽀또) or above (몇기야, 2 words) the person, because a 0.6-1.5 s window holds little voice and often
   both people. Precision-3, which decides on long speaker segments, separated the person in 뽀또 (10/10). Decisions should
   lean on speaker segments and on comparison with the other speakers present, not on a fixed score of one snippet.
3. Hearing the words is not the bottleneck (45 of 46 heard).
