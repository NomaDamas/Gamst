# Lessons learned (sports recipes, from 2026-11-27)

Keep adding to this file. Each lesson says what happened, the evidence, and what we do now.

1. **An MIT toolkit can ship AGPL weights.** roboflow/sports is MIT, but its soccer models are YOLOv8 `.pt` files trained
   with ultralytics (downloaded by `setup.sh`). Evidence: research 2026-11-27 in `docs/handoff-sports-recipes.md`.
   Now: check code licence and weight licence separately; port only the MIT logic onto RF-DETR (Apache-2.0).
2. **A vendor "sub-5 ms" names no model, input size or hardware.** The Ultralytics sports page gives one generic number.
   Now: any latency we compare is measured on this Mac with the model, input size and device written next to it.
3. **Check the disk before planning downloads.** The data volume had 3.9 GiB free (100% used) on 2026-11-27.
   Now: state file size and free space before asking the owner to approve a download.

## Implementation (2026-11-28)

4. **A stack must install its realtime fallback too.** A realtime run may switch to `choose.fallback` after the start
   benchmark; plain setup installed only the item's own candidate. Evidence: `setup racket-rally --stack ultralytics
   --timing realtime --dry-run` now plans `yolo26m.pt` and `yolo26s.pt`. Now: setup passes over both timings and the
   fallbacks.
5. **Check what setup would install for every stack, not only that the recipe loads.** The trackers module ran BoT-SORT
   but its role candidate defaulted to `gamst.ocsort`, so setup would not have installed `trackers`. Evidence: the
   `roles()` printout per stack and timing. Now: `tests/test_sports.py` loads every stack × timing and checks `roles`.
6. **Verify the wiring assumption in the real Pipeline before building on it.** The sports modules route by the
   producer's module name (`frame.sensor.sensor_id`); a Pipeline + Store test (detect → ball → contacts → rally, then
   `count_events`) confirmed it: 1 hit, 1 bounce, 1 rally.
7. **Copy links and numbers from the source, not from memory.** The TrackNetV3 checkpoint link and the YOLO26 digests
   were taken from the README and the GitHub release API; the checkpoint turned out to be a zip
   (`TrackNetV3_ckpts.zip`), not the `.pt` file.
8. **Run the suite the way CI does.** `pytest tests` fails at collection (`from tests.x import` without a package);
   CI runs `python -m unittest discover -s tests`.
9. **An independent review found 13 defects that 39 green synthetic tests did not.** Examples: Ultralytics'
   `model.track` hides detections whose track is not confirmed, so a fast ball would never reach the ball module;
   the record-speed bound refused 30 fps badminton, the frame rate of TrackNetV3's own data; a long pass that flew
   loose was never counted; one missed detection ended a billiards shot; a 60-frame median stalled realtime for
   about 500 ms. Evidence: the review report and the fixes in this branch (tests added for each). Now: an unanchored
   review runs before every sports PR, and synthetic tests include missed detections and long gaps.
10. **`str.replace` replaces every occurrence.** A fix meant for `check_config` also landed in a constructor with the
    same line and broke the module. Now: edits by script assert `count(old) == 1`.
