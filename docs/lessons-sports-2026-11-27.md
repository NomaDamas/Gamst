# Lessons learned (sports recipes, from 2026-11-27)

Keep adding to this file. Each lesson says what happened, the evidence, and what we do now.

1. **An MIT toolkit can ship AGPL weights.** roboflow/sports is MIT, but its soccer models are YOLOv8 `.pt` files trained
   with ultralytics (downloaded by `setup.sh`). Evidence: research 2026-11-27 in `docs/handoff-sports-recipes.md`.
   Now: check code licence and weight licence separately; port only the MIT logic onto RF-DETR (Apache-2.0).
2. **A vendor "sub-5 ms" names no model, input size or hardware.** The Ultralytics sports page gives one generic number.
   Now: any latency we compare is measured on this Mac with the model, input size and device written next to it.
3. **Check the disk before planning downloads.** The data volume had 3.9 GiB free (100% used) on 2026-11-27.
   Now: state file size and free space before asking the owner to approve a download.
