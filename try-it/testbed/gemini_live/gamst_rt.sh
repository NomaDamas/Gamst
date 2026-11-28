#!/bin/bash
# GAMST realtime on the owner's clips, Main's chosen configuration; run it through try-it/guard.sh.
#   gamst_rt.sh NAME gwan|man MOVEMENT.json
# The recipe gets the clip at its own pace (--realtime), RTMW-l + YOLOX-m, person binding with the identity-recall
# chosen settings (SCRFD faces, face 0.15 / reject 0.10, reject_after 2, PersonViT 0.85, reach_back), the reference
# 감스트.person.json, and records poses + bindings (gamst.eval.pose_dump). Out: <try-it>/testbed/gemini_live/gamst/NAME.*
set -u
T=${GAMST_TRYIT:-/Users/jinminseong/Desktop/catch-your-mind/.local-run/try-it}
W=$(cd "$(dirname "$0")/../../.." && pwd)
TB=${GAMST_TESTBED:-"/Users/jinminseong/Desktop/홍명보/gamst/GAMST TestBed"}
n=$1; short=$2; ex=$3
if [ "$short" = man ]; then V="$TB/감스트 맨까송 - 맨가대(가습기X김바쏘).mp4"; else V="$TB/감스트-관제탑 모음집.mp4"; fi
O=$T/testbed/gemini_live/gamst; mkdir -p "$O"
# The realtime spill needs disk above the floor (lesson 107): refuse to start below it, and record what was free.
free_kb=$(df -k ~ | tail -1 | awk '{print $4}'); floor_gb=${GAMST_SPILL_FLOOR_GB:-5}
echo "$(date +%T) free $((free_kb / 1024)) MB on ~ (floor ${floor_gb} GB)" > "$O/$n.df"
if [ "$free_kb" -lt $((floor_gb * 1024 * 1024)) ]; then echo "STOP: $((free_kb / 1024)) MB free < ${floor_gb} GB"; exit 3; fi
export PYTHONPATH=$W GAMST_NO_STAR_PROMPT=1 GAMST_SPILL_FLOOR_GB=5
cd "$T"
start=$(python3 -c 'import time; print(time.time())')
host/bin/python -m gamst.eval.pose_dump record "$O/$n.jsonl.gz" -- person-action-count --video "$V" --realtime \
  --headless --max-records 200000 --report "$O/$n.report.json" --exemplar "$ex" \
  --person testbed/identity/clips/감스트.person.json --profile testbed/idrecall/profile-identity-scrfd.json \
  --set person_binding.face_threshold=0.15 --set person_binding.face_reject=0.1 --set person_binding.reject_after=2 \
  --set person_binding.body_threshold=0.85 --set person_binding.reach_back=true > "$O/$n.log" 2>&1
rc=$?
python3 -c "import time,sys; print(round(time.time()-float(sys.argv[1]),1))" "$start" > "$O/$n.wall_s"
echo "$n rc=$rc wall_s=$(cat "$O/$n.wall_s")"
exit $rc
