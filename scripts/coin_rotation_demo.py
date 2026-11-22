"""Local 30fps object-rotation demo.

Semantic mode asks a Qwen3.5-VL-compatible endpoint for a target box. For a
Qwen-free mechanical test, pass ``--no-semantic --roi x,y,w,h``: the box is
explicit test input and is labelled ``unknown-object``; no object identity is
claimed. In both modes, GAMST's existing/additive rotation capability counts
feature phase locally.
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from urllib.request import Request, urlopen

import cv2
import numpy as np

from gamst.capabilities import SensorMetadata
from gamst.plugins import InputFrame
from gamst.providers.angular_motion import AngularMotionCounter


def features(gray, mask, max_points=64):
    points = cv2.goodFeaturesToTrack(gray, maxCorners=max_points, qualityLevel=.01,
                                     minDistance=5, blockSize=7, mask=mask)
    return None if points is None else points.reshape(-1, 2)


def semantic_detect(frame, endpoint, model, target):
    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return None
    body = {"model": model, "temperature": 0, "max_tokens": 120,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": f"Find visible {target}. Return JSON only: found boolean, normalized box [x0,y0,x1,y1], confidence number."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode()}}]}],
            "response_format": {"type": "json_object"}}
    try:
        req = Request(endpoint.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=30) as response:
            result = json.loads(json.loads(response.read())["choices"][0]["message"].get("content") or "{}")
        box = result.get("box")
        if not result.get("found") or not isinstance(box, list) or len(box) != 4:
            return None
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = [float(value) for value in box]
        if not all(0 <= value <= 1 for value in (x0, y0, x1, y1)) or not x0 < x1 or not y0 < y1:
            return None
        return int(x0*w), int(y0*h), int((x1-x0)*w), int((y1-y0)*h), float(result.get("confidence", 0))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--vlm-endpoint")
    parser.add_argument("--vlm-model", default="qwen3.5-9b")
    parser.add_argument("--target", default="coin")
    parser.add_argument("--semantic-interval", type=float, default=1.0)
    parser.add_argument("--no-semantic", action="store_true")
    parser.add_argument("--roi", help="Qwen-free mechanical test ROI: x,y,w,h")
    args = parser.parse_args()
    if args.no_semantic and not args.roi:
        raise SystemExit("--no-semantic requires --roi x,y,w,h; identity remains unknown-object")
    if not args.no_semantic and not args.vlm_endpoint:
        raise SystemExit("semantic mode requires --vlm-endpoint")
    test_roi = tuple(int(value) for value in args.roi.split(',')) if args.roi else None
    if test_roi is not None and len(test_roi) != 4:
        raise SystemExit("--roi requires x,y,w,h")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("카메라를 열 수 없습니다. macOS 카메라 권한과 --camera를 확인하세요.")
    provider = AngularMotionCounter(min_radius=3, min_quality=.35)
    roi = previous_gray = previous_points = None
    sequence = 0
    started = time.monotonic()
    next_semantic = 0.0
    measured_frames = 0
    last_value = None
    status = "Watching..." if args.no_semantic else f"Waiting for semantic {args.target} detection..."
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            elapsed = time.monotonic() - started
            now_ms = int(elapsed * 1000)
            detected = None
            if args.no_semantic:
                detected = (*test_roi, 1.0)
                status = "unknown-object mechanical rotation test"
            elif elapsed >= next_semantic:
                detected = semantic_detect(frame, args.vlm_endpoint, args.vlm_model, args.target)
                next_semantic = elapsed + args.semantic_interval
                if detected is None:
                    roi = previous_gray = previous_points = None
                    provider.reset()
                    status = f"Waiting for semantic {args.target} detection..."
                else:
                    status = f"Semantic scene target: {args.target} ({detected[4]:.2f})"
            if detected is not None and (roi is None or args.no_semantic or elapsed >= next_semantic - args.semantic_interval):
                x, y, w, h, _confidence = detected
                roi = (x, y, w, h)
                gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mask = np.zeros(gray_now.shape, np.uint8)
                mask[y:y+h, x:x+w] = 255
                previous_points = features(gray_now, mask)
                previous_gray = gray_now
                provider.reset()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if roi is not None:
                x, y, w, h = roi
                current_points = None
                if previous_points is not None and len(previous_points) >= 4:
                    nxt, flow_status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray,
                        previous_points.reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .01))
                    if nxt is not None:
                        valid = flow_status.reshape(-1).astype(bool)
                        current_points = nxt.reshape(-1, 2)[valid]
                        previous_points = previous_points[valid]
                if current_points is not None and len(current_points) >= 4:
                    sensor = SensorMetadata(source_id="mac-camera", sensor_id="rotation-demo", sequence=sequence,
                                            stream_epoch="coin-demo")
                    last_value = provider.process(InputFrame({"track_id":"semantic-object","center":[x+w/2,y+h/2],
                        "points":current_points.tolist(),"feature_scores":[1.0]*len(current_points)},"video",now_ms,sensor=sensor))
                    sequence += 1
                    measured_frames += 1
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,180,0), 2)
                    cv2.polylines(frame, [np.int32(current_points).reshape(-1,1,2)], False, (0,220,0), 1)
                    cv2.putText(frame, f"rotation={last_value['rotation_count']:.2f} turns phase={last_value['phase_degrees']:.0f} deg", (20,35), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)
                    cv2.putText(frame, f"confidence={last_value['confidence']:.2f} points={len(current_points)}", (20,65), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2)
                else:
                    roi = previous_points = None
                    status = "Object lost; waiting for semantic detection" if not args.no_semantic else "Object lost; mechanical test waiting"
                previous_gray = gray
            cv2.putText(frame, status, (20,95), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,0), 2)
            cv2.imshow("GAMST 30fps object rotation", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'),27):
                break
            if key == ord('r'):
                roi = previous_gray = previous_points = None
                provider.reset()
        print(f"measured_frames={measured_frames}; last={last_value}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
