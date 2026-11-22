"""Local automatic 30fps object-rotation demo.

    PYTHONPATH=. python scripts/coin_rotation_demo.py --camera 0

No manual ROI is needed. The demo learns a static background, finds a moving
compact object, starts feature tracking automatically, and sends the tracked
center/points to GAMST ``AngularMotionCounter``. Press ``r`` to clear the
tracker and relearn the background; press ``q``/ESC to quit.

This local first pass does not call Qwen3.5-VL and does not infer heads/tails.
It measures rotation only. Camera permissions must be granted to Terminal.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from gamst.capabilities import SensorMetadata
from gamst.plugins import InputFrame
from gamst.providers.angular_motion import AngularMotionCounter


def features(gray, mask, max_points=64):
    points = cv2.goodFeaturesToTrack(gray, maxCorners=max_points, qualityLevel=.01,
                                     minDistance=5, blockSize=7, mask=mask)
    return None if points is None else points.reshape(-1, 2)


def automatic_roi(background, frame):
    delta = cv2.absdiff(background, frame)
    mask = cv2.threshold(delta, 28, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = frame.shape[:2]
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < max(100, width * height * .00005) or area > width * height * .35:
            continue
        ratio = max(w / max(h, 1), h / max(w, 1))
        if ratio > 4:
            continue
        candidates.append((area, (x, y, w, h)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("카메라를 열 수 없습니다. macOS 카메라 권한과 --camera를 확인하세요.")
    provider = AngularMotionCounter(min_radius=3, min_quality=.35)
    # The ROI detector uses the explicit adaptive background below.
    roi = None
    previous_gray = None
    previous_points = None
    sequence = 0
    started = time.monotonic()
    measured_frames = 0
    last_value = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now_ms = int((time.monotonic() - started) * 1000)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            background = getattr(main, "background", None)
            if background is None:
                main.background = gray.copy()
                background = main.background
            else:
                main.background = cv2.addWeighted(background, .96, gray, .04, 0)
                background = main.background
            if roi is None:
                roi = automatic_roi(background, gray)
                if roi is not None:
                    x, y, w, h = roi
                    mask = np.zeros(gray.shape, np.uint8)
                    mask[y:y+h, x:x+w] = 255
                    previous_points = features(gray, mask)
                    previous_gray = gray
                    provider.reset()
                    sequence = 0
                cv2.putText(frame, "Watching... move object to auto-detect (r=reset, q=quit)", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
            else:
                x, y, w, h = roi
                current_points = None
                if previous_points is not None and len(previous_points) >= 4:
                    nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                        previous_gray, gray, previous_points.reshape(-1, 1, 2), None,
                        winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .01))
                    if nxt is not None:
                        valid = status.reshape(-1).astype(bool)
                        current_points = nxt.reshape(-1, 2)[valid]
                        previous_points = previous_points[valid]
                if current_points is None or len(current_points) < 4:
                    mask = np.zeros(gray.shape, np.uint8)
                    mask[y:y+h, x:x+w] = 255
                    current_points = features(gray, mask)
                    previous_points = current_points
                if current_points is not None and len(current_points) >= 4:
                    center = [x + w / 2, y + h / 2]
                    data = {"track_id": "auto-object", "center": center,
                            "points": current_points.tolist(), "feature_scores": [1.0] * len(current_points)}
                    sensor = SensorMetadata(source_id="mac-camera", sensor_id="rotation-demo", sequence=sequence,
                                            stream_epoch="coin-demo")
                    last_value = provider.process(InputFrame(data, "video", now_ms, sensor=sensor))
                    sequence += 1
                    measured_frames += 1
                    cv2.polylines(frame, [np.int32(current_points).reshape(-1, 1, 2)], False, (0, 220, 0), 1)
                    cv2.circle(frame, (int(center[0]), int(center[1])), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 180, 0), 2)
                    cv2.putText(frame, f"rotation={last_value['rotation_count']:.2f} turns  phase={last_value['phase_degrees']:.0f} deg",
                                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2)
                    cv2.putText(frame, f"velocity={last_value['angular_velocity_deg_s']:.0f} deg/s  confidence={last_value['confidence']:.2f}  points={len(current_points)}",
                                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                else:
                    roi = None
                    cv2.putText(frame, "Tracking lost - watching for movement", (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                previous_gray = gray
            cv2.imshow("GAMST automatic 30fps object rotation", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                roi = None
                previous_points = None
                previous_gray = None
                main.background = gray.copy()
                provider.reset()
        print(f"측정 프레임: {measured_frames}; 마지막 결과: {last_value}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
