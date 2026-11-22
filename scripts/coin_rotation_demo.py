"""Local 30fps coin/object rotation demo.

Run on macOS with a camera:
    python scripts/coin_rotation_demo.py --camera 0

Draw an ROI around the coin/object, press ENTER, then watch the live overlay.
The demo uses OpenCV feature tracking plus GAMST AngularMotionCounter. It does
not call Qwen3.5-VL and does not claim face/heads/tails semantics. Press r to
select a new ROI and q/ESC to quit. Camera permissions must be granted to the
terminal/Python process.
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
    roi = None
    previous_gray = None
    previous_points = None
    previous_center_y = None
    sequence = 0
    started = time.monotonic()
    measured_frames = 0
    try:
        while True:
            if roi is None:
                shown = frame.copy()
                cv2.putText(shown, "Drag ROI around coin/object, ENTER to start, q to quit", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
                selected = cv2.selectROI("GAMST 30fps object rotation", shown, fromCenter=False, showCrosshair=True)
                if selected[2] < 20 or selected[3] < 20:
                    break
                x, y, w, h = map(int, selected)
                roi = (x, y, w, h)
                previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mask = np.zeros(previous_gray.shape, np.uint8)
                mask[y:y+h, x:x+w] = 255
                previous_points = features(previous_gray, mask)
                previous_center_y = y + h / 2
                provider.reset()
                sequence = 0
                started = time.monotonic()
                continue
            ok, frame = cap.read()
            if not ok:
                break
            now_ms = int((time.monotonic() - started) * 1000)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x, y, w, h = roi
            center = [x + w / 2, y + h / 2]
            current_points = None
            if previous_points is not None and len(previous_points) >= 4:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, previous_points.reshape(-1, 1, 2), None,
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
                data = {"track_id": "manual-object", "center": center,
                        "points": current_points.tolist(), "feature_scores": [1.0] * len(current_points)}
                sensor = SensorMetadata(source_id="mac-camera", sensor_id="rotation-demo", sequence=sequence,
                                        stream_epoch="coin-demo")
                value = provider.process(InputFrame(data, "video", now_ms, sensor=sensor))
                sequence += 1
                measured_frames += 1
                cv2.polylines(frame, [np.int32(current_points).reshape(-1, 1, 2)], False, (0, 220, 0), 1)
                cv2.circle(frame, (int(center[0]), int(center[1])), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"rotation={value['rotation_count']:.2f} turns  phase={value['phase_degrees']:.0f} deg",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2)
                cv2.putText(frame, f"velocity={value['angular_velocity_deg_s']:.0f} deg/s  confidence={value['confidence']:.2f}  points={len(current_points)}",
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "feature points lost - hold object still or press r", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 180, 0), 2)
            cv2.imshow("GAMST 30fps object rotation", frame)
            previous_gray = gray
            previous_center_y = center[1]
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                roi = None
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"측정 프레임: {measured_frames}")


if __name__ == "__main__":
    main()
