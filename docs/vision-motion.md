# Image motion provider

`humansense.providers.motion.OpenCVFlow` executes OpenCV Lucas-Kanade or Farneback optical flow locally. It keeps one bounded grayscale image and returns pixel displacement, median pixel speed, moving-sample fraction and the actual capture interval. It does not require a neural checkpoint or VLM call.

```python
import cv2
from humansense import SensorStreams
from humansense.providers.motion import OpenCVFlow

streams = SensorStreams(human)
spec = OpenCVFlow.capability("lucas_kanade", version=cv2.__version__)
streams.register(spec, lambda: OpenCVFlow(method="lucas_kanade"))
await streams.open("camera1", "sensor1", [spec.id])
# Submit uint8 BGR or grayscale InputFrame values with SensorMetadata.
```

Install the optional `vision` extra. Importing the module or describing a capability does not import OpenCV or start capture. Construction explicitly loads the installed backend. There is no model download.

The output's `motion` object includes `dx_pixels`, `dy_pixels`, `median_speed_pixels_per_second`, `moving_fraction`, `sample_count`, `interval_ms`, original dimensions, algorithm and OpenCV version. Scaling is converted back to original image pixels after internal resizing. Sparse flow rejects tracks with poor forward/backward agreement. Dense flow samples textured image locations. Blank images, initialization, shape changes and excessive capture gaps yield no estimate, rather than claiming zero motion.

These are image observations. Camera and object motion are not separated; the output explicitly says `camera_motion_separated=false`. Pixel speed cannot become metres per second without a separate calibrated geometry pipeline. Median motion can miss small moving objects; it is not an object-specific tracker, collision detector or complete optical-flow field. Providers consume strictly increasing capture timestamps and reset when the stream reports discontinuity.

Validation uses real OpenCV calls on deterministic textured images with known translations, testing both methods, resize scaling, texture absence, lifecycle and the public SensorStreams-to-context path. This establishes algorithm execution and synthetic displacement accuracy, not real-camera robustness or the complete depth/mapping family. CI installs the supported optional vision dependency to exercise these tests; the dependency-free core suite may skip backend tests.

Local execution evidence on 2026-11-14: Apple Silicon macOS 15.6.1, OpenCV 5.0.0, NumPy 2.2.6. This existing local installation is outside the package's declared OpenCV `<5` range; the dedicated CI job tests the supported dependency resolution separately. No broader model-quality, cost or physical-device completion is inferred from these tests. Related required family: #22.
