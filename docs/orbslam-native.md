# ORB-SLAM3 native ABI — software slice

This private binding is not yet a registered gamst Context provider. It is the
first slice of [#186](https://github.com/NomaDamas/Gamst/issues/186), tracked by
[#191](https://github.com/NomaDamas/Gamst/issues/191). Family6 remains incomplete.
No trained vocabulary, upstream runtime or camera dataset is installed here.

## Runtime boundary

The C++ bridge calls the actual upstream `System` API; the Python wrapper uses
ctypes, not an assumed official Python package. Only an explicitly owned worker
may load it (`GAMST_PROVIDER_WORKER=1`). That environment marker is a misuse guard,
not a security sandbox. Do not load it in an application/server process. Upstream
can terminate the process on invalid native inputs or block during shutdown;
a supervising worker must enforce deadlines and reap only its own child. The
registered provider and its supervision integration are still required in #186.

There is one native construction per process, including failed construction.
`close()` is idempotent in Python; reset does not replenish the lifetime frame
budget (default300, maximum3000). A new native session requires a new process.
The budget bounds accepted tracking calls, not upstream native RSS or per-call
execution time. No hard memory or real-time guarantee is established by this slice.

## Build with preprovisioned assets

Use an already built checkout of
[ORB-SLAM3 at 4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4](https://github.com/UZ-SLAMLab/ORB_SLAM3/tree/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4).
CMake checks HEAD and tracked/staged modifications. Preprovision C++14, CMake3.19+,
OpenCV4, Eigen3, Pangolin and upstream's linked libraries in the same runtime.
From this repository checkout:

```sh
cmake -S native/orbslam3 -B .local-run/orb-build \
  -DORB_SLAM3_ROOT=/absolute/path/to/prebuilt/ORB_SLAM3
cmake --build .local-run/orb-build
```

This recipe has NOT been compiled against the real upstream dependency tree in
this delivery. Its successful compilation and real inference are open acceptance
gates. A wheel does not bundle the native bridge, upstream dependencies or assets;
use the repository checkout for native sources. No command downloads them.
Upstream is GPL-3.0; its dependencies have separate licenses. The SDK's license
metadata does not relicense a linked binary; distribution/license review is needed.

The private `ORBSLAMNative` constructor requires absolute paths and exact lowercase
SHA256 values for the bridge library, vocabulary and settings; maximum sizes are
256MiB, 256MiB and64KiB respectively. Use trusted, immutable local assets. Hashes
cover those files, not dynamically linked dependencies, and do not prevent a
concurrent filesystem replacement. Record compiler, dependencies, binary hashes
and source revision separately for a real acceptance run. Exported revision is a
build declaration, not cryptographic attestation of the upstream library.

## Calibration and observations

Only the flat version1 PinHole settings subset is accepted. The pinned upstream
[Settings parser](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4/src/Settings.cc)
requires camera, ORB extractor and viewer fields even when the viewer is disabled.
Provide the fields in the upstream monocular example, with integer syntax for
width/height/fps/RGB and extractor integer fields, and explicit decimal syntax for
real fields. The wrapper rejects missing/duplicate/unknown fields, atlas filenames,
YAML extensions, incomplete distortion coefficients, nonfinite values and invalid
bounds. Optional distortion is k1/k2/p1/p2, optionally k3. Input is uint8 grayscale
matching the calibration dimensions exactly: height32..1080, width32..1920.
The smallest pyramid level must retain at least80pixels on each axis. At every
level, the usable width after feature borders must exceed half the usable height;
conservative floor/ceil bounds cover rounding and avoid a zero-cell octree.
No resize, image-based calibration estimation or default camera intrinsics occur.

Extractor limits: features1..5000, levels1..8, scaleFactor1.01..2.0,
FAST thresholds1..255 with minimum no greater than initial. Capture seconds are
finite, nonnegative and strictly increasing within each reset epoch. A reset
permits a new time origin but consumes the same lifetime budget.

`track()` returns `state`, `map_changed` and `world_to_camera`. Only upstream
tracking state2 (OK) returns a validated rigid4x4 transform. Every other valid
state returns null, never a confident identity pose. `world_to_camera` is T_cw:
world coordinates to camera coordinates, with arbitrary monocular scale. It does
not measure meters, object motion, force or physical world velocity. `MapChanged`
is an upstream change hint, not an atomic/stable map identifier or a proof of
continuity. Relocalization, reset and loop closure require explicit provenance
before future Context consumers compare coordinates across frames.

## Reproducible software check

```sh
python3 -m unittest discover -s tests -p test_orbslam_native.py -v
```

With numpy and a C++ compiler, this compiles the real `bridge.cpp` against a small
**test-double System.h**. It tests ctypes/C ABI calls, lost-state abstention,
reset/frame budgets, shape/type/time/hash/config rejection, native exception and
single-session/close ownership. It does not compile actual ORB-SLAM3 or measure
SLAM accuracy. CI's optional lane explicitly supplies numpy and checks the compiler.
Actual calibrated replay, reference trajectories, map continuity, capture-to-Context
latency and memory evidence remain required in #186.
