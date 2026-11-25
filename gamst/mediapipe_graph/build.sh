#!/usr/bin/env bash
# Build GAMST's MediaPipe graph runner from a pinned MediaPipe source revision.
#
#   GAMST_MEDIAPIPE_BUILD_DIR=/Volumes/External/gamst-mediapipe gamst/mediapipe_graph/build.sh [--gpu] [--check]
#
# Where the build goes is the owner's decision (source + Bazel cache: estimated
# 8-10 GB, not measured), so there is no default location. Everything (source
# checkout, Bazel output root, staged runner) stays under GAMST_MEDIAPIPE_BUILD_DIR.
#
#   --check   check tools, OpenCV and free space, print the plan, download nothing
#   --gpu     build gamst_graph_runner_gpu (Linux with EGL/OpenGL ES; not macOS)
#
# Output: $GAMST_MEDIAPIPE_BUILD_DIR/out/<commit12>-<cpu|gpu>/
#   bin/gamst_graph_runner[_gpu]   graphs/*.pbtxt   resources/mediapipe/modules/...
#   manifest.json                  SHA-256 of the binary, every model file and graph
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN="$HERE/pin.json"
pin() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$PIN" "$1"; }
REPOSITORY="$(pin repository)"
COMMIT="$(pin commit)"
TAG="$(pin tag)"
BAZEL_VERSION="$(pin bazel)"

CHECK=0
VARIANT=cpu
for arg in "$@"; do
  case "$arg" in
    --check) CHECK=1 ;;
    --gpu) VARIANT=gpu ;;
    *) echo "unknown option: $arg" >&2; exit 64 ;;
  esac
done

: "${GAMST_MEDIAPIPE_BUILD_DIR:?set GAMST_MEDIAPIPE_BUILD_DIR to a directory with room for the build (owner decision: external disk, cleaned space or a GPU server)}"
MIN_FREE_GB="${GAMST_MEDIAPIPE_MIN_FREE_GB:-12}"
ROOT="$(mkdir -p "$GAMST_MEDIAPIPE_BUILD_DIR" && cd "$GAMST_MEDIAPIPE_BUILD_DIR" && pwd)"
SRC="$ROOT/src"
OUT="$ROOT/out/${COMMIT:0:12}-$VARIANT"
TARGET=gamst_graph_runner
[ "$VARIANT" = gpu ] && TARGET=gamst_graph_runner_gpu

problems=()
BAZEL=""
if command -v bazelisk >/dev/null 2>&1; then BAZEL=bazelisk
elif command -v bazel >/dev/null 2>&1; then BAZEL=bazel
else problems+=("Bazel $BAZEL_VERSION is missing: install bazelisk (it reads MediaPipe's .bazelversion)")
fi
command -v git >/dev/null 2>&1 || problems+=("git is missing")
command -v python3 >/dev/null 2>&1 || problems+=("python3 is missing")
python3 -c 'import numpy' 2>/dev/null || problems+=("python3 needs numpy (MediaPipe build rules import it)")

OPENCV=""
case "$(uname -s)" in
  Darwin)
    [ "$VARIANT" = gpu ] && problems+=("--gpu needs Linux with EGL; MediaPipe desktop GPU is not supported on macOS")
    if command -v brew >/dev/null 2>&1 && [ -d "$(brew --prefix opencv 2>/dev/null)/include/opencv4" ]; then
      OPENCV="$(cd "$(brew --prefix opencv)" && pwd -P)"
    else
      problems+=("OpenCV 4 is missing: brew install opencv (MediaPipe's CPU image calculators link it)")
    fi ;;
  Linux)
    if [ -d /usr/include/opencv4 ] || [ -d /usr/local/include/opencv4 ]; then OPENCV=system
    else problems+=("OpenCV is missing: apt install libopencv-core-dev libopencv-imgproc-dev libopencv-video-dev libopencv-highgui-dev libopencv-calib3d-dev libopencv-features2d-dev")
    fi
    if [ "$VARIANT" = gpu ]; then
      [ -e /usr/include/EGL/egl.h ] || problems+=("--gpu needs EGL/GLES headers: apt install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev")
    fi ;;
  *) problems+=("unsupported OS $(uname -s)") ;;
esac

free_gb="$(df -Pk "$ROOT" | awk 'NR==2 {printf "%d", $4/1048576}')"
[ "$free_gb" -ge "$MIN_FREE_GB" ] || problems+=("only ${free_gb} GB free under $ROOT; the build needs about ${MIN_FREE_GB} GB (estimate, not measured; override with GAMST_MEDIAPIPE_MIN_FREE_GB)")

echo "MediaPipe $TAG @ $COMMIT ($REPOSITORY), Bazel $BAZEL_VERSION"
echo "build dir: $ROOT (${free_gb} GB free)  target: //mediapipe/gamst:$TARGET  output: $OUT"
if [ "${#problems[@]}" -gt 0 ]; then
  printf 'not ready:\n'; printf '  - %s\n' "${problems[@]}"
  exit 1
fi
if [ "$CHECK" = 1 ]; then
  echo "ready (check only; nothing downloaded)"
  exit 0
fi

# 1. Pinned source: fetch exactly one commit; git verifies it by its hash.
if [ ! -d "$SRC/.git" ]; then
  git init -q "$SRC"
  git -C "$SRC" remote add origin "$REPOSITORY"
fi
if [ "$(git -C "$SRC" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$COMMIT" ]; then
  git -C "$SRC" fetch -q --depth 1 origin "$COMMIT"
  git -C "$SRC" checkout -q --force FETCH_HEAD
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = "$COMMIT" ] || { echo "checkout is not $COMMIT" >&2; exit 1; }
[ "$(cat "$SRC/.bazelversion")" = "$BAZEL_VERSION" ] || { echo "pin.json bazel differs from .bazelversion" >&2; exit 1; }

# 2. GAMST overlay: //mediapipe/gamst (runner, calculator, graphs).
rm -rf "$SRC/mediapipe/gamst"
mkdir -p "$SRC/mediapipe/gamst/graphs"
cp "$HERE"/runner/BUILD "$HERE"/runner/*.cc "$HERE"/runner/*.h "$SRC/mediapipe/gamst/"
cp "$HERE"/graphs/*.pbtxt "$SRC/mediapipe/gamst/graphs/"

# 3. Local OpenCV (macOS: Homebrew OpenCV 4 instead of MediaPipe's opencv@3 default).
if [ "$(uname -s)" = Darwin ]; then
  python3 - "$SRC" "$OPENCV" <<'PY'
import re, sys
src, prefix = sys.argv[1], sys.argv[2]
path = f'{src}/WORKSPACE'
text = open(path).read()
text, count = re.subn(r'(name = "macos_opencv",\n\s*build_file = "@//third_party:opencv_macos.BUILD",\n(?:\s*#.*\n)*\s*path = )"[^"]*"',
                      lambda m: m.group(1) + '"' + prefix + '"', text)
if count != 1:
    sys.exit('WORKSPACE macos_opencv block not found; update build.sh for this MediaPipe revision')
open(path, 'w').write(text)
open(f'{src}/third_party/opencv_macos.BUILD', 'w').write('''licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

# Written by GAMST build.sh for Homebrew OpenCV 4 (include/opencv4).
cc_library(
    name = "opencv",
    srcs = glob(["lib/libopencv_%s.dylib" % m for m in
                 ["core", "calib3d", "features2d", "highgui", "imgcodecs", "imgproc", "video", "videoio"]]),
    hdrs = glob(["include/opencv4/opencv2/**/*.h*"]),
    includes = ["include/opencv4/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
''')
PY
fi

# 4. Build. The Bazel output root lives under the chosen build dir too.
FLAGS=(-c opt "--action_env=PYTHON_BIN_PATH=$(command -v python3)" "--copt=-DGAMST_MEDIAPIPE_REVISION=\"$COMMIT\"")
if [ "$VARIANT" = cpu ]; then FLAGS+=(--define MEDIAPIPE_DISABLE_GPU=1)
else FLAGS+=(--copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11)
fi
(cd "$SRC" && "$BAZEL" "--output_user_root=$ROOT/bazel-root" build "${FLAGS[@]}" \
  "//mediapipe/gamst:$TARGET" "//mediapipe/gamst:holistic_${VARIANT}_binarypb")

# 5. Stage binary, graphs and model files, and record their SHA-256.
rm -rf "$OUT"
mkdir -p "$OUT/bin" "$OUT/graphs" "$OUT/resources"
cp -L "$SRC/bazel-bin/mediapipe/gamst/$TARGET" "$OUT/bin/"
cp "$HERE"/graphs/*.pbtxt "$OUT/graphs/"
# Model files come from the binary's runfiles (generated .tflite links and source .txt alike).
python3 - "$SRC/bazel-bin/mediapipe/gamst/$TARGET.runfiles" "$OUT/resources" <<'PY'
import os, shutil, sys
runfiles, resources = sys.argv[1:]
staged = 0
for folder, _, names in os.walk(runfiles, followlinks=True):
    for name in names:
        path = os.path.join(folder, name)
        marker = path.find('/mediapipe/modules/')
        if marker < 0 or not name.endswith(('.tflite', '.txt')):
            continue
        relative = path[marker + 1:]
        os.makedirs(os.path.join(resources, os.path.dirname(relative)), exist_ok=True)
        shutil.copyfile(path, os.path.join(resources, relative))
        staged += 1
if not staged:
    sys.exit(f'no model files under {runfiles}')
PY
python3 - "$OUT" "$TARGET" "$COMMIT" "$TAG" "$BAZEL_VERSION" "$VARIANT" <<'PY'
import hashlib, json, os, sys, time
out, target, commit, tag, bazel, variant = sys.argv[1:]
def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
files = {}
for base in ('graphs', 'resources'):
    for folder, _, names in os.walk(os.path.join(out, base)):
        for name in names:
            path = os.path.join(folder, name)
            files[os.path.relpath(path, out)] = digest(path)
manifest = dict(kind='gamst.mediapipe_graph.runner', variant=variant, mediapipe_tag=tag, mediapipe_commit=commit,
                bazel=bazel, built_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                runner=f'bin/{target}', runner_sha256=digest(os.path.join(out, 'bin', target)),
                resource_root='resources', files=dict(sorted(files.items())))
json.dump(manifest, open(os.path.join(out, 'manifest.json'), 'w'), indent=2)
print(json.dumps(dict(runner=manifest['runner'], runner_sha256=manifest['runner_sha256'], files=len(files))))
PY
echo "staged: $OUT"
echo "use:    MediaPipeGraphModule.from_manifest('$OUT/manifest.json')"
