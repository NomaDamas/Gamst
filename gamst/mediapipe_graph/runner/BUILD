# GAMST graph runner, built inside a pinned MediaPipe checkout as //mediapipe/gamst
# (build.sh copies this directory and ../graphs there; see ../pin.json for the revision).
#
#   CPU (macOS, Linux):  bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 //mediapipe/gamst:gamst_graph_runner
#   GPU (Linux + EGL):   bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 \
#                          //mediapipe/gamst:gamst_graph_runner_gpu

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//mediapipe/framework/tool:mediapipe_graph.bzl", "mediapipe_binary_graph")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "gamst_frame",
    hdrs = ["gamst_frame.h"],
    deps = ["//mediapipe/framework:packet"],
)

cc_library(
    name = "gamst_frame_calculator",
    srcs = ["gamst_frame_calculator.cc"],
    deps = [
        ":gamst_frame",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

# Calculators and subgraphs the holistic graphs name. A user-composed graph adds
# its own deps list next to these and its own runner target below.
cc_library(
    name = "holistic_common_deps",
    deps = [
        ":gamst_frame_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:split_proto_list_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/util:landmarks_smoothing_calculator",
    ],
)

cc_library(
    name = "holistic_cpu_deps",
    deps = [
        ":holistic_common_deps",
        "//mediapipe/modules/holistic_landmark:face_landmarks_from_pose_cpu",
        "//mediapipe/modules/holistic_landmark:hand_landmarks_left_and_right_cpu",
        "//mediapipe/modules/pose_landmark:pose_landmark_cpu",
    ],
)

cc_library(
    name = "holistic_gpu_deps",
    deps = [
        ":holistic_common_deps",
        "//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",
        "//mediapipe/modules/holistic_landmark:face_landmarks_from_pose_gpu",
        "//mediapipe/modules/holistic_landmark:hand_landmarks_left_and_right_gpu",
        "//mediapipe/modules/pose_landmark:pose_landmark_gpu",
    ],
)

# Model files the holistic subgraphs load at run time (fetched by Bazel from
# MediaPipe's pinned external_files.bzl; build.sh stages them with their SHA-256).
HOLISTIC_MODELS = [
    "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
    "//mediapipe/modules/face_landmark:face_landmark.tflite",
    "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
    "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
    "//mediapipe/modules/hand_landmark:handedness.txt",
    "//mediapipe/modules/holistic_landmark:hand_recrop.tflite",
    "//mediapipe/modules/pose_detection:pose_detection.tflite",
    "//mediapipe/modules/pose_landmark:pose_landmark_full.tflite",
    "//mediapipe/modules/pose_landmark:pose_landmark_heavy.tflite",
    "//mediapipe/modules/pose_landmark:pose_landmark_lite.tflite",
]

RUNNER_DEPS = [
    ":gamst_frame",
    "//mediapipe/framework:calculator_framework",
    "//mediapipe/framework/formats:classification_cc_proto",
    "//mediapipe/framework/formats:detection_cc_proto",
    "//mediapipe/framework/formats:image_frame",
    "//mediapipe/framework/formats:landmark_cc_proto",
    "//mediapipe/framework/formats:rect_cc_proto",
    "//mediapipe/framework/port:file_helpers",
    "//mediapipe/framework/port:parse_text_proto",
    "//mediapipe/framework/port:status",
    "//mediapipe/util:resource_util",
    "@com_google_absl//absl/flags:flag",
    "@com_google_absl//absl/flags:parse",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/strings",
]

cc_binary(
    name = "gamst_graph_runner",
    srcs = ["gamst_graph_runner.cc"],
    data = HOLISTIC_MODELS + ["graphs/holistic_cpu.pbtxt"],
    deps = RUNNER_DEPS + [":holistic_cpu_deps"],
)

cc_binary(
    name = "gamst_graph_runner_gpu",
    srcs = ["gamst_graph_runner.cc"],
    data = HOLISTIC_MODELS + ["graphs/holistic_gpu.pbtxt"],
    local_defines = ["GAMST_WITH_GPU"],
    deps = RUNNER_DEPS + [
        ":holistic_gpu_deps",
        "//mediapipe/gpu:gpu_shared_data_internal",
    ],
)

# Parses the graph text at build time, so a typo fails the build, not the run.
mediapipe_binary_graph(
    name = "holistic_cpu_binarypb",
    graph = "graphs/holistic_cpu.pbtxt",
    output_name = "holistic_cpu.binarypb",
    deps = [":holistic_cpu_deps"],
)

mediapipe_binary_graph(
    name = "holistic_gpu_binarypb",
    graph = "graphs/holistic_gpu.pbtxt",
    output_name = "holistic_gpu.binarypb",
    deps = [":holistic_gpu_deps"],
)
