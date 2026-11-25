"""MediaPipe Framework inside GAMST: a C++ CalculatorGraph as one Pipeline module.

- ``runner/``: ``gamst_graph_runner`` (C++) and ``GamstFrameCalculator``, built inside a pinned
  MediaPipe checkout by ``build.sh`` (revision in ``pin.json``).
- ``graphs/``: graph configs (``holistic_cpu.pbtxt``, ``holistic_gpu.pbtxt``).
- :class:`MediaPipeGraphModule`: the Pipeline module that owns the runner process.

See docs/mediapipe-graph.md.
"""
from .module import (HOLISTIC_BRANCHES, HOLISTIC_CPU, HOLISTIC_GPU, HOLISTIC_OUTPUTS, HOLISTIC_SIDE_PACKETS, PIN,
                     PROTOCOL, MediaPipeGraphModule, RunnerFailed, sha256_file)

# What the crew registry needs to install this part once (the single install path): the source pin,
# the build entry point and the staged artifact that is checked by SHA-256 (manifest.json written by
# build.sh). Registration in gamst/crew happens when feat/absorb-gamstcrew is merged.
ROLE = dict(
    role='vision.mediapipe_graph',
    summary='user-composed MediaPipe CalculatorGraph (C++) as one Pipeline module in its own process',
    source=dict(repository=PIN['repository'], tag=PIN['tag'], commit=PIN['commit'], bazel=PIN['bazel']),
    build='gamst/mediapipe_graph/build.sh',
    artifact='manifest.json',
    artifact_check='sha256 of runner binary, graphs and model files (MediaPipeGraphModule.from_manifest)',
    license='code: Apache-2.0; models: see MediaPipe model cards (not re-checked here)',
)

__all__ = ['HOLISTIC_BRANCHES', 'HOLISTIC_CPU', 'HOLISTIC_GPU', 'HOLISTIC_OUTPUTS', 'HOLISTIC_SIDE_PACKETS', 'PIN',
           'PROTOCOL', 'ROLE', 'MediaPipeGraphModule', 'RunnerFailed', 'sha256_file']
