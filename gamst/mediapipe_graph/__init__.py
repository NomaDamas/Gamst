"""MediaPipe Framework inside GAMST: a C++ CalculatorGraph as one Pipeline module.

- ``runner/``: ``gamst_graph_runner`` (C++) and ``GamstFrameCalculator``, built inside a pinned
  MediaPipe checkout by ``build.sh`` (revision in ``pin.json``).
- ``graphs/``: graph configs (``holistic_cpu.pbtxt``, ``holistic_gpu.pbtxt``).
- :class:`MediaPipeGraphModule`: the Pipeline module that owns the runner process.

Installing it is the GAMST role ``vision.mediapipe_graph`` (gamst/crew/registry.py). The C++ side has not been
compiled yet; see docs/mediapipe-graph.md.
"""
from .module import (HOLISTIC_BRANCHES, HOLISTIC_CPU, HOLISTIC_GPU, HOLISTIC_OUTPUTS, HOLISTIC_SIDE_PACKETS, PIN,
                     PROTOCOL, MediaPipeGraphModule, RunnerFailed, holistic_plugin)

__all__ = ['HOLISTIC_BRANCHES', 'HOLISTIC_CPU', 'HOLISTIC_GPU', 'HOLISTIC_OUTPUTS', 'HOLISTIC_SIDE_PACKETS', 'PIN',
           'PROTOCOL', 'MediaPipeGraphModule', 'RunnerFailed', 'holistic_plugin']
