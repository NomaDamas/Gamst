"""GAMST 240 live (``gamst demo rotation --iphone/--iphone-replay``): the coin-flip recipe's Pipeline.

The phone streams continuously so the HUD can frame the shot; Start/Stop on the phone marks a
*take*. The parts live where the Pipeline composes them:

- the iPhone receiver and the ``raw.gmst`` wire format: :class:`gamst.inputs.IPhone240Input`;
- take recording and the after-take flip count (``refined.json``, ``refined.mp4``,
  ``vision.rotation.refined`` records): :class:`gamst.takes.TakeRecorder`, :class:`gamst.takes.TakeFlips`;
- the HUD: :class:`gamst.halo.HaloCoinView`.

The names below stay importable from here.
"""
from __future__ import annotations

import sys

from ..halo import CANVAS, Hud, Text, render_refined  # noqa: F401
from ..inputs import (FRONT_CAMERA, FULL, HEADER_SIZE, RECORDING, FrameHeader, decode,  # noqa: F401
                      frame_to_upright, parse_header, read_recording, sensor_to_upright,
                      upright_rect_to_sensor, upright_to_sensor)
from ..takes import Take, VideoSink, refine_take  # noqa: F401


async def run(args):
    """The demo's options -> the coin-flip recipe's Pipeline (one implementation: gamst.recipes)."""
    from .. import recipes
    recipe, options = recipes.from_rotation_demo(args)
    return await recipes.run_pipeline(recipe, options)


if __name__ == '__main__':
    from ..takes import main
    main(sys.argv[1:])
