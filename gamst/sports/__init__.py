"""Sports parts: geometry, presets and the object contract shared by the sports recipes.

GAMST does not get a sports engine. A sport is a recipe (``gamst/recipes/racket-rally.json`` ...) that connects
generic parts; this package holds what several of those parts share and nothing sport-specific runs here:

- :mod:`gamst.sports.court`: an image-to-court homography from points the user marks (a reference), court sizes;
- :mod:`gamst.sports.presets`: per sport the ball, its fastest speed, the shortest time between two hits and the
  detector labels, used by ``gamst recipe check`` (physics, no tuning) and as recipe defaults;
- :mod:`gamst.sports.objects`: reading the ``objects`` contract every detector and tracker of GAMST emits
  (RF-DETR, YOLOE, YOLO26) and the ``balls`` contract of a ball model (TrackNetV3).

The modules that turn these into events live in ``gamst/providers/sports_*.py``; docs/sports.md has the whole map.
"""
