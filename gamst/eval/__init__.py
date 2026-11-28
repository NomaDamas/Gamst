"""Evaluation against owner-labelled clips (measurement only; nothing here runs in a recipe).

- :mod:`gamst.eval.pose_dump`: record what the per-track movement counter saw in one run (poses with joints, the person
  binding's records) into a local file the owner names, and replay it through the same counter with another movement.
- :mod:`gamst.eval.multi_person`: score events and identity against the owner's truth files
  (``docs/ground-truth/2026-11-28/``), with excluded ranges, held-out halves and an error breakdown.
"""
