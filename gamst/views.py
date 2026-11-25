"""View modules: the application's screen for a running recipe (the HUD).

A view is application UI, outside GAMST's context boundary. It does two things:

- **input**: keys (and, in richer apps, buttons) become commands to the Pipeline: quit, reset the
  modules, switch a module on or off;
- **output**: it draws the camera frame with values read back from committed records, the same
  records an agent reads over MCP. A view never computes a measurement itself.

``gamst recipe run`` picks the view a recipe names. ``--headless`` replaces it with JSON lines.
The GAMST 240 HUD of the coin-flip recipe (``halo-coin``) is :class:`gamst.halo.HaloCoinView`.
"""
from __future__ import annotations

import json
import textwrap


class _Window:
    title = 'GAMST'

    def __init__(self, pipeline, *, headless=False, capabilities=()):
        self.pipeline, self.headless = pipeline, headless
        self.capabilities = tuple(capabilities)
        self.latest = {}
        self.printed = set()

    async def _refresh(self, records):
        for record in records:
            capability = ((record.get('observation') or {}).get('capability') or {}).get('id')
            if capability in self.capabilities:
                self.latest[capability] = record
                events = record['observation']['value'].get('events') if isinstance(
                    record['observation']['value'], dict) else None
                if events and record['id'] not in self.printed:
                    self.printed.add(record['id'])
                    print(json.dumps(dict(capability=capability, timestamp_ms=record['timestamp_ms'],
                                          events=events), ensure_ascii=False), flush=True)

    def _keys(self):
        import cv2
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            return False
        if key == ord('r'):
            for name in self.pipeline.modules:
                self.pipeline.disable(name)
                self.pipeline.enable(name)
        return True

    def close(self):
        if not self.headless:
            import cv2
            cv2.destroyAllWindows()


class CountView(_Window):
    """Any recipe that counts: the frame, then per declared capability its count, status and last events."""
    title = 'GAMST recipe'

    async def update(self, input_id, frame, records):
        await self._refresh(records)
        if self.headless or frame.modality != 'video':
            return True
        import cv2
        import numpy as np
        image = frame.data
        panel = np.full((image.shape[0], 460, 3), 24, np.uint8)
        lines = [f'GAMST  {self.pipeline.name}', '']
        for capability in self.capabilities:
            record = self.latest.get(capability)
            value = record['observation']['value'] if record else {}
            lines.append(capability)
            lines.append(f"  count {value.get('count', 0)}   {value.get('status', 'waiting')}")
            if 'distance' in value:
                lines.append(f"  distance {value['distance']:.3f}")
            for event in (value.get('events') or [])[-2:]:
                lines += textwrap.wrap(json.dumps(event, ensure_ascii=False), 52, initial_indent='  - ',
                                       subsequent_indent='    ')
            lines.append('')
        status = self.pipeline.status()['modules']
        lines.append('modules (r: reset, q: quit)')
        lines += [f"  {name}: {'on' if item['enabled'] else 'off'}  {item['last_latency_ms'] or 0:.0f} ms"
                  for name, item in status.items()]
        for index, line in enumerate(lines[:max(1, image.shape[0] // 22 - 1)]):
            cv2.putText(panel, line, (10, 26 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, .48,
                        (80, 220, 255) if index == 0 else (255, 255, 255), 1)
        cv2.imshow(self.title, np.hstack([image, panel]))
        return self._keys()


class RotationView(_Window):
    """The rotation panel: tracked regions, turns and rev/s from ``vision.rotation`` records."""
    title = 'GAMST rotation'

    async def update(self, input_id, frame, records):
        # The rotation module commits its own records, so read the newest one back.
        rows = await self.pipeline.human.get_observations(capability_id='vision.rotation', window='1h', limit=1)
        record = rows['items'][0] if rows['items'] else None
        await self._refresh([record] if record else [])
        if self.headless or frame.modality != 'video':
            return True
        import cv2
        from .demos.rotation import Panel
        cv2.imshow(self.title, Panel().draw(frame.data.copy(), record, [], 0))
        return self._keys()


VIEWS = {'count': CountView, 'rotation': RotationView}
