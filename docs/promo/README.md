# gamst — English HTML presentation

Open `index.html` in a browser for nine self-contained slides. No external fonts or libraries are required.

- Arrow keys / PageUp / PageDown: navigate.
- F: fullscreen.
- Print PDF: print all slides as 16:9 pages.
- `?capture=1#4`: hide presentation controls and show slide 4.
- Local dashboard: `http://127.0.0.1:8766` (requires the local server).
- `gallery.html`: English presentation screenshots.

## Evidence and scope

Slides 4 and 5 use local QA results from public model-card audio, not a recording of the user's microphone or camera. The 92 ms metric rounds the last voice-emotion model call (91.9 ms); it excludes input accumulation, transcription, and VLM latency.

The current stack includes body pose landmarks and geometric raised-wrist flags. Trained gesture classification, SAM3, biometric identification, and speaker-to-face association are not advertised as installed capabilities. The transcription language selector does not validate Chinese or Korean voice-emotion accuracy.

Sources:

- [SUPERB model card](https://huggingface.co/superb/wav2vec2-base-superb-er)
- `docs/voice-analysis.md`
- `docs/tracking.md`
- Local QA: `.local-run/qa/voice-live-report.json` and `.local-run/qa/voice-report.json`. No original personal media is included in these presentation assets.

## Screenshots

The nine slide screenshots are actual browser captures of the English HTML deck. Native captures include display padding. The gallery now uses matching crop rectangles for previews and PNG downloads, with embedded captures so export also works when opened from disk. Original JPEGs remain unchanged. The slides export at 2560×1440. The dashboard captures predate the latest modal/toggle redesign.

Dashboard QA completed a real synthetic-image VLM call in 397 ms. Model and overlay switches were toggled off and restored; camera/microphone capture remained stopped. The two additional dashboard screenshots show English settings (including the successful test and MCP schemas) and the idle observability dashboard. These screenshots show capture stopped; zero gauges are not live measurements.

Existing screenshots are historical captures and may show the previous project branding. The maintained pages and repository links now use gamst.
