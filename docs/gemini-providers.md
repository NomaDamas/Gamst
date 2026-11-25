# Gemini Companion providers (example consumer)

The Companion is an example application in `examples/companion/` that reads
GAMST context and asks a chat model for a reply; it is not part of the GAMST
core, MCP or dashboard. Gemini Flash uses the OpenAI-compatible Chat Completions
API; Gemini Live keeps one persistent bidirectional session. Both are explicit
remote consumers: neither is selected by default, and observations remain
measurements rather than claims about a person's emotions or intentions.

## Gemini Flash

The Flash preset needs no additional Python SDK. Set the API key and select the
provider when constructing the example Companion from a source checkout:

```bash
export GAMST_COMPANION_PROVIDER=gemini-flash
export GEMINI_API_KEY='YOUR_KEY'
# Optional: export GAMST_GEMINI_MODEL=gemini-3.5-flash-lite
```

```python
from gamst import Gamst
from examples.companion import Companion, companion_backend_from_env

async with Gamst(consent=True) as human:
    reply = await Companion(human, companion_backend_from_env()).reply("What changed?")
```

The defaults are model `gemini-3.5-flash-lite` and endpoint
`https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`.
The key is read only when a request is made. This endpoint and bearer-key usage
match Google's [Gemini OpenAI compatibility guide](https://ai.google.dev/gemini-api/docs/openai).
The preset deliberately sets `allow_remote=True`; choosing it authorizes sending
the bounded Companion conversation and derived text context to Google.

### Vertex AI

Select Vertex with `GAMST_GEMINI_VERTEX=true`, supply its project and location,
and provide an OAuth access token through `GOOGLE_OAUTH_ACCESS_TOKEN`:

```bash
export GOOGLE_OAUTH_ACCESS_TOKEN="$(gcloud auth print-access-token)"
export GAMST_GEMINI_VERTEX=true
export GAMST_GEMINI_PROJECT='YOUR_PROJECT_ID'
export GAMST_GEMINI_LOCATION='us-central1'
export GAMST_GEMINI_MODEL='google/gemini-3.5-flash-lite'
export GAMST_COMPANION_PROVIDER=gemini-flash
```

The preset constructs
`https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi/chat/completions`.
It uses the token as the bearer credential and never stores it in records or
prompts. Access tokens expire; refresh the environment value before it expires
or restart with a newly printed token. Google's [Vertex OpenAI setup](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/openai)
and [Chat Completions sample](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-gemini-chat-completions-non-streaming)
document the `/v1/.../endpoints/openapi` base URL, OAuth token, and `google/`
model-name form. Use the exact model ID available to the selected project and
region.

## Gemini Live

Install the optional SDK; the core GAMST package does not import or require it:

```bash
.venv/bin/python -m pip install -e '.[gemini]'
export GEMINI_API_KEY='YOUR_KEY'
export GAMST_COMPANION_PROVIDER=gemini-live
```

The default model is `gemini-live-2.5-flash-native-audio`; set
`GAMST_GEMINI_MODEL` to another Live model available to the account. Despite the
native-audio model name, this adapter requests **TEXT responses** and sends text
only. It lazily opens one `client.aio.live.connect(...)` session on the first
Companion reply, reuses that session for subsequent replies, and serializes each
context push followed by a completed user turn. Responses are collected until
`turn_complete`, with a 30-second default timeout in the direct API.

For Vertex Live, configure ADC and the same Vertex selection variables:

```bash
gcloud auth application-default login
export GAMST_GEMINI_VERTEX=true
export GAMST_GEMINI_PROJECT='YOUR_PROJECT_ID'
export GAMST_GEMINI_LOCATION='us-central1'
export GAMST_COMPANION_PROVIDER=gemini-live
```

Live Vertex construction is `genai.Client(vertexai=True, project=..., location=...)`;
it uses ADC rather than the Flash preset's `GOOGLE_OAUTH_ACCESS_TOKEN`. See
Google's [Live session setup](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api/start-manage-session).
The SDK session/config and ordered `send_client_content` calls follow the
[GenAI SDK Live guide](https://ai.google.dev/gemini-api/docs/live-api/get-started-sdk?lang=python)
and [SDK session reference](https://googleapis.github.io/python-genai/genai.html#genai.live.AsyncSession).

Direct users can inject a session factory for deterministic tests:

```python
from examples.companion.gemini import GeminiLiveCompanion

async with GeminiLiveCompanion() as live:
    await live.push_context({"event": "speech.started", "capture_ms": 10,
                             "now_ms": 20, "fresh": True,
                             "evidence_id": "evidence-1", "value": 1})
    answer = await live.ask("Summarize the measured change.")
```

Each context delta contains only `event`, `capture_ms`, `now_ms`, `fresh`,
`evidence_id`, and `value`. Other top-level fields, including audio and image
fields, are dropped. A delta larger than 4 KiB of UTF-8 JSON is rejected. The
Companion adapter bounds larger aggregate context with a marked excerpt before
sending it. Its system instruction treats quoted context as untrusted data and
prohibits inner-state claims.

## Scene VLM boundary

The Flash Chat Completions URL above is also the Gemini endpoint for an
OpenAI-compatible scene-VLM integration. This is separate from selecting the
Companion consumer. The Companion example never configures a scene VLM; the
existing `LocalScenePlugin` is loopback-only. An application
that separately configures a remote scene VLM must explicitly authorize and
authenticate that path.

## Privacy, pricing, and limits

- Flash Companion sends prepared conversation text and derived context. Live
  sends the newest user turn plus the bounded context delta. These consumer
  adapters send no raw camera frames, JPEG data, PCM, or other media.
- Raw media leaves the machine only if a separate scene VLM is explicitly pointed
  at Gemini (or another remote provider). Selecting a Gemini Companion alone does
  not do that.
- Live sessions have provider-side connection/session limits. This adapter does
  not implement session resumption, automatic reconnect, audio input, video
  input, audio output, or tool execution; it is a text-event consumer.
- Consult the current [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
  and [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) before use.
  For media-capable Live integrations, Google's [Live API overview](https://ai.google.dev/gemini-api/docs/live-api?lang=python)
  specifies JPEG image input at up to **1 FPS**; token/billing implications depend
  on the selected model and modality. This GAMST adapter currently sends no video.
- Model availability, quotas, data-use terms, and prices can change. The unit
  tests use fake sessions and do not establish live-service availability, model
  quality, latency, or cost.
