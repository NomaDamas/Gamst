# Context-aware Companion

The Companion uses the same gamst runtime and retention store as perception. A real configured chat-completions endpoint generates the assistant's response; there is no canned fallback pretending to be a model. Generation is not proof of the reply's factual correctness.

```python
from humansense import HumanSense
from humansense.companion import Companion, LocalChatBackend

async with HumanSense(consent=True) as human:
    backend = LocalChatBackend(
        "http://127.0.0.1:11434/v1/chat/completions",
        "YOUR_INSTALLED_MODEL_NAME",
    )
    companion = Companion(human, backend)
    result = await companion.reply("설명이 너무 길어요. 핵심만 알려 주세요.")
    print(result["assistant_text"])
```

The prompt includes bounded prior turns, measured aggregate context and the recommended strategy. Inferences are explicitly identified as unverified hypotheses. If a reasoning backend is also configured, it first performs evidence-citing analysis; otherwise it uses the runtime's rule-based inference. The generated assistant turn is recorded in the same local history and serves as a timing anchor for the next user response. It does not execute outside tools or diagnose a person.

## Dashboard and MCP

Configure a provider you actually run:

```bash
export HUMANSENSE_COMPANION_URL=http://127.0.0.1:11434/v1/chat/completions
export HUMANSENSE_COMPANION_MODEL=YOUR_INSTALLED_MODEL_NAME
python -m humansense dashboard
```

Consent and start collection, enter a message, and click **Companion에게 보내기**. For MCP, start `python -m humansense.mcp` with the same environment and call `chat` with `{"text":"설명을 짧게 해 주세요"}`. WebSocket clients call the same tool. The UI does not automatically upload camera recordings to the provider or continuously call a model for every frame.

If separate Companion variables are absent, an explicitly configured reasoning endpoint/model is reused. If neither is configured, chat reports a missing-backend error. For an external provider use HTTPS, `HUMANSENSE_COMPANION_ALLOW_REMOTE=true`, and optionally `HUMANSENSE_COMPANION_KEY_ENV=YOUR_KEY_VARIABLE`. Obtain consent to transmit conversation/context before enabling remote service. Keys remain in the environment and are not observations.

Stop/delete invalidates in-flight model results so a late answer cannot restore deleted or paused observations. User-visible transcripts already displayed by other applications or remote provider copies are outside the SDK's deletion boundary.

Tests exercise real runtime/context coordination with an injected backend and actual request serialization through an injected HTTP opener. No live model endpoint has been reachable in the development sandbox; generated reply quality, provider compatibility and hardware/video interaction still need host validation.
