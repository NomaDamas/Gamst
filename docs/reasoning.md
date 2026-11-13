# Evidence-citing interaction analysis

The optional `ChatCompletionsBackend` connects a local model server or an explicitly authorized remote provider implementing the JSON chat-completions interface. It analyzes supplied transcripts and scalar measurements. It does not send raw camera/audio buffers or identify people. No provider model is installed or downloaded by gamst.

```python
from humansense.reasoning import ChatCompletionsBackend
human.set_analysis_backend(ChatCompletionsBackend(
    endpoint="http://127.0.0.1:11434/v1/chat/completions",
    model="YOUR_INSTALLED_MODEL_NAME",
))
result = await human.analyze(window="5m")
```

The model must support a JSON completion with the schema in `humansense.reasoning.SYSTEM_PROMPT`. Not every server/model supports `response_format`; compatibility requires a real local/remote endpoint test. The model name above is a placeholder for the model you actually run.

Returned outputs:

- `conversation.expressions`: recognized expression-pattern scores, never inner-state probabilities.
- `conversation.hypotheses`: clarification, shorter-answer/help or expressed-interest/confusion hypotheses with record IDs.
- `consistency.candidates`: possible free-text contradictions supported by two distinct exact source excerpts; not established facts or deception judgments.
- `behavior`: the SDK's measured personal trends, not language-model invented baselines.
- `recommended_action`: an evidence-linked question/offer selected by the SDK; it is not executed automatically.

Strict parsing rejects unknown fields, invented evidence, claimed calibrated probabilities, unsupported labels and inaccurate excerpts. This validates output format and citations, not the semantic truth of a hypothesis. Empty lists are valid abstention. Participants remain the authority on their feelings and needs.

## MCP / dashboard setup

Set these environment variables in the terminal or your MCP server environment:

```bash
export HUMANSENSE_REASONING_URL=http://127.0.0.1:11434/v1/chat/completions
export HUMANSENSE_REASONING_MODEL=YOUR_INSTALLED_MODEL_NAME
python -m humansense.mcp
# or python -m humansense dashboard
```

Use the MCP `analyze` tool or the dashboard's model-analysis button. Collection and rule-based events remain separate so continuous camera/audio capture does not wait on an LLM response. Repeated calls are initiated by the application; the current demo does not continuously invoke a slow model on every frame.

For an authorized remote provider, HTTPS and explicit `HUMANSENSE_REASONING_ALLOW_REMOTE=true` are required. `HUMANSENSE_REASONING_KEY_ENV=YOUR_PROVIDER_KEY_VARIABLE` names a separate environment variable holding the key. Obtain participant consent to transmit transcripts/measurements before enabling remote analysis. Credentials are never included in observations. Disable remote configuration to remain local. Redirects are disabled.

No remote API call or live local model execution has been verified in this socket-restricted environment. Tests validate actual request serialization and response handling through an injected HTTP opener, plus runtime integration. Hardware/model validation is still required.
