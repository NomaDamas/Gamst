# Call controller runtime and MCP integration

Register a fresh controller explicitly with `human.register_call_controller('main', controller)`. Up to eight controllers may be registered; IDs must be unique. Registration requires consent and a non-paused, open Gamst instance. A controller cannot bind to a second runtime or be reused after standalone source activity. Registration never starts a model, device, transport or periodic delivery task.

`pause()`, `reset_processing()` and the normal plugin reset path close source handles, cancel undispatched reservations and discard future output from already-dispatched requests. Resume requires opening new sources with newly registered observations. Assigning `human.consent = False` invalidates the processing generation and closes controller sources; granting consent again does not revive old handles.

`delete_data(subject_id)` removes that subject's requests, source associations, retained history and observation bodies. Old result acknowledgements are rejected. Gamst's global processing reset closes other source handles and suppresses their pending results, but does not erase other subjects' stored observations. Closing Gamst forgets sensory associations and closes admission. Application-owned dispatch copies, external providers and transport buffers have their own deletion obligations.

Python and MCP expose `get_call_status(controller_id=None, limit=100, max_bytes=65536)`. It returns bounded diagnostic entries for controllers, requests, recent history and sources: desired routes, blockers, capture/source evidence, request states and admission provenance. It does not expose full observation values, prompt text or generated responses. No remote MCP tool creates controllers or reserves/dispatches calls.

The entry limit is 1–100 and the aggregate serialized entry budget is 1024–262144 bytes, excluding the response envelope. Whole entries are retained or omitted; `total_count`, `returned_count`, `omitted_count` and `truncated` are explicit. Select `controller_id` to narrow results. This is a bounded current snapshot, not stable pagination or a durable audit log. MCP polling is not automatic Context injection into an LLM.

Run `python -m examples.call_controller_demo` from the repository root. The example uses registered observations, controller admission, evidence/prompt hashes, application-owned async invocation, result acknowledgement and status retrieval. Its callback is explicitly synthetic: no model weights, paid endpoint or camera are used. Replace the callback and evidence capture with verified application implementations.

A failed or cancelled transport should still acknowledge failure through `finish`. An invalidated result is discarded rather than published. The controller suppresses output acceptance but does not terminate an application-owned external request. Callback success and fixture tests do not establish real-model quality.
