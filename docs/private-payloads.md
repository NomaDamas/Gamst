# Worker-private payloads

Joint positions, identity embeddings and separated waveforms are answers for one module, never context.
`gamst/private_payload.py` is the one mechanism for them.

- The owning module opens the worker with `ProcessProvider.open(..., private=True)` and calls `query(frame, request)`.
  The worker answers the distinct wire operation `query` only; `process`, `process_regions`, `finish` and `endpoint` fail.
- The answer has two parts. The public value is validated by the profile's `CapabilitySpec` (boxes, counts, "usable").
  The private array travels as bytes beside it and is never a `CapabilitySpec` field.
- `CONTRACTS` declares each profile's kind, shape and value bound; worker and owner both check it, before any private byte
  is read, and check that both parts describe the same answer.

| Profile | Private payload | Owner |
|---|---|---|
| `mediapipe.body.video` (also has a public mode) | 33 x (x, y, visibility, presence), at most 4 people | `PoseWorkerPlugin` |
| `sapiens2.pose` | 308 x (x, y, visibility), at most 8 people | `Sapiens2PosePlugin` |
| `speaker.embedding`, `face.embedding`, `body.embedding` | one vector, 16..4096 | `SpeakerVerify`, `TargetPerson`, enrollment |
| `target_speech.local` | samples, exactly as many as the block | `TargetSpeech`, `SAMSpanSpeech` |

`ExemplarEvents` reads a joints input only for a rule that uses joint relations; its record is one violation number.

`Private` (the owner's handle) and `PrivatePoses` (joints on a derived input such as `pose_joints`) are not JSON, cannot be
pickled or copied, and `push_signal` / `push_observation` refuse them. Private workers are refused by
`SensorStreams`, `Pipeline.add_module`, `Gamst.register`, `WorkerModule` and `role_module` (so `ProfileRuntime`).

Name guard (`gamst.capabilities.is_raw_key`): `keypoints`, `joints`, `skeleton`, `*_landmarks`, `embedding`, `waveform`, `pcm`
and similar names, without regard to case or separators, are refused in registered values and dropped from compact context exactly
like `landmarks`. Unregistered signals refuse them too. The name guard is the second line; the type and the
separate wire part are the first.

`gamst.separation_media` (SAM-Audio pairs with ownership and revocation) is older, stricter about its windows and unchanged.

## Where raw data can still enter a record or a file

Measured with a fake record: values under `landmarks` in an unregistered signal are stored and returned by MCP `get_state`
and `inspect_interval`; `get_context` does not return them.

1. **In-process `MediaPipeFacePlugin` and `MediaPipePosePlugin`** (`gamst dashboard`, MCP `start_capture` with
   `face_model_path`): `scene.faces` / `scene.poses` keep `landmarks`; the dashboard overlay draws them.
2. **MediaPipe graph module**: `scene.poses` (with `world_landmarks`), `scene.hands`, `scene.face_landmarks`, and custom landmark
   streams, whose values are bare point lists that no name guard can see.
3. **Third-party plugins**: `push_signal` accepts any JSON; only the names above are refused, and `landmarks` is not.
   `push_claim` and `push_text` take what the caller sends.
4. **Derived numbers that are records by design**: pose boxes (their edges are extreme joints), joint angles and
   wrist offsets (`vision.pose_channels`), face and voice similarity scores.
5. **Files the owner asks for**: enrollment files hold embeddings and a consent record (mode 0600); GAMST 240 takes hold raw video.
   Movement files (`gamst recipe enroll`, mode 0600) hold joint angles and wrist offsets of the demonstrations, or a rule.
   Proposal files hold intervals, texts and the path and SHA256 of the recording.
6. **Memory only**: references and galleries hold embeddings, derived inputs hold crops, joints and separated audio while the
   pipeline runs. `get_pipeline` shows none of it.
