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
and similar names, without regard to case or separators, are refused in registered values and dropped from compact context, and
so are `landmarks` and `world_landmarks`. Unregistered signals refuse all of them too. The name guard is the second line; the type and the
separate wire part are the first.

`gamst.separation_media` (SAM-Audio pairs with ownership and revocation) is older, stricter about its windows and unchanged.

## The local screen

The in-process `MediaPipeFacePlugin` and `MediaPipePosePlugin` (`gamst dashboard`, MCP `start_capture` with `face_model_path`)
and the MediaPipe graph module no longer put points into records: `scene.faces`, `scene.poses`, `scene.hands` and
`scene.face_landmarks` hold boxes, geometry flags, blendshapes and `landmark_count`. Of a custom landmark stream of a
graph only `landmark_counts` is recorded. The points go two other ways:

- **Derived inputs**, when a Pipeline asks: `MediaPipePosePlugin(joints=NAME)`, `MediaPipeGraphModule(derived={'poses': NAME,
  'hands': NAME, 'face': NAME, '<stream>': NAME})`, recipe part `mediapipe-graph` with `"joints"` / `"derived"`.
- **`gamst.visualization.Drawing`**, per plugin, in memory: the newest two frames, at most 4 items of 1024 points, 5 s, not
  serializable, cleared by pause, stop and `delete_data`. Only `guide_packet` reads it, for the dashboard page of the same
  session (loopback, session token), and only for the frame of a record that is in the Store.

**Existing store files are not changed.** Records written before this keep their `landmarks`, and MCP `get_state` /
`inspect_interval` return them as they are. Only `delete_data` or retention removes them.

## Where raw data can still enter a record or a file

1. **Third-party plugins**: `push_signal` accepts any JSON; only the names above are refused. Bare lists of numbers, or
   coordinates under another name, pass. `push_claim`, `push_text` and `push_semantic_state` take what the caller sends.
2. **MediaPipe graph**: custom streams of type string, number or rectangle are recorded as they are.
3. **Derived numbers that are records by design**: pose, face and hand boxes (their edges are extreme points), joint angles and
   wrist offsets (`vision.pose_channels`), face and voice similarity scores, gaze and head angles, blendshapes, outlines
   of segmented things (`scene.segments` polygons, SAM 3, a person if that is the prompt), positions of tracked objects.
4. **Files the owner asks for**: enrollment files hold embeddings and a consent record (mode 0600); GAMST 240 takes hold raw video.
   Movement files (`gamst recipe enroll`, mode 0600) hold joint angles and wrist offsets of the demonstrations, or a rule.
   Proposal files hold intervals, texts and the path and SHA256 of the recording. Store files from before, see above.
5. **Memory only**: references and galleries hold embeddings; derived inputs hold crops, joints and separated audio while the
   pipeline runs; a movement enrolled while a recipe runs is kept as channel vectors by the run and not written to a file.
   `get_pipeline` shows none of it.
6. **Browser playground** (`site/`): MediaPipe runs in the page; its points stay there and reach no GAMST process.
