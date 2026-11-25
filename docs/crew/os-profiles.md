# OS profiles

`gamst.crew.os_profiles.profile()` reports OS, architecture, Python and accelerator hints. This is discovery only; a runtime must declare compatibility and readiness explicitly before it is used.

## Profile identity and launch responsibility

A profile describes a provider runtime boundary, not a promise that the host
can run every provider. The catalog currently names five deployment families:

- `macos-arm64`: `uv+onnx+mlx+llama.cpp`; typically a host subprocess with
  Apple Silicon libraries.
- `linux-cuda`: `docker+nvidia-toolkit+triton`; a container or wrapper that
  supplies the tested CUDA stack.
- `linux-cpu`: `uv+onnx+llama.cpp`; typically a host subprocess with CPU
  dependencies.
- `windows-cpu-directml`: `uv+onnx+llama.cpp`; typically a host subprocess
  with CPU/DirectML dependencies.
- `jetson-arm64`: `jetpack+tensorrt+gstreamer`; a host subprocess or
  JetPack-managed wrapper.

The runtime hint tells deployment which environment family must be prepared.
It does not make `launch_profile()` a package manager or container builder.
`launch_profile()` validates the selected profile, then supervises the exact
`argv` supplied by the caller. For `linux-cuda`, that `argv` may be a tested
  `docker run` command; for the other profiles it is commonly a Python worker
  command. A `ProcessRuntime` command must emit Crew's bounded readiness line
  and packet responses. A provider worker itself runs through GAMST's `ProcessProvider`
  handshake instead; both paths preserve the packet contract.

Use a container where the provider needs image-level reproducibility or a
Linux/CUDA/Triton userspace that should not be merged with the application.
Use a host subprocess where the profile's native environment already satisfies
the provider's lock and accelerator contract. A container adds namespaces and
an image boundary; a subprocess still gives each provider an independent
interpreter and crash boundary without claiming container-level isolation.

Crew does not put all providers into one image by default. Their Python and
native ABI requirements, accelerator stacks, artifact locks and upgrade cycles
can differ. Profiles keep those dependencies and failure domains explicit.
Compatible providers may share a deliberately tested image, but that is a
deployment decision recorded by the profile rather than an automatic merge.

Profiles may additionally declare `lockfile` and `artifact` paths. The optional
`ProviderProfile.verify_artifacts(root)` helper verifies SHA-256 only when a
declared file exists under the supplied root; missing files remain explicitly
`unmeasured`, and a mismatch fails closed. Python metadata must use
`3.11`/`python 3.11`-style version syntax. Metadata alone is never presented as
installed or benchmark-accepted evidence.

## Profile manifests

`gamst/crew/profiles/runtime-profiles.example.json` shows the required metadata shape. It is an
example only: `model_revision: manifest-pending` and the zero hash are not a
claim of installed weights or benchmark acceptance. A real profile must replace
both with a pinned revision and dependency-lock SHA256 before launch.

Launch also validates declared architecture metadata and verifies every
declared lockfile or artifact before starting a child process. A declared file
that is missing or mismatched fails closed; profiles without local artifacts
remain explicitly unmeasured and still require target-host acceptance.
When the host descriptor includes its Python version, launch compares the
major/minor pair with the profile's pinned Python metadata and rejects a
mismatch before spawning a child. A host descriptor without Python remains
backward-compatible and leaves that check unmeasured.
## Declarative profile catalog and diagnostics

Crew exposes `profile_catalog()`, `provider_compatibility()` and
`install_diagnostics()` for the five supported execution profiles: macOS
Apple Silicon, Linux CUDA, Linux CPU, Windows CPU/DirectML and Jetson ARM64.
Each provider reports `available`, `installed`, `runtime-compatible`,
`weight-ready` and `benchmark-accepted`. Diagnostics only inspect commands and
Python modules; they never install packages or download weights. Host
acceptance remains explicitly `unmeasured` until a target machine run is
attached.
