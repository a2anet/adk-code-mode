# ADK Code Mode

A [Code Mode](https://blog.cloudflare.com/code-mode/) code executor for [Agent Development Kit (ADK)](https://github.com/google/adk-python).
The `CodeModeExecutor` allows ADK to write Python code to call tools and list, save, and load Artifacts.

The code is executed in a Docker container, tool calls are forwarded to the host, and are run through ADK's normal tool pipeline (callbacks, plugins, error handling).
The default Docker image supports The Python Standard Library, extra Python packages can be added by building a custom Docker image.
Files can be added to a single execution with `input_files`, and saved files are output as `output_files`.
By default, `CodeModeExecutor` adds `list_artifacts`, `save_artifact`, and `load_artifact` tools to the execution environment to use files between executions.

Inspired by Cloudflare's [Code Mode](https://blog.cloudflare.com/code-mode/) and Anthropic's [Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp).

## ✨ Features

- **Call ADK tools from sandbox code** — imports against the `tools` package proxy back to the host and run through ADK's `before_tool` / `after_tool` / `on_error` callbacks and the plugin manager exactly as direct tool calls would.
- **Credentials stay on the host** — API keys, OAuth tokens, and connection strings never enter the container; only tool arguments and return values cross the boundary.
- **Bake any Python package into the image** — extend the published base image with anything the model's code needs to `import`, no runtime `pip install` required.
- **Cross-turn persistence via ADK Artifacts** — `save_artifact` / `load_artifact` / `list_artifacts` are auto-injected and route through your configured `ArtifactService`.
- **Bounded stdout/stderr** — overflow lands in a session artifact instead of poisoning the prompt.
- **Local development** — `DockerRuntime` runs the sandbox against your local Docker daemon for fast iteration.

|                                     | BuiltInCodeExecutor | AgentEngineSandboxCodeExecutor  | VertexAiCodeExecutor            | ContainerCodeExecutor | GkeCodeExecutor | CodeModeExecutor         |
| ----------------------------------- | ------------------- | ------------------------------- | ------------------------------- | --------------------- | --------------- | ------------------------ |
| Call ADK tools from code            | no                  | no                              | no                              | no                    | no              | yes (with limitations)   |
| Extra Python packages               | no                  | no (more than stdlib but fixed) | no (more than stdlib but fixed) | yes                   | yes             | yes                      |
| Variables are stateful              | no                  | yes                             | yes                             | no                    | no              | no                       |
| Input files                         | no                  | yes                             | yes                             | no                    | no              | yes                      |
| Output files                        | no                  | yes                             | yes                             | no                    | no              | yes                      |
| Storage                             | no                  | yes (via variables)             | yes (via variables)             | no                    | no              | yes (via ADK Artifacts)  |
| Local development version available | no                  | no                              | no                              | yes                   | yes             | yes                      |
| Bounded stdout/stderr               | no                  | no                              | no                              | no                    | no              | yes (`max_output_chars`) |

## 📋 Requirements

- Python 3.10+
- [Docker](https://docs.docker.com/get-docker/) — `DockerRuntime` launches the sandbox via the local Docker daemon, so the agent process needs to reach a Docker socket.
- (optional) [uv](https://docs.astral.sh/uv/) if you prefer it over pip.

## 📦 Install

```bash
pip install adk-code-mode
```

Or with uv:

```bash
uv add adk-code-mode
```

## 🚀 Usage

Build a `CodeModeExecutor`, wire `code_mode_before_model_callback` into the agent, and put `CODE_MODE_SYSTEM_INSTRUCTION` somewhere in the agent's `instruction`:

```python
from google.adk.agents import LlmAgent
from adk_code_mode import (
    CODE_MODE_SYSTEM_INSTRUCTION,
    CodeModeExecutor,
    DockerRuntime,
    code_mode_before_model_callback,
)

executor = CodeModeExecutor(
    tools=[my_fn_tool, McpToolset(...), OpenAPIToolset(...)],
    runtime=DockerRuntime(image="ghcr.io/a2anet/adk-code-mode:0.1.0"),
)

root_agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-pro",
    instruction=f"You are a helpful assistant.\n\n{CODE_MODE_SYSTEM_INSTRUCTION}",
    tools=[],  # do NOT also bind tools here; the executor owns them.
    code_executor=executor,
    before_model_callback=code_mode_before_model_callback(executor),
)
```

The callback is what injects the tool catalog. Skip it and the model has no idea what tools exist.

Inside the sandbox, the model writes code like:

```python
from tools.slack import send_message
print(send_message(channel="C123", text="hi"))
```

## 🗂️ Storage

Code Mode exposes two separate file surfaces:

- **`/workspace`** — per-run working directory for ordinary I/O.
- **ADK Artifacts** — persistent, cross-turn files and data via `save_artifact` / `load_artifact` / `list_artifacts`.

### `/workspace`

`/workspace` is the sandbox's current working directory for a single execution. Any ADK `input_files` are staged there by filename before the code runs, so plain paths like `open("input.csv")` work.

Files created or modified under `/workspace` are returned as `CodeExecutionResult.output_files` at the end of the run — these are developer-internal, intended for things like staged inputs or generated reports a downstream tool consumes. They are not re-hydrated next turn unless the code explicitly persists them via `save_artifact`.

### Artifacts

`CodeModeExecutor` injects three regular tools into the catalog so model code can persist files across turns. They appear as top-level `from tools import …` imports:

```python
import json
from tools import save_artifact, load_artifact, list_artifacts

save_artifact(
    filename="report.json",
    content=json.dumps({"status": "ready"}),
    mime_type="application/json",
)
print(list_artifacts())
report = load_artifact(filename="report.json")
if report is not None and report["kind"] == "text":
    payload = json.loads(report["data"])
```

To opt out and supply your own artifact tools (or none), pass `include_artifact_tools=False`.

To react when the model saves an artifact (for example to surface it as an A2A artifact-update event), pass `on_artifacts_saved`:

```python
async def on_saved(invocation_context, delta):
    # ``delta`` is ``{filename: version}`` for everything saved this turn.
    ...

CodeModeExecutor(tools=..., runtime=..., on_artifacts_saved=on_saved)
```

The hook fires once per `execute_code` call, after the sandbox closes, only when the dispatcher recorded at least one save. Exceptions raised inside the hook are logged and swallowed.

## 🐳 Sandbox Image

### Building

The published base image (`ghcr.io/a2anet/adk-code-mode` on GitHub Container Registry) ships with `adk-code-mode-sandbox` already installed and works as-is for any tools whose execution is fully host-side. To bake in extra Python packages the model's code can `import`, extend it from your project's Dockerfile:

```dockerfile
FROM ghcr.io/a2anet/adk-code-mode:0.1.0
RUN pip install --no-cache-dir pandas==2.2.*
```

```bash
docker build -t myorg/code-mode:1.0 .
```

The same tag works for both local `DockerRuntime` and any future cloud runtime. Packages are baked in at build time — there is no runtime `pip install`. To build the local development image directly from this repo instead of the published one, run `make docker-image` (it builds the sandbox wheel and tags `adk-code-mode:local`).

### Deploying with GCP

`DockerRuntime` launches the sandbox via the Docker daemon on the agent host's machine, so the agent process must run somewhere it can reach a Docker socket. Supported targets today:

- **Compute Engine VMs** with Docker installed.
- **GKE pods** that mount the host's Docker socket (or run Docker-in-Docker).

To deploy to a supported GCP environment, mirror the published image into your project's [Artifact Registry](https://cloud.google.com/artifact-registry) and point `DockerRuntime` at the registry tag:

```bash
# One-time per project / region.
gcloud auth configure-docker <region>-docker.pkg.dev
gcloud artifacts repositories create adk-code-mode \
    --repository-format=docker \
    --location=<region>

# Pull, retag, push.
docker pull ghcr.io/a2anet/adk-code-mode:0.1.0
docker tag  ghcr.io/a2anet/adk-code-mode:0.1.0 \
    <region>-docker.pkg.dev/<project>/adk-code-mode/adk-code-mode:0.1.0
docker push <region>-docker.pkg.dev/<project>/adk-code-mode/adk-code-mode:0.1.0
```

Then in the agent:

```python
DockerRuntime(
    image="<region>-docker.pkg.dev/<project>/adk-code-mode/adk-code-mode:0.1.0",
)
```

If you extended the base image to install extra Python packages, push that derived image instead.

## ⚙️ Configuration

### Catalog overflow

For very large tool surfaces the rendered catalog can dominate the prompt. `CodeModeExecutor.max_catalog_chars` (default `50_000`) is a soft cap. When the catalog exceeds it, the callback drops every tool section and replaces it with a short prose note telling the model how to navigate `/tools/` from Python:

```
<tools>
A `tools` package is available in the sandbox. List `/tools/` with
`pathlib.Path('/tools').iterdir()`. Each entry is either a `.py` file
(a top-level tool, importable as `from tools import <name>`) or a
subdirectory (a namespace, with tools importable as
`from tools.<namespace> import <name>`). To see a tool's signature and
docstring, read its `.py` file with `open(...).read()`.
</tools>
```

Tune `max_catalog_chars` for your model's context budget. Pass it on the executor:

```python
CodeModeExecutor(tools=..., runtime=..., max_catalog_chars=20_000)
```

### Output truncation

`max_output_chars=50_000` (default) caps the stdout and stderr handed back to the model. If either stream exceeds the cap, the model sees a head-and-tail view plus an inline marker:

```
Output exceeded 50,000 characters. Try again with a smaller output.
Full stdout saved as artifact: code_mode/stdout/<execution-id>.txt
```

The full stream is saved as a regular session-scoped ADK artifact at the path printed in the marker. The model can recover it on demand the same way it loads anything else:

```python
from tools import load_artifact
spilled = load_artifact(filename="code_mode/stdout/<execution-id>.txt")
print(spilled["data"][-2000:])
```

So oversize output stays out of context but remains addressable. Developers can also fetch the same artifact directly via the configured `ArtifactService`.

## 🏗️ Architecture

ADK Code Mode has two pieces:

**Host wheel (`adk-code-mode`).** Lives in the same Python process as your `LlmAgent`. Extends ADK's `BaseCodeExecutor`. The `before_model_callback` resolves your tools (including any `BaseToolset` instances), renders the catalog, and appends it to the system prompt. The catalog rendering and tool resolution are cached per-invocation so the follow-up `execute_code` call doesn't re-resolve toolsets. At code-execution time, the executor: (a) generates a small `tools/` Python package whose functions are thin stubs, (b) prepends the built-in `save_artifact` / `load_artifact` / `list_artifacts` tools (unless `include_artifact_tools=False`), (c) stages any `input_files` into `/workspace`, and (d) launches the sandbox.

**Sandbox wheel (`adk-code-mode-sandbox`).** A stdlib-only package pre-installed in the container image. When the model's code calls a stub from `tools/…`, the stub sends a JSON-Lines frame over a TCP control connection to the host; the host runs the real tool (including ADK's `plugin_manager` / `before_tool` / `after_tool` callbacks) and sends the result back.

The only things crossing the host ↔ sandbox boundary are: your code, tool call arguments, tool return values, and log frames. Wire format lives in `src/adk_code_mode/runtime/protocol.py` (and a byte-identical copy in the sandbox wheel so the two sides cannot drift).

### What the model sees

The system prompt the model receives is your `instruction` (which contains `CODE_MODE_SYSTEM_INSTRUCTION`) followed by a `<tools>` block appended by the callback:

```
…your instruction…

<tools>

# tools.slack

from tools.slack import list_channels, send_message

def list_channels() -> Any:
    """List Slack channels."""
    ...

def send_message(*, channel: str, text: str, thread_ts: str | None = ...) -> Any:
    """Send a message to a Slack channel.

    Args:
        channel: Channel ID like C123.
        text: Message text.
        thread_ts: Thread timestamp.
    """
    ...

# tools

from tools import save_artifact, load_artifact, list_artifacts

def save_artifact(*, filename: str, content: str, mime_type: str | None = ...) -> int:
    """Save an artifact to the session. Returns the new version number.
    …
    """
    ...

…

</tools>
```

Each module is one section. The first line of each section is the exact import the model should copy. Bodies are `...` placeholders — the on-disk stubs do the real work via the control channel. When the rendered catalog grows past `max_catalog_chars`, the per-tool sections are dropped in favour of the prose note shown in [Catalog overflow](#catalog-overflow). At code-execution time, oversize stdout/stderr is replaced with a head-and-tail view plus a marker pointing at a session artifact (`code_mode/stdout/<execution-id>.txt`) the model can `load_artifact(...)` on demand.

### Artifact wire format

The artifact tools' wire format is JSON. Text and JSON-like MIME types travel as plain strings; binary content is base64-encoded by the model before `save_artifact` and decoded by the model after `load_artifact` (`load_artifact` returns `{"kind": "text" | "bytes", "data": str, "mime_type": str | None}`). All three call ADK's `ToolContext` artifact APIs on the host, so callbacks, plugins, and the configured artifact service run normally.

## 🛡️ Safety

- **Credentials never enter the sandbox.** API keys, OAuth tokens, DB connection strings — anything your tools use — stay in the host process. The container only gets the result of a tool call, not the means to make it. Tool dispatch goes through `src/adk_code_mode/tools/dispatcher.py`.
- **Read-only rootfs by default.** `DockerRuntime(read_only=True)` is the default. The writable mount is `/workspace` for the current run; persistent data goes through host-side ADK artifact APIs.
- **Bounded stdout/stderr.** `max_output_chars` caps what the model sees; overflow lands in an artifact for you, not in the model's context. Prevents runaway printing from poisoning the context or pushing large payloads into chat history.
- **Bounded execution.** `timeout_seconds` caps overall runtime; `per_tool_timeout_seconds` caps each individual tool call.
- **Resource limits.** `DockerRuntime` defaults to `mem_limit="1g"` and one full vCPU (`cpu_period=100_000`, `cpu_quota=100_000`). Override any of them on the runtime to raise or remove the cap.
- **Network posture.** The container reaches the host over `host.docker.internal` for the control channel; outbound traffic otherwise follows Docker's default bridge. For stricter setups, pass a custom `network_mode` via `run_kwargs`.
- **Control channel is token-gated.** Each `DockerRuntime.start()` mints a per-run shared secret and refuses any TCP peer that does not present it as the first line on connect. Defends against a process on the host racing the real sandbox onto the listener.
- **Tool dispatch still runs ADK's guard callbacks.** `before_tool`, `after_tool`, `on_error`, and the plugin manager all fire for calls originating in sandbox code — any allow-list, redaction, or audit-log you already have keeps working.

What this does **not** protect against: sandbox escapes in the container runtime itself, malicious tool implementations you wrote, exfiltration through legitimate tool calls (e.g. `send_email("attacker", ...)`), or side channels over the control pipe. Keep your tool surface least-privilege.

## ⚠️ Limitations

- **Credential-requesting tools are not supported in this release.** Tools or toolsets that need ADK to request credentials should not be exposed through Code Mode yet. Tool calls that request credentials, confirmations, UI widgets, agent transfer, escalation, compaction, agent state, rewind, or that yield without an immediate response (long-running tools) are rejected with a structured tool error — code mode has no resume path for an async function call.
- **`DockerRuntime` deployment targets.** Does **not** work on Cloud Run, Cloud Functions, or Vertex AI Agent Engine — none of those expose a Docker daemon to the workload. Supported targets are Compute Engine VMs and GKE pods that can reach a Docker socket. A managed-sandbox runtime that targets the serverless environments is on the roadmap but not in this release.
- **No state across executions.** Variables defined in one turn don't survive to the next; each `execute_code` call runs in a fresh sandbox. Use `save_artifact` / `load_artifact` to persist across turns, or `/workspace` within a single run.
- **Sandbox is stdlib-only at runtime.** Extra Python packages must be baked into the image at build time; there is no runtime `pip install` from inside the sandbox.

## 🛠️ Development

```bash
make install       # uv sync --group dev
make ci            # ruff + mypy + pytest
```

Docker integration tests are opt-in:

```bash
uv run pytest -m docker
```

## 📄 License

`adk-code-mode` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.

## 🤝 Join the A2A Net Community

A2A Net is a site to find and share AI agents and open-source community. Join to share your A2A agents, ask questions, stay up-to-date with the latest A2A news, be the first to hear about open-source releases, tutorials, and more!

- 🌍 Site: [A2A Net](https://a2anet.com)
- 🤖 Discord: [Join the Discord](https://discord.gg/674NGXpAjU)
