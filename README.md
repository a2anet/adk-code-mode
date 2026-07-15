# ADK Code Mode

A [Code Mode](https://blog.cloudflare.com/code-mode/) sandboxed code-execution tool for [Agent Development Kit (ADK)](https://github.com/google/adk-python).

`ExecuteCodeTool` is a regular ADK `BaseTool` — add it to `tools=[...]` like any other tool — that lets the model write Python to call tools and read and write files.
Code runs inside a sandboxed container, and tools (and their credentials) are executed on the host.
The base image comes with the stdlib and can be extended with any Python package you want.
The sandboxed container can list, load, and save ADK Artifacts, and any files it creates are returned as artifacts too.

Inspired by Cloudflare's [Code Mode](https://blog.cloudflare.com/code-mode/) and Anthropic's [Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) — both of which keep code execution as one native structured tool call, never disabling function-calling. `ExecuteCodeTool` follows the same shape: the model calls it exactly like any other tool, with normal `tool_use`/`function_call` turn-taking, instead of writing a fenced code block in plain text for ADK to regex out.

## ✨ Features

- **Call ADK tools from sandbox code** — imports against the `tools` package proxy back to the host and run through ADK's `before_tool` / `after_tool` / `on_error` callbacks and the plugin manager exactly as direct tool calls would.
- **Bake any Python package into the image** — extend the published base image with anything the model's code needs to `import`, no runtime `pip install` required.
- **Cross-turn persistence via ADK Artifacts** — `save_artifact` / `load_artifact` / `list_artifacts` are auto-injected and route through your configured `ArtifactService`. Files the code creates or changes are saved as artifacts automatically too.
- **Tool results saved as artifacts** — on by default; every tool's result is persisted as a `code_mode.tool_result` artifact (with optional model-supplied name/description) so hosts can forward outputs and large results stay out of the prompt. Opt out with `save_tool_results_as_artifacts=False`.
- **Bounded stdout/stderr** — overflow lands in a session artifact instead of poisoning the prompt.
- **Production-ready remote sandbox** — `RemoteBackend` connects to an isolated per-turn container over WebSocket, reused across the turn's `execute_code` calls. Deploy on any cloud platform (Cloud Run, Fargate, ACI, Kubernetes, Fly.io, etc.).
- **Local development** — `UnsafeLocalDockerBackend` runs the sandbox against your local Docker daemon for fast iteration. **Not for production** — see [Safety](#-safety).

|                                     | BuiltIn | AgentEngineSandbox              | VertexAi                        | Container | Gke | CodeMode                 |
| ----------------------------------- | ------- | -------------------------------- | -------------------------------- | --------- | --- | ------------------------ |
| Call ADK tools from code            | no      | no                                | no                                | no        | no  | yes (with limitations)   |
| Extra Python packages               | no      | no (more than stdlib but fixed)  | no (more than stdlib but fixed)  | yes       | yes | yes                      |
| Variables are stateful              | no      | yes                               | yes                               | no        | no  | yes (within a turn)      |
| Input files                         | no      | yes                               | yes                               | no        | no  | no (use Artifacts)       |
| Output files                        | no      | yes                               | yes                               | no        | no  | yes (as Artifacts)       |
| Storage                             | no      | yes (via variables)               | yes (via variables)               | no        | no  | yes (via ADK Artifacts)  |
| Local development version available | no      | no                                | no                                | yes       | yes | yes                      |
| Bounded stdout/stderr               | no      | no                                | no                                | no        | no  | yes (`max_output_chars`) |
| Native structured tool-calling      | yes     | no                                | no                                | yes       | yes | yes                      |

## 📦 Install

```bash
pip install adk-code-mode
```

Or with uv:

```bash
uv add adk-code-mode
```

For local development with `UnsafeLocalDockerBackend`, install the `docker` extra:

```bash
pip install adk-code-mode[docker]
```

Requires Python 3.10+. Local development requires [Docker](https://docs.docker.com/get-docker/); remote deployment only needs network access to the sandbox URL.

## 🚀 Usage

Build an `ExecuteCodeTool` and add it to the agent's `tools=[...]` like any other tool. Wire `release_invocation` into `after_agent_callback` to release the turn's sandbox container as soon as the turn ends — an idle reaper (`session_idle_timeout_seconds`, default `600`) is a backstop, so containers are still reclaimed without it, just later.

### Production (remote sandbox)

```python
from google.adk.agents import LlmAgent
from adk_code_mode import ExecuteCodeTool, RemoteBackend

tool = ExecuteCodeTool(
    tools=[my_fn_tool, McpToolset(...), OpenAPIToolset(...)],
    backend=RemoteBackend(
        url="https://sandbox-xyz.run.app",  # your deployed sandbox URL
        token="your-secret-token",           # bearer token for auth
    ),
)

async def _release_sandbox(callback_context):
    await tool.release_invocation(callback_context.invocation_id)

root_agent = LlmAgent(
    name="assistant",
    model="gemini-3.5-flash",
    instruction="You are a helpful assistant.",
    tools=[tool],
    after_agent_callback=[_release_sandbox],
)
```

That's it — no `code_executor=`, no `before_model_callback=`, no `generate_content_config` to disable native function-calling. `execute_code` is a normal structured tool call the model can interleave with any other tool, and calls it again in a follow-up turn to keep iterating.

### Local development only

> **`UnsafeLocalDockerBackend` is not safe for production or multi-tenant use.** See [Safety](#-safety).

```python
from adk_code_mode import ExecuteCodeTool, UnsafeLocalDockerBackend

tool = ExecuteCodeTool(
    tools=[my_fn_tool, McpToolset(...), OpenAPIToolset(...)],
    backend=UnsafeLocalDockerBackend(image="ghcr.io/a2anet/adk-code-mode:latest"),
)
```

Inside the sandbox, the model writes code like:

```python
from tools.slack import send_message
print(send_message(channel="C123", text="hi"))
```

## 🌐 Remote Deployment

**Every turn runs in its own container**, which the platform destroys when the turn ends — no cross-turn or cross-tenant state. The sandbox runs as a WebSocket server (set `ADK_CODE_MODE_CONTROL_HTTP=1`) and accepts exactly one connection, so you **must** configure your platform for one container per turn (`--concurrency 1` on Cloud Run, or equivalent).

### Deploy to Cloud Run

```bash
# Push the sandbox image to Artifact Registry
gcloud auth configure-docker <region>-docker.pkg.dev
docker pull --platform linux/amd64 ghcr.io/a2anet/adk-code-mode:latest
docker tag  ghcr.io/a2anet/adk-code-mode:latest \
    <region>-docker.pkg.dev/<project>/<repository>/adk-code-mode-sandbox:latest
docker push <region>-docker.pkg.dev/<project>/<repository>/adk-code-mode-sandbox:latest

# Create a VPC connector with no egress routes (blocks outbound network from sandbox)
gcloud compute networks create adk-sandbox-vpc --subnet-mode=custom
gcloud compute networks subnets create adk-sandbox-subnet \
    --network=adk-sandbox-vpc \
    --region=<region> \
    --range=10.8.0.0/28
gcloud compute firewall-rules create adk-sandbox-deny-all-egress \
    --network=adk-sandbox-vpc \
    --direction=EGRESS \
    --action=DENY \
    --rules=all \
    --priority=1000
gcloud compute networks vpc-access connectors create adk-sandbox-connector \
    --region=<region> \
    --subnet=adk-sandbox-subnet

# Deploy — note --concurrency 1, --vpc-egress=all-traffic, and the /health startup probe
gcloud run deploy adk-code-mode-sandbox \
    --image <region>-docker.pkg.dev/<project>/<repository>/adk-code-mode-sandbox:latest \
    --region <region> \
    --port 8080 \
    --cpu 1 \
    --memory 1Gi \
    --concurrency 1 \
    --timeout 3600 \
    --max-instances 120 \
    --allow-unauthenticated \
    --vpc-connector=adk-sandbox-connector \
    --vpc-egress=all-traffic \
    --set-env-vars "ADK_CODE_MODE_CONTROL_HTTP=1" \
    --set-secrets "ADK_CODE_MODE_AUTH_TOKEN=<your-secret-name>:latest" \
    --startup-probe "httpGet.path=/health,httpGet.port=8080,timeoutSeconds=3,periodSeconds=3,failureThreshold=80"
```

> These flags are **recommendations to tune per deployment**, not hard requirements. `--timeout 3600` (Cloud Run's max) is the per-turn ceiling since the container holds the WebSocket for the whole turn; `--max-instances` should cover your peak concurrent *turns* (`120` covers a 10–100 target — verify your region's Cloud Run vCPU quota). The `/health` startup probe avoids cold-start `HTTP 503`s — Cloud Run's default TCP probe opens a raw socket the WebSocket server rejects.

Then in your agent:

```python
RemoteBackend(
    url="https://adk-code-mode-sandbox-xxxxx.run.app",
    token="<your-secret>",
)
```

> **`--concurrency 1` is critical for security.** It pins one turn to one container. Without this flag, Cloud Run may route multiple turns to the same container. The sandbox rejects the second connection, but the misconfiguration itself is a risk.

> **`--vpc-egress=all-traffic` with a deny-all VPC is critical for security.** Without it, user code can make arbitrary outbound requests — including hitting the GCP metadata endpoint (`169.254.169.254`) to steal the service account token, exfiltrating data, or scanning your VPC. The sandbox only needs to _accept_ inbound connections; it never needs outbound access.

### Deploy on other platforms

The same pattern works on any platform that runs Docker containers as HTTP services (AWS Fargate/ECS, Azure Container Instances, Kubernetes, Fly.io, etc.):

1. **One container per turn.** Each container handles exactly one turn (one or more `execute_code` calls) and exits.
2. **Block all outbound network access.** Without egress restrictions, user code can exfiltrate data, access cloud metadata endpoints, or scan internal networks.
3. **Keep `/workspace` and `/tools` writable.** The sandbox stages the working directory and materialises the `tools` package into `/tools` at connect time. If you set a read-only root filesystem (e.g., `readOnlyRootFilesystem: true` in Kubernetes), mount both as writable volumes (e.g., an `emptyDir`).
4. **Authenticate connections.** Set `ADK_CODE_MODE_AUTH_TOKEN` and layer platform-level auth (IAM, NetworkPolicy, security groups) on top.

Required env vars:

| Env var                              | Required | Default | Purpose                          |
| ------------------------------------ | -------- | ------- | -------------------------------- |
| `ADK_CODE_MODE_CONTROL_HTTP`         | yes      | —       | Set to `1` to run the sandbox as a WebSocket server (required for remote) |
| `ADK_CODE_MODE_AUTH_TOKEN`           | yes      | —       | Bearer token for WebSocket auth  |
| `PORT`                               | no       | `8080`  | Listen port                      |
| `ADK_CODE_MODE_MAX_UPLOAD_TOOLS`     | no       | 100 MiB | Max tools tar archive size       |
| `ADK_CODE_MODE_MAX_UPLOAD_WORKSPACE` | no       | 100 MiB | Max workspace tar archive size   |

Connection tuning, retry, and the same upload limits (plus a download limit) are configurable on `RemoteBackend`:

```python
RemoteBackend(
    url="...",
    token="...",
    connect_timeout=10.0,             # seconds to wait for the WS handshake (default)
    start_attempts=3,                 # connect attempts before giving up (default)
    start_retry_delay_seconds=1.0,    # linear backoff base: delay * attempt (default)
    start_retry_jitter_seconds=0.25,  # uniform jitter added per retry (default)
    max_upload_tools_bytes=100 * 1024 * 1024,       # 100 MiB (default)
    max_upload_workspace_bytes=100 * 1024 * 1024,    # 100 MiB (default)
    max_download_workspace_bytes=100 * 1024 * 1024,  # 100 MiB (default)
)
```

## 🗂️ Storage

Code Mode exposes two file surfaces:

- **The working directory** — the turn's workspace. It persists across the turn's `execute_code` calls and resets between turns. Files created or changed by a call are collected afterward and saved as session artifacts automatically, returned to the model as a list of filenames (reloadable via `load_artifact`) — nothing is re-hydrated into the working directory on the next turn unless the model explicitly loads it back with `load_artifact`.

- **ADK Artifacts** — persistent cross-turn storage. `ExecuteCodeTool` injects three tools into the sandbox:

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

Pass `include_artifact_tools=False` to opt out. To react when the model saves an artifact, pass `on_artifacts_saved`:

```python
async def on_saved(invocation_context, delta):
    # delta is {filename: version} for everything the sandbox-side save_artifact
    # calls (or wrapped tool results) saved this turn.
    ...

ExecuteCodeTool(tools=..., backend=..., on_artifacts_saved=on_saved)
```

### Tool results as artifacts

By default (`save_tool_results_as_artifacts=True`) every non-artifact tool is wrapped so its return value is saved as a session artifact tagged `code_mode.tool_result = "true"`. This lets a host forward tool outputs to the user (read the marker in `on_artifacts_saved`) and keeps large results out of the model's context — a result whose serialised form exceeds the threshold is replaced in the reply with a short note pointing at the artifact, which the model reloads with `load_artifact`.

Naming is transparent: the filename is derived from the tool name and call id. Each wrapped tool also gains two **optional** parameters — `artifact_name` and `artifact_description` — that the model may pass to name/describe the saved artifact; both land in the artifact's `custom_metadata` (`code_mode.artifact_name` / `code_mode.artifact_description`) alongside the marker. Set `save_tool_results_as_artifacts=False` to return tool results inline without persisting them.

```python
ExecuteCodeTool(tools=..., backend=..., save_tool_results_as_artifacts=True)
```

## 🐳 Sandbox Image

The published base image (`ghcr.io/a2anet/adk-code-mode`) works as-is for tools whose execution is fully host-side. To bake in extra Python packages:

```dockerfile
FROM ghcr.io/a2anet/adk-code-mode:latest
RUN pip install --no-cache-dir pandas==2.2.*
```

The same image works for both `RemoteBackend` and `UnsafeLocalDockerBackend`. To build directly from this repo, run `make docker-image`.

## ⚙️ Configuration

All settings are `ExecuteCodeTool` constructor arguments:

| Argument | Default | Purpose |
| --- | --- | --- |
| `append_function_stubs_to_system_instruction` | `True` | Appends a `<code-mode>` block listing every available function's signature and docstring to the system instruction on every model turn. If the rendered block would exceed `max_catalog_chars`, **nothing** is appended for that turn — no block, no fallback message either. The model can always fall back to discovering functions by listing `/tools/` and reading a stub's docstring from within the code it runs. |
| `max_catalog_chars` | `50_000` | Only consulted when `append_function_stubs_to_system_instruction` is true; see above. |
| `max_output_chars` | `50_000` | Caps stdout/stderr handed back to the model. Overflow is saved as a session artifact at `code_mode/stdout/<call-id>.txt` and the model sees a head-and-tail view pointing to it. |
| `max_code_chars` | `1_000_000` | Rejects oversized code payloads before starting a container. |
| `timeout_seconds` | `None` | Caps overall execution time of one `execute_code` call. Defaults to the platform request timeout (e.g. Cloud Run `--timeout`); set explicitly for defense in depth. |
| `per_tool_timeout_seconds` | `None` | Caps each individual tool call made from within the sandbox. |
| `session_idle_timeout_seconds` | `600` | Idle reaper: closes a turn's container once it goes untouched this long. Backstop for turns that never call `release_invocation`. |

The model can read spilled stdout back from the overflow artifact:

```python
from tools import load_artifact
spilled = load_artifact(filename="code_mode/stdout/<call-id>.txt")
print(spilled["data"][-2000:])
```

### Turn-scoped sessions

A sandbox container is held open for one **turn** (one ADK invocation) and reused across that turn's `execute_code` calls, so cold start is paid at most once per turn. Python globals **and** the working directory persist across a turn's calls and reset between turns — use ADK Artifacts for cross-turn persistence. The container is released when the turn ends (via `await tool.release_invocation(...)`, wired in [Usage](#-usage)) and destroyed by the platform, so no state survives into another turn or tenant.

## 🏗️ Architecture

**Host wheel (`adk-code-mode`).** Lives in the same process as your `LlmAgent`. `ExecuteCodeTool.process_llm_request` resolves tools and, when `append_function_stubs_to_system_instruction` is enabled, renders the catalog and appends it to the system instruction. At execution time (`run_async`), it generates a `tools/` Python package of thin stubs, stages the working directory, and opens (or reuses) the turn's sandbox connection — the container spans the whole turn.

**Sandbox wheel (`adk-code-mode-sandbox`).** Pre-installed in the container image. When model code calls a stub, it sends a JSON-Lines frame over the control connection; the host runs the real tool (with callbacks and plugins) and sends the result back.

The only things crossing the boundary are: code, tool call arguments, tool return values, and log frames.

| Backend                    | Transport              | Multi-tenant safe? | When to use                     |
| -------------------------- | ---------------------- | ------------------- | -------------------------------- |
| `RemoteBackend`            | WebSocket over HTTPS   | **Yes**            | Production — any cloud platform |
| `UnsafeLocalDockerBackend` | TCP over Docker bridge | No                  | Local development only          |

### What the model sees

`execute_code` is declared like any other tool — a `FunctionDeclaration` with a single `code: string` parameter and a short, fixed description of what it does. With `append_function_stubs_to_system_instruction` enabled (the default), the system instruction also gets a `<code-mode>` block appended on every turn:

~~~
…your instruction…

Reference catalog of the functions available inside execute_code's sandbox.
These are not separate callable tools — they are Python functions to import
and call from within the code you pass to execute_code (e.g.
`from tools.slack import send_message`).

<code-mode>

# tools.slack

from tools.slack import list_channels, send_message

def list_channels() -> Any:
    """List Slack channels."""
    ...

def send_message(*, channel: str, text: str, thread_ts: str | None = ...) -> Any:
    """Send a message to a Slack channel."""
    ...

# tools

from tools import save_artifact, load_artifact, list_artifacts
…

</code-mode>
~~~

With `save_tool_results_as_artifacts` enabled (the default), each non-artifact tool above — e.g. `list_channels` and `send_message` — also carries two optional `artifact_name: str | None = ...` / `artifact_description: str | None = ...` parameters for naming its saved result.

If the rendered catalog would exceed `max_catalog_chars`, nothing is appended to the system instruction that turn — not even a fallback note. The model can still navigate the sandbox from Python:

```python
import pathlib
print(list(pathlib.Path("/tools").iterdir()))
print(open("/tools/slack/send_message.py").read())  # signature + docstring
```

Text and JSON-like MIME types travel as plain strings in artifact tools; binary content is base64-encoded. `load_artifact` returns `{"kind": "text" | "bytes", "data": str, "mime_type": str | None}`.

## 🛡️ Safety

### `RemoteBackend` (production)

`RemoteBackend` is designed for multi-tenant production use where untrusted users submit arbitrary Python code:

- **One container per turn (one tenant, one invocation).** Within a turn the process/filesystem are reused across that turn's `execute_code` calls; the container is destroyed at turn end, with **no cross-turn or cross-tenant sharing**.
- **Environment sanitization.** All env vars are stripped except a safe allowlist (`PATH`, `HOME`, `USER`, locale vars, Python config) before user code runs.
- **Credentials never enter the sandbox.** API keys, OAuth tokens, and connection strings stay in the host process. The container only receives tool results.
- **Bearer token authentication.** WebSocket connections without a valid token are rejected. Always set `ADK_CODE_MODE_AUTH_TOKEN` and layer platform-level auth on top.
- **Hardened tar extraction.** Path traversal (`../`), symlinks, hardlinks, and absolute paths are rejected.
- **Non-root user.** The sandbox runs as `sandbox`, not root.
- **Tool dispatch runs ADK's guard callbacks.** `before_tool`, `after_tool`, `on_error`, and the plugin manager all fire normally.
- **Bounded inputs and outputs.** See [Configuration](#-configuration) for `max_code_chars`, `max_output_chars`, `timeout_seconds`, `per_tool_timeout_seconds`, and upload/download size limits.

### `UnsafeLocalDockerBackend` (development only)

> **Do not use in production or for multi-tenant workloads.**

Named "Unsafe" intentionally: it binds a TCP listener on `0.0.0.0`, communicates over unencrypted TCP, and relies on the local Docker daemon. It does still sanitize env vars, run as non-root, drop all Linux capabilities (`cap_drop=["ALL"]`), and mount the root filesystem read-only — but it is not a security boundary for untrusted users.

### What this does NOT protect against

- **Network egress (if you skip egress restrictions).** The sandbox does NOT block outbound network by itself — configure this at the platform level. Without it, user code can exfiltrate data, access cloud metadata endpoints (`169.254.169.254`), or scan internal networks. See [Remote Deployment](#-remote-deployment).
- **Container runtime escapes.** Keep your container runtime patched.
- **Exfiltration through legitimate tool calls.** If your tool surface includes `send_email`, a prompt-injected payload could use it. Keep your tool surface least-privilege.
- **Denial of service within resource limits.** User code can consume its full CPU/memory allocation. Set platform-level limits.

## ⚠️ Limitations

- **No credential-requesting tools.** Tools that need ADK to request credentials, confirmations, UI widgets, agent transfer, escalation, or that yield without an immediate response are rejected with a structured error.
- **State is turn-scoped.** Variables and the working directory persist across `execute_code` calls **within** a turn, but reset between turns. Use `save_artifact` / `load_artifact` to persist across turns.
- **No runtime package installation.** The sandbox ships with the Python Standard Library and the runtime's own dependencies only. Extra packages must be baked into the image at build time.

## ⬆️ Migrating from `CodeModeCodeExecutor`

`CodeModeCodeExecutor` (a `BaseCodeExecutor`), `code_mode_before_model_callback`, and `CODE_MODE_SYSTEM_INSTRUCTION` are gone, replaced by a single `ExecuteCodeTool` (a `BaseTool`):

- Drop `code_executor=`, `before_model_callback=code_mode_before_model_callback(executor)`, and the `generate_content_config=` block that disabled native function-calling — none of that is needed anymore.
- Drop `CODE_MODE_SYSTEM_INSTRUCTION` from your agent's `instruction=`.
- Add the tool to `tools=[...]` instead: `ExecuteCodeTool(tools=[...], backend=...)`.
- `release_invocation` is now `async` — `await tool.release_invocation(callback_context.invocation_id)` in `after_agent_callback`.
- The model now calls `execute_code(code=...)` as a normal structured tool call instead of writing a fenced ` ```python ` block in plain text — this is the actual fix: it removes the `MALFORMED_FUNCTION_CALL` failure mode some Gemini model versions hit when function-calling is disabled, and gives the model a clean `tool_use`-style stop signal each call.
- The tool catalog now defaults to being injected upfront (`append_function_stubs_to_system_instruction=True`); pass `False` for the old progressive-disclosure-only behavior.
- A code block's changed output files are now returned as a list of artifact filenames (reloadable via `load_artifact`) instead of raw bytes on `CodeExecutionResult.output_files`.

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
