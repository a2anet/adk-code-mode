# Changelog

## [1.3.0](https://github.com/a2anet/adk-code-mode/compare/adk-code-mode-v1.2.0...adk-code-mode-v1.3.0) (2026-07-23)


### Features

* describe the sandbox to the model in a tagged `<code-mode>` block ([#14](https://github.com/a2anet/adk-code-mode/issues/14)) ([662a281](https://github.com/a2anet/adk-code-mode/commit/662a28138e8632b434e5f1ecd52793f7d83ee097))

## [1.2.0](https://github.com/a2anet/adk-code-mode/compare/adk-code-mode-v1.1.0...adk-code-mode-v1.2.0) (2026-07-23)


### Features

* add `custom_metadata` to the `save_artifact` tool ([dfd9a59](https://github.com/a2anet/adk-code-mode/commit/dfd9a59bf1718084a1d5c02cc1ea453269223f16))

## [1.1.0](https://github.com/a2anet/adk-code-mode/compare/adk-code-mode-v1.0.0...adk-code-mode-v1.1.0) (2026-07-15)


### Features

* expose `tools` and `backend` as public properties on `ExecuteCodeTool` ([acfb704](https://github.com/a2anet/adk-code-mode/commit/acfb704d388da5d66e318775e03fe055467679c2))

## [1.0.0](https://github.com/a2anet/adk-code-mode/compare/adk-code-mode-v0.4.1...adk-code-mode-v1.0.0) (2026-07-15)


### ⚠ BREAKING CHANGES

* `CodeModeCodeExecutor`, `code_mode_before_model_callback`, and `CODE_MODE_SYSTEM_INSTRUCTION` no longer exist. Add `ExecuteCodeTool` to the agent's `tools=[...]` instead of wiring `code_executor=`; drop `before_model_callback=`, `generate_content_config=`, and `CODE_MODE_SYSTEM_INSTRUCTION` entirely; `release_invocation` is now `async`. See the README's migration section.

### Features

* replace `CodeModeCodeExecutor` with `ExecuteCodeTool` ([#10](https://github.com/a2anet/adk-code-mode/issues/10)) ([b78572e](https://github.com/a2anet/adk-code-mode/commit/b78572e72446930548760756255390d1a3cc388a))

## [0.4.1](https://github.com/a2anet/adk-code-mode/compare/adk-code-mode-v0.4.0...adk-code-mode-v0.4.1) (2026-07-10)


### Documentation

* note the sandbox writes to `/tools` in remote deployment guide ([96bb45e](https://github.com/a2anet/adk-code-mode/commit/96bb45e4e18d9d54709f4c10b6b09cf3567ee385))

## [0.4.0](https://github.com/a2anet/adk-code-mode/compare/adk-code-mode-v0.3.0...adk-code-mode-v0.4.0) (2026-07-09)


### Features

* save tool results as artifacts with optional model-supplied naming ([76a3671](https://github.com/a2anet/adk-code-mode/commit/76a36713b2c52bd5b8dc51d3669226a47e7108e5))


### Bug Fixes

* align code-mode example with the `<code-mode>` catalog and fix fences ([4a0cc6d](https://github.com/a2anet/adk-code-mode/commit/4a0cc6d0fc99d5425f6682cc489e76c9ddef7d46))
* clarify code-mode tools are a Python library, not function tools ([48d1018](https://github.com/a2anet/adk-code-mode/commit/48d1018152a2ed2a8e7361de6ce0f43b31b429ce))
* nudge the model to call tools via Python in the code-mode block ([141baf3](https://github.com/a2anet/adk-code-mode/commit/141baf3a44976e82cf437b2064666be893d6d409))
* update `CODE_MODE_SYSTEM_INSTRUCTION` and `README.md` ([933d42c](https://github.com/a2anet/adk-code-mode/commit/933d42cb237b9d6deeb6611511cba28a0d5e9261))


### Documentation

* streamline `README.md` and fix broken code fences ([c237514](https://github.com/a2anet/adk-code-mode/commit/c2375149df48e21318967090b6c85d0b1f411ea2))

## [0.3.0](https://github.com/a2anet/adk-code-mode/compare/adk-code-mode-v0.2.2...adk-code-mode-v0.3.0) (2026-07-06)


### Features

* hold one sandbox container open per turn (protocol v2) ([#6](https://github.com/a2anet/adk-code-mode/issues/6)) ([aebe17d](https://github.com/a2anet/adk-code-mode/commit/aebe17d2bbbf21cc0f05852efaf910ff6adc3154))


### Documentation

* lower startup probe period in `README.md` for faster cold starts ([e251af2](https://github.com/a2anet/adk-code-mode/commit/e251af2cf7307c9944361e1a0e37c7191822a9e1))
* note HTTP `/health` startup probe in `README.md` Cloud Run section ([d58fb10](https://github.com/a2anet/adk-code-mode/commit/d58fb10f7aeda95e5c8cf29263a13e1333f661dd))

## [0.2.2](https://github.com/a2anet/adk-code-mode/compare/adk-code-mode-v0.2.1...adk-code-mode-v0.2.2) (2026-05-08)


### Bug Fixes

* use `python` fences instead of `tool_code` in `CodeModeCodeExecutor` ([efcc11e](https://github.com/a2anet/adk-code-mode/commit/efcc11e1020c7544223074e00a8649f9c85d4ad6))

## [0.2.1](https://github.com/a2anet/adk-code-mode/compare/adk-code-mode-v0.2.0...adk-code-mode-v0.2.1) (2026-05-05)


### Bug Fixes

* rename `<tools>` tag to `<code-mode>` and update `CODE_MODE_SYSTEM_INSTRUCTION` ([0a06bfb](https://github.com/a2anet/adk-code-mode/commit/0a06bfbdf22db6e0226e477d0ed074a6cc4d1515))


### Documentation

* update `README.md` with deployment instructions ([1dd7a95](https://github.com/a2anet/adk-code-mode/commit/1dd7a95197fd56231ba97182526a825baf18508a))

## [0.2.0](https://github.com/a2anet/adk-code-mode/compare/adk-code-mode-v0.1.0...adk-code-mode-v0.2.0) (2026-05-03)


### Features

* add `RemoteBackend`, sandbox hardening, and make `README.md` more concise ([a1a70ee](https://github.com/a2anet/adk-code-mode/commit/a1a70eec0686f164ce3796ba1d82bb82b7fbda59))


### Bug Fixes

* improve `CODE_MODE_SYSTEM_INSTRUCTION` ([0d00863](https://github.com/a2anet/adk-code-mode/commit/0d008636e2e273b438192f737e20e207c165a610))

## 0.1.0 (2026-04-29)


### Features

* ADK Code Mode: `CodeModeExecutor`, `DockerRuntime`, and more ([6ab1ac2](https://github.com/a2anet/adk-code-mode/commit/6ab1ac28236d21ecb740d8e920e08e4d9bd969c5))


### Documentation

* shorten ADK code executor comparison table headers ([28116cf](https://github.com/a2anet/adk-code-mode/commit/28116cf9f40b3623c31782a55885fa5c4f8c6bbf))
