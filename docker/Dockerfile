FROM python:3.13-slim

LABEL org.opencontainers.image.source="https://github.com/a2anet/adk-code-mode"
LABEL org.opencontainers.image.description="Sandbox base image for ADK Code Mode"
LABEL org.opencontainers.image.licenses="Apache-2.0"

COPY sandbox-wheel/dist/adk_code_mode_sandbox-*.whl /tmp/
RUN pip install --no-cache-dir /tmp/adk_code_mode_sandbox-*.whl \
    && rm /tmp/adk_code_mode_sandbox-*.whl

WORKDIR /workspace
ENTRYPOINT ["python", "-m", "adk_code_mode_sandbox"]
