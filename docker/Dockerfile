FROM python:3.13-slim

COPY sandbox-wheel/dist/adk_code_mode_sandbox-*.whl /tmp/
RUN pip install --no-cache-dir /tmp/adk_code_mode_sandbox-*.whl \
    && rm /tmp/adk_code_mode_sandbox-*.whl

WORKDIR /workspace
ENTRYPOINT ["python", "-m", "adk_code_mode_sandbox"]
