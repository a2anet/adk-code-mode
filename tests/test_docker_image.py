from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_default_dockerfile_installs_only_sandbox_wheel() -> None:
    dockerfile = (_REPO_ROOT / "docker" / "Dockerfile").read_text()

    assert "FROM python:3.13-slim" in dockerfile
    assert "sandbox-wheel/dist/adk_code_mode_sandbox-*.whl" in dockerfile
    assert "adk_code_mode_sandbox" in dockerfile
    assert "adk-code-mode " not in dockerfile


def test_makefile_builds_default_docker_image_tag() -> None:
    makefile = (_REPO_ROOT / "Makefile").read_text()

    assert "docker-image:" in makefile
    assert "uv build --wheel --out-dir sandbox-wheel/dist sandbox-wheel" in makefile
    assert "docker build -f docker/Dockerfile -t adk-code-mode:local ." in makefile
