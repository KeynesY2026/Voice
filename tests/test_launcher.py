import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows batch launcher")
LAUNCHER = Path(__file__).resolve().parents[1] / "start-auto.bat"


def run_launcher(tmp_path, *, environments=(), uv_exit=0, provide_uv=True, cuda_in_default=False):
    project = tmp_path / "project with spaces"
    project.mkdir()
    assert LAUNCHER.is_file(), "The Windows auto-mode launcher is missing"
    shutil.copyfile(LAUNCHER, project / LAUNCHER.name)
    for environment in environments:
        scripts = project / environment / "Scripts"
        scripts.mkdir(parents=True)
        (scripts / "python.exe").touch()
    if cuda_in_default:
        (project / ".venv" / "Lib" / "site-packages" / "torch-2.8.0+cu128.dist-info").mkdir(
            parents=True,
        )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if provide_uv:
        (bin_dir / "uv.cmd").write_text(
            "@echo off\n"
            "echo cwd=%CD%>\"%LAUNCH_TEST_OUTPUT%\"\n"
            "echo environment=%UV_PROJECT_ENVIRONMENT%>>\"%LAUNCH_TEST_OUTPUT%\"\n"
            "echo arguments=%*>>\"%LAUNCH_TEST_OUTPUT%\"\n"
            "echo hf_hub_offline=%HF_HUB_OFFLINE% >> \"%LAUNCH_TEST_OUTPUT%\"\n"
            "echo use_modelscope=%USE_MODELSCOPE%>>\"%LAUNCH_TEST_OUTPUT%\"\n"
            f"exit /b {uv_exit}\n",
            encoding="ascii",
        )
    output = tmp_path / "invocation.txt"
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join((str(bin_dir), str(Path(env["SYSTEMROOT"]) / "System32")))
    env["LAUNCH_TEST_OUTPUT"] = str(output)
    result = subprocess.run(
        [env["COMSPEC"], "/d", "/c", str(project / LAUNCHER.name)],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    invocation = [line.rstrip() for line in output.read_text(encoding="ascii").splitlines()] if output.exists() else []
    return project, result, invocation


@pytest.mark.parametrize(
    ("environments", "expected_environment", "expected_extra"),
    [
        ((".venv-gpu", ".venv"), ".venv-gpu", "gpu"),
        ((".venv",), ".venv", "cpu"),
    ],
)
def test_launcher_starts_auto_with_installed_environment(
    tmp_path, environments, expected_environment, expected_extra,
):
    project, result, invocation = run_launcher(tmp_path, environments=environments)

    assert result.returncode == 0, result.stdout + result.stderr
    assert invocation == [
        f"cwd={project}",
        f"environment={expected_environment}",
        f"arguments=run --no-sync --no-dev --extra {expected_extra} voice-reader-web --device auto",
        "hf_hub_offline=1",
        "use_modelscope=false",
    ]


def test_launcher_reports_missing_environment(tmp_path):
    _, result, invocation = run_launcher(tmp_path)

    assert result.returncode != 0
    assert "environment" in result.stdout.lower()
    assert not invocation


def test_launcher_preserves_gpu_installation_in_default_environment(tmp_path):
    project, result, invocation = run_launcher(
        tmp_path, environments=(".venv",), cuda_in_default=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert invocation == [
        f"cwd={project}",
        "environment=.venv",
        "arguments=run --no-sync --no-dev --extra gpu voice-reader-web --device auto",
        "hf_hub_offline=1",
        "use_modelscope=false",
    ]


def test_launcher_reports_missing_uv(tmp_path):
    _, result, invocation = run_launcher(
        tmp_path, environments=(".venv-gpu",), provide_uv=False,
    )

    assert result.returncode != 0
    assert "uv" in result.stdout.lower()
    assert not invocation


def test_launcher_reports_and_preserves_startup_failure(tmp_path):
    _, result, invocation = run_launcher(
        tmp_path, environments=(".venv-gpu",), uv_exit=42,
    )

    assert result.returncode == 42
    assert "failed" in result.stdout.lower()
    assert invocation
