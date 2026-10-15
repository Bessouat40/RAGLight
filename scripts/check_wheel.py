"""Install the freshly built wheel in isolation and verify the public package."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    uv = shutil.which("uv")
    if uv is None:
        print("uv is required to validate the package.", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="raglight-package-check-") as directory:
        temporary_root = Path(directory)
        artifacts = temporary_root / "dist"
        subprocess.run(
            [uv, "build", "--out-dir", str(artifacts)],
            cwd=PROJECT_ROOT,
            check=True,
        )
        wheels = sorted(artifacts.glob("raglight-*.whl"))
        if not wheels:
            print("The package build produced no wheel.", file=sys.stderr)
            return 1

        environment = temporary_root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", str(wheels[-1])],
            check=True,
        )
        subprocess.run(
            [str(python), "-c", "import raglight; print(raglight.__name__)"],
            check=True,
        )

    print(f"Validated wheel: {wheels[-1].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
