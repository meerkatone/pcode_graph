from pathlib import Path
import subprocess
import sys
from .fixtures import outdir

def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pcode_graph", *args], capture_output=True, text=True
    )


def test_assembly_file(outdir: Path):
    assembly_path = outdir / "input.asm"
    assembly_path.write_text("mov rax, qword ptr [rsp]\nret\n")

    result = run_cli("pcode", str(assembly_path), "--arch", "x86_64")

    assert result.returncode == 0, result.stderr
    assert "load" in result.stdout


def test_assembly_file_requires_arch(outdir: Path):
    assembly_path = outdir / "input.asm"
    assembly_path.write_text("ret\n")

    result = run_cli("pcode", str(assembly_path))

    assert result.returncode == 1
    assert "requires --arch" in result.stderr
