import shutil
import subprocess
from pathlib import Path

import lief
import pytest

from pcode_graph.lief_importer import FunctionCode, iter_functions, parse_binary

SOURCE = """
int leaf(int value) { return value * 3 + 7; }
int main(int argc, char **argv) { return leaf(argc); }
"""


def compile_source(directory: Path, output_name: str, *options: str) -> Path:
    source_path = directory / "source.c"
    source_path.write_text(SOURCE)
    output_path = directory / output_name
    subprocess.run(["gcc", "-O0", *options, "-o", output_path, source_path], check=True)
    return output_path


def get_leaf(binary_path: Path) -> FunctionCode:
    return next(f for f in iter_functions(binary_path) if f.name == "leaf")


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")
@pytest.mark.parametrize("link_options", [["-pie"], ["-no-pie"]])
def test_iter_functions_executable(tmp_path: Path, link_options: list[str]):
    object_leaf = get_leaf(compile_source(tmp_path, "source.o", "-c"))
    executable_path = compile_source(tmp_path, "program", *link_options)
    executable_leaf = get_leaf(executable_path)

    symbol = parse_binary(executable_path).get_symbol("leaf")
    assert isinstance(symbol, lief.ELF.Symbol)
    assert executable_leaf.address == symbol.value
    # Leaf function code has no relocation: identical in object and executable
    assert len(executable_leaf.content) == symbol.size
    assert executable_leaf.content == object_leaf.content
