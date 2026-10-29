from pypcode import OpCode

from pcode_graph.analysis.alias_analysis import MemoryAnalyzer
from pcode_graph.analysis.load_store import LoadStoreInserter
from pcode_graph.analysis.passes import run_analyses
from pcode_graph.analysis.program import Program
from pcode_graph.arch import Arch
from pcode_graph.analysis.native_pcode import (
    CONST_SPACE,
    SyntheticOperation,
    SyntheticSpace,
    SyntheticVarnode,
)
from pcode_graph.pcode import dump_operation

RAM = SyntheticSpace("ram")
REGISTER = SyntheticSpace("register")


def constant(value: int, size: int = 8) -> SyntheticVarnode:
    return SyntheticVarnode(CONST_SPACE, value, size)


def opcodes(operations: list) -> list[str]:
    return [operation.opcode.name for operation in operations]


def test_read_write_in_one_instruction():
    variable = SyntheticVarnode(RAM, 0x1000, 8)
    flag = SyntheticVarnode(REGISTER, 0x200, 1)
    operations = [
        SyntheticOperation(OpCode.IMARK, [SyntheticVarnode(RAM, 0, 4)], None),
        SyntheticOperation(OpCode.INT_CARRY, [variable, constant(1)], flag),
        SyntheticOperation(OpCode.INT_ADD, [variable, constant(1)], variable),
        SyntheticOperation(OpCode.INT_EQUAL, [variable, constant(0)], flag),
    ]

    result = make_memory_accesses_explicit(operations, Arch.x86_64)  # type: ignore

    assert opcodes(result) == [
        "IMARK",
        "LOAD",
        "INT_CARRY",
        "INT_ADD",
        "STORE",
        "INT_EQUAL",
    ]
    loaded = result[1].output
    written = result[3].output
    assert result[2].inputs[0] == result[3].inputs[0] == loaded
    assert result[4].inputs[1:] == [constant(0x1000), written]
    # The written value is reused, without reloading it
    assert result[5].inputs[0] == written
    assert loaded.space.name == written.space.name == "unique"
    assert loaded.offset != written.offset
    for operation in result:
        dump_operation(operation)  # type: ignore


def test_relative_jumps_are_shifted():
    variable = SyntheticVarnode(RAM, 0x1000, 4)
    register = SyntheticVarnode(REGISTER, 0, 4)
    condition = SyntheticVarnode(REGISTER, 0x200, 1)
    operations = [
        SyntheticOperation(OpCode.IMARK, [SyntheticVarnode(RAM, 0, 4)], None),
        # Jumps to the INT_ADD, over the operation reading memory
        SyntheticOperation(OpCode.CBRANCH, [constant(2, 4), condition], None),
        SyntheticOperation(OpCode.COPY, [variable], register),
        SyntheticOperation(OpCode.INT_ADD, [variable, register], register),
        # Jumps back to the COPY
        SyntheticOperation(OpCode.BRANCH, [constant(0xFFFFFFFE, 4)], None),
    ]

    program = Program(Arch.x86_32, operations)  # type: ignore
    run_analyses(program)

    result = program.operations

    assert opcodes(result) == [
        "IMARK",
        "CBRANCH",
        "LOAD",
        "COPY",
        "LOAD",
        "INT_ADD",
        "BRANCH",
    ]
    # Jump targets are the first operation replacing the original target
    assert result[1].inputs[0] == constant(3, 4)
    assert result[6].inputs[0] == constant(0xFFFFFFFC, 4)
    # Values are not reused across jump targets
    assert result[4].output != result[2].output
    assert result[2].inputs[1] == result[4].inputs[1] == constant(0x1000, 4)


def test_code_addresses_are_kept():
    target = SyntheticVarnode(RAM, 0x1000, 8)
    operations = [
        SyntheticOperation(OpCode.IMARK, [SyntheticVarnode(RAM, 0, 4)], None),
        SyntheticOperation(OpCode.BRANCH, [target], None),
    ]

    assert make_memory_accesses_explicit(operations, Arch.x86_64) is operations  # type: ignore
