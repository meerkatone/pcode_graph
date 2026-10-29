"""Helpers to deal with pypcode native operations.
"""

from dataclasses import dataclass
from typing import Any

import pypcode

# Operations whose first input is not a data value (code address, space or user operation id)
NON_DATA_FIRST_INPUT_OPCODES = frozenset(
    {
        pypcode.OpCode.CBRANCH,
        pypcode.OpCode.BRANCH,
        pypcode.OpCode.CALL,
        pypcode.OpCode.LOAD,
        pypcode.OpCode.STORE,
        pypcode.OpCode.CALLOTHER,
    }
)

JUMP_OPCODES = {
    pypcode.OpCode.BRANCH,
    pypcode.OpCode.BRANCHIND,
    pypcode.OpCode.CALL,
    pypcode.OpCode.CALLIND,
    pypcode.OpCode.CBRANCH,
    pypcode.OpCode.RETURN,
    pypcode.OpCode.CALLOTHER,
}

RELATIVE_JUMP_OPCODES = frozenset({pypcode.OpCode.BRANCH, pypcode.OpCode.CBRANCH})


def is_data_input(opcode: pypcode.OpCode, position: int) -> bool:
    """Tells whether the given input of an operation is a value read by this operation."""

    if opcode == pypcode.OpCode.IMARK:
        return False
    return position != 0 or opcode not in NON_DATA_FIRST_INPUT_OPCODES


@dataclass(frozen=True, slots=True)
class SyntheticSpace:
    name: str


@dataclass(frozen=True, slots=True)
class SyntheticVarnode:
    """Mimics the parts of pypcode.Varnode used by the analyses."""

    space: SyntheticSpace
    offset: int
    size: int

    def getRegisterName(self) -> str:
        return ""


@dataclass(slots=True)
class SyntheticOperation:
    """Mimics the parts of pypcode.PcodeOp used by the analyses."""

    opcode: pypcode.OpCode
    inputs: list[Any]
    output: Any


CONST_SPACE = SyntheticSpace("const")
UNIQUE_SPACE = SyntheticSpace("unique")


def accesses_ram(operation: pypcode.PcodeOp) -> bool:
    if operation.output is not None and operation.output.space.name == "ram":
        return True
    return any(
        varnode.space.name == "ram" and is_data_input(operation.opcode, position)
        for position, varnode in enumerate(operation.inputs)
    )
