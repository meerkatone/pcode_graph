"""Rewrites direct accesses to "ram" varnodes into explicit LOAD and STORE operations.

P-Code may read or write a static memory address directly, as for PC-relative accesses.
Once rewritten, memory is only accessed through LOAD and STORE.
"""

from dataclasses import dataclass
from typing import Any, cast

from pypcode import OpCode, PcodeOp

from .native_pcode import (
    CONST_SPACE,
    JUMP_OPCODES,
    RELATIVE_JUMP_OPCODES,
    UNIQUE_SPACE,
    SyntheticOperation,
    SyntheticVarnode,
    is_data_input,
)
from .program import Program

from .pass_base import PassBase
from pcode_graph.bit_utils import signed_to_unsigned, unsigned_to_signed


def _relative_jump_target(index: int, operation: PcodeOp) -> int | None:
    if operation.opcode not in RELATIVE_JUMP_OPCODES:
        return None
    address = operation.inputs[0]
    if address.space.name != "const":
        return None
    return index + unsigned_to_signed(address.offset, address.size)


class LoadStoreInserter(PassBase):

    def __init__(self, program: Program) -> None:

        super().__init__(program)

        self.pointer_size = self.program.arch.pointer_size
        self.next_unique_offset = max(
            (
                varnode.offset + varnode.size
                for operation in self.operations
                for varnode in [operation.output, *operation.inputs]
                if varnode is not None and varnode.space.name == "unique"
            ),
            default=0,
        )
        # Temporaries holding the value of static addresses, by (address, size)
        self.cached_values: dict[tuple[int, int], SyntheticVarnode] = {}

    @property
    def operations(self):
        return self.program.operations

    def __call__(self):
        jump_targets = {
            index: target
            for index, operation in enumerate(self.operations)
            if (target := _relative_jump_target(index, operation)) is not None
        }
        reached_by_jump = set(jump_targets.values())

        expansions: list[list[Any]] = []
        for index, operation in enumerate(self.operations):
            # Cached values are only reused inside straight-line code
            if index in reached_by_jump or operation.opcode == OpCode.IMARK:
                self.cached_values.clear()
            expansions.append(self._expand(operation))
            if operation.opcode in JUMP_OPCODES:
                self.cached_values.clear()

        starts = [0]
        for expansion in expansions:
            starts.append(starts[-1] + len(expansion))

        # Relative jumps inside an instruction are shifted by inserted operations
        for index, target in jump_targets.items():
            expansion = expansions[index]
            (position,) = [
                position
                for position, operation in enumerate(expansion)
                if operation.opcode in RELATIVE_JUMP_OPCODES
            ]
            jump = expansion[position]
            offset = starts[target] - (starts[index] + position)
            address = jump.inputs[0]
            if offset != unsigned_to_signed(address.offset, address.size):
                new_address = SyntheticVarnode(
                    CONST_SPACE, signed_to_unsigned(offset, address.size), address.size
                )
                expansion[position] = SyntheticOperation(
                    jump.opcode, [new_address, *jump.inputs[1:]], jump.output
                )

        self.program.operations = cast(
            list[PcodeOp], [op for expansion in expansions for op in expansion]
        )

    def _expand(self, operation: PcodeOp) -> list[Any]:
        """Returns the operations replacing the given one."""

        expansion: list[Any] = []
        inputs = list(operation.inputs)
        changed = False
        for position, varnode in enumerate(operation.inputs):
            if varnode.space.name != "ram" or not is_data_input(
                operation.opcode, position
            ):
                continue
            key = (varnode.offset, varnode.size)
            temporary = self.cached_values.get(key)
            if temporary is None:
                temporary = self._new_temporary(varnode.size)
                expansion.append(
                    SyntheticOperation(
                        OpCode.LOAD,
                        [self._space_id(), self._address(varnode.offset)],
                        temporary,
                    )
                )
                self.cached_values[key] = temporary
            inputs[position] = temporary  # type: ignore
            changed = True

        output = operation.output
        store = None
        if output is not None and output.space.name == "ram":
            written = range(output.offset, output.offset + output.size)
            for address, size in list(self.cached_values):
                if address < written.stop and written.start < address + size:
                    del self.cached_values[(address, size)]
            temporary = self._new_temporary(output.size)
            self.cached_values[(output.offset, output.size)] = temporary
            store = SyntheticOperation(
                OpCode.STORE,
                [self._space_id(), self._address(output.offset), temporary],
                None,
            )
            output = temporary
            changed = True

        expansion.append(
            SyntheticOperation(operation.opcode, inputs, output)
            if changed
            else operation
        )
        if store is not None:
            expansion.append(store)
        return expansion

    def _new_temporary(self, size: int) -> SyntheticVarnode:
        temporary = SyntheticVarnode(UNIQUE_SPACE, self.next_unique_offset, size)
        self.next_unique_offset += size
        return temporary

    def _space_id(self) -> SyntheticVarnode:
        return SyntheticVarnode(CONST_SPACE, 0, self.pointer_size)

    def _address(self, address: int) -> SyntheticVarnode:
        return SyntheticVarnode(CONST_SPACE, address, self.pointer_size)
