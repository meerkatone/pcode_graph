from typing import override

from pypcode import OpCode, PcodeOp

from pcode_graph.analysis.pass_base import PassBase
from pcode_graph.bit_utils import unsigned_to_signed
from pcode_graph.pcode import from_native_opcode

from .program import (
    BasicBlock,
    EntryIndex,
    ExitIndex,
    OpIndex,
    Program,
    UnknownCodeIndex,
    Address,
)


class CFGBuilder(PassBase):

    def __init__(self, program: Program) -> None:
        super().__init__(program)

        # First P-Code operation for each instruction address
        self.addresses: dict[Address, OpIndex] = {}

    def __call__(self):
        # First collects instruction addresses encoded in IMARK operations
        for i, op in enumerate(self.program.operations):
            if op.opcode == OpCode.IMARK:
                self.addresses[op.inputs[0].offset] = i

        # Successors of jumping operations
        self._jumps: dict[OpIndex, list[OpIndex]] = {}
        for index, op in enumerate(self.program.operations):
            self.visit(from_native_opcode(op.opcode), index, op)

        op_count = len(self.program.operations)
        leaders = {0} | {index + 1 for index in self._jumps}
        for targets in self._jumps.values():
            leaders.update(target for target in targets if target >= 0)
        sorted_leaders = sorted(leaders - {op_count}) if op_count else []

        self.program.basic_blocks = []
        self.program.block_indexes = []
        for start, end in zip(sorted_leaders, sorted_leaders[1:] + [op_count]):
            self.program.block_indexes.extend(
                [len(self.program.basic_blocks)] * (end - start)
            )
            self.program.basic_blocks.append(BasicBlock(start, end))

        for block_index, block in enumerate(self.program.basic_blocks):
            last_index = block.last_op_index
            for successor in self._jumps.get(
                last_index, [self._next_index(last_index)]
            ):
                if successor >= 0:
                    successor = self.program.block_indexes[successor]
                    self.program.basic_blocks[successor].pred_blocks.add(block_index)
                block.succ_blocks.append(successor)

        if self.program.basic_blocks:
            self.program.basic_blocks[0].pred_blocks.add(EntryIndex)

    def _next_index(self, index: OpIndex) -> OpIndex:
        return index + 1 if index + 1 < len(self.program.operations) else ExitIndex

    def _get_jump_target(self, index: OpIndex, op: PcodeOp) -> OpIndex:
        address = op.inputs[0]
        match address.space.name:
            case "ram":
                return self.addresses.get(address.offset, UnknownCodeIndex)
            case "const":
                # Relative jump inside the current instruction
                target = index + unsigned_to_signed(address.offset, address.size)
                assert 0 <= target < len(self.program.operations)
                return target
            case "register" | "unique":
                return UnknownCodeIndex
            case _ as name:
                raise ValueError(f"Unexpected address space for jump address: {name}")

    @override
    def visit_default(self, index: OpIndex, op: PcodeOp) -> None:
        pass

    @override
    def visit_branch(self, index: OpIndex, op: PcodeOp) -> None:
        self._jumps[index] = [self._get_jump_target(index, op)]

    @override
    def visit_branchind(self, index: OpIndex, op: PcodeOp) -> None:
        self._jumps[index] = [self._get_jump_target(index, op)]

    @override
    def visit_return(self, index: OpIndex, op: PcodeOp) -> None:
        self._jumps[index] = [self._get_jump_target(index, op)]

    @override
    def visit_call(self, index: OpIndex, op: PcodeOp) -> None:
        target = self._get_jump_target(index, op)
        if target == UnknownCodeIndex:
            # Calling an unseen function: we suppose it returns here after
            self._jumps[index] = [target, self._next_index(index)]
        else:
            self._jumps[index] = [target]

    @override
    def visit_callind(self, index: OpIndex, op: PcodeOp) -> None:
        self.visit_call(index, op)

    @override
    def visit_cbranch(self, index: OpIndex, op: PcodeOp) -> None:
        self._jumps[index] = [self._get_jump_target(index, op), self._next_index(index)]

    @override
    def visit_callother(self, index: OpIndex, op: PcodeOp) -> None:
        self._jumps[index] = [UnknownCodeIndex, self._next_index(index)]
