from dataclasses import dataclass, field
from enum import Enum
from functools import cache
import heapq
from typing import Iterator, override
from pypcode import OpCode, PcodeOp, Varnode
from pcode_graph.arch import Arch
from pcode_graph.pcode import dump_operation, from_native_opcode
from pcode_graph.bit_utils import unsigned_to_signed
from pcode_graph.visitor import PcodeVisitor

OpIndex = int
EntryIndex = -1
UnknownCodeIndex = -2
ExitIndex = -3

Address = int

BlockIndex = int


def dump_index(i: OpIndex) -> str:
    match i:
        case -1:
            return "entry"
        case -2:
            return "elsewhere"
        case -3:
            return "exit"
        case _:
            assert i >= 0
            return str(i)


class VarKinds(Enum):
    # Register or named P-Code temporary
    Register = 0
    # P-Code temporary
    Unique = 1
    # Numerical constant
    Constant = 2
    # PC-relative address value
    Ram = 3
    # Dummy access to the memory
    IndirectAccess = 4


@dataclass(frozen=True, order=True)
class Var:
    """Hashable and comparable wrapper for temporaries, registers, constants and memory."""

    kind: VarKinds
    size: int
    value: int
    pretty_name: str = field(compare=False)

    def __str__(self) -> str:
        return self.pretty_name


def wrap(var: Varnode) -> Var:
    match var.space.name:
        case "register":
            name = var.getRegisterName() or "UNKNOWN"
            return Var(VarKinds.Register, var.size, var.offset, name)
        case "unique":
            return Var(VarKinds.Unique, var.size, var.offset, f"${var.offset}")
        case "const":
            return Var(VarKinds.Constant, var.size, var.offset, f"${var.offset}")
        case "ram":
            return Var(VarKinds.Ram, var.size, var.offset, f"[0x{var.offset:x}]")
        case _ as name:
            raise ValueError(f"Unexpected space for input variable: {name}")


# Special variable singleton
MEMORY = Var(VarKinds.IndirectAccess, 0, 0, "MEMORY")

# Writing operations by variable byte (see split), or for MEMORY
Definitions = dict[Var, set[OpIndex]]


def split(var: Var) -> tuple[Var, ...]:
    """Splits a register into its bytes to track definitions.
    Overlapping registers share bytes (RAX, AX, AH) so that a partial write only kills the definitions of the bytes it writes.
    """

    if var.kind in {VarKinds.IndirectAccess, VarKinds.Constant}:
        return (var,)
    return _split_bytes(var.kind, var.value, var.size)


@cache
def _split_bytes(kind: VarKinds, value: int, size: int) -> tuple[Var, ...]:
    return tuple(
        Var(kind, 1, value + offset, f"{kind.name}:{value + offset:#x}")
        for offset in range(size)
    )


@dataclass(slots=True)
class BasicBlock:
    """Maximal sequence of operations with a single entry and a single exit."""

    # Operation indexes range: start included, end excluded
    start: OpIndex
    end: OpIndex

    # Block-level CFG edges, special indexes (EntryIndex, ExitIndex...) are kept as is.
    # Order of successors matters, as for operations.
    pred_blocks: set[BlockIndex] = field(default_factory=set)
    succ_blocks: list[BlockIndex] = field(default_factory=list)

    reachable: bool = False

    @property
    def last_op_index(self) -> OpIndex:
        return self.end - 1

    @property
    def op_indexes(self) -> range:
        return range(self.start, self.end)


def apply_operation(
    definitions: Definitions, index: OpIndex, op: PcodeOp
) -> Definitions:
    """Returns the new definitions after the given operation is applied."""

    if op.output:
        result = dict(definitions)
        written = {index}
        for byte in split(wrap(op.output)):
            result[byte] = written
        return result
    if op.opcode == OpCode.STORE:
        # As we do not analyze which part of memory is written,
        # a write to memory won't kill previous definitions of MEMORY.
        result = dict(definitions)
        result[MEMORY] = definitions.get(MEMORY, set()) | {index}
        return result
    return definitions


@dataclass
class RichPcodeList:
    """A class to run static analysis on a list of P-Code operations."""

    # Source architecture
    arch: Arch

    # List of P-Code operations
    operations: list[PcodeOp]

    # First P-Code operation for each instruction address
    addresses: dict[Address, OpIndex] = field(default_factory=dict)

    # Basic blocks sorted by operation index, and block of each operation
    basic_blocks: list[BasicBlock] = field(default_factory=list)
    block_indexes: list[BlockIndex] = field(default_factory=list)

    # Reaching definitions. For each operation, by variable, list of:
    #   - definitions (= writing operation index) that reach this operation,
    #   - or EntryIndex (for initial definitions of inputs).
    # Sets are shared between operations: do not modify them.
    reaching_defs: list[Definitions] = field(default_factory=list)

    def __post_init__(self):
        InstructionAddressCollector(self)()
        CFGBuilder(self)()
        UnreachableCodeFinder(self)()
        ReachingDefBuilder(self)()

    def __str__(self) -> str:
        return self.dump()

    def block_of(self, index: OpIndex) -> BasicBlock:
        return self.basic_blocks[self.block_indexes[index]]

    def reachable(self, index: OpIndex) -> bool:
        return self.block_of(index).reachable

    def pred_ops(self, index: OpIndex) -> set[OpIndex]:
        """Operation-level CFG predecessors, derived from the block-level CFG."""

        block = self.block_of(index)
        if index != block.start:
            return {index - 1}
        return {
            self.basic_blocks[pred].last_op_index if pred >= 0 else pred
            for pred in block.pred_blocks
        }

    def succ_ops(self, index: OpIndex) -> list[OpIndex]:
        """Operation-level CFG successors, derived from the block-level CFG."""

        block = self.block_of(index)
        if index != block.last_op_index:
            return [index + 1]
        return [
            self.basic_blocks[successor].start if successor >= 0 else successor
            for successor in block.succ_blocks
        ]

    def dump(self) -> str:
        """Dumps the program and analysis results in a markdown table."""

        s = "|index|op|preds|succs|input defs|reachable|exit def|\n"
        exit_defs = self.get_exit_defs()

        s += "|--|--|--|--|--|--|--|\n"
        for i, op in enumerate(self.operations):
            s += f"|{i}|{dump_operation(op)}|"
            s += ", ".join(dump_index(p) for p in self.pred_ops(i)) + "|"
            s += ", ".join(dump_index(p) for p in self.succ_ops(i)) + "|"
            var_defs = []
            for var, defs in self.iter_input_defs(i):
                if defs:
                    var_defs.append(
                        str(var) + " from " + " or ".join(dump_index(d) for d in defs)
                    )

            s += ", ".join(var_defs) + "|"
            if self.reachable(i):
                s += "x"
            s += "|"
            exit_def_here = []
            for var, defs in exit_defs.items():
                for j in defs:
                    if i == j:
                        exit_def_here.append(str(var))
            if exit_def_here:
                s += ", ".join(exit_def_here)
            s += "|\n"

        return s

    def get_defs(self, index: OpIndex, var: Var) -> set[OpIndex]:
        """Returns all the operations defining the value at given operation of the given variable."""

        definitions = self.reaching_defs[index]
        defs: set[OpIndex] = set()
        for byte in split(var):
            defs |= definitions.get(byte, set())
        return defs

    def get_exit_defs(self) -> Definitions:
        """Computes side effects of the program."""

        exit_defs: Definitions = {}
        for block in self.basic_blocks:
            if not block.reachable or not {UnknownCodeIndex, ExitIndex}.intersection(
                block.succ_blocks
            ):
                continue
            index = block.last_op_index
            outputs = apply_operation(
                self.reaching_defs[index], index, self.operations[index]
            )
            for byte, defs in outputs.items():
                if byte.kind == VarKinds.Unique:
                    # Ignore P-Code temporary
                    continue
                for writer in defs - {EntryIndex}:
                    # Group bytes back by written variable
                    output = self.operations[writer].output
                    var = wrap(output) if output else MEMORY
                    exit_defs.setdefault(var, set()).add(writer)
        return exit_defs

    def iter_inputs(self, index: OpIndex) -> Iterator[Var]:
        """Iterates operation inputs, skipping **special** ones
        that have no semantic interest to put in the graph (see P-Code documentation).
        """

        op = self.operations[index]
        if op.opcode == OpCode.LOAD:
            yield MEMORY

        for i, var in enumerate(op.inputs):
            if i != 0 or op.opcode not in {
                OpCode.IMARK,
                OpCode.CBRANCH,
                OpCode.BRANCH,
                OpCode.CALL,
                OpCode.LOAD,
                OpCode.STORE,
                OpCode.CALLOTHER,
            }:
                yield wrap(var)

    def iter_input_defs(self, index: OpIndex) -> Iterator[tuple[Var, set[OpIndex]]]:
        """For each operand, returns the written variable and the operation(s) defining it."""

        for var in self.iter_inputs(index):
            yield var, self.get_defs(index, var)


class PassBase(PcodeVisitor):

    def __init__(self, program: RichPcodeList) -> None:
        super().__init__()
        self.program = program


class InstructionAddressCollector(PassBase):
    """Collects instruction addresses removing IMARK operations"""

    def __call__(self):
        for i, op in enumerate(self.program.operations):
            if op.opcode == OpCode.IMARK:
                self.program.addresses[op.inputs[0].offset] = i


class CFGBuilder(PassBase):

    def __call__(self):
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
                return self.program.addresses.get(address.offset, UnknownCodeIndex)
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


class UnreachableCodeFinder(PassBase):

    def __call__(self):

        pending: list[BlockIndex] = [0] if self.program.basic_blocks else []
        while pending:
            block = self.program.basic_blocks[pending.pop()]
            if block.reachable:
                continue
            block.reachable = True
            pending.extend(
                successor for successor in block.succ_blocks if successor >= 0
            )


class ReachingDefBuilder(PassBase):

    def __call__(self):
        self.program.reaching_defs = [{} for _ in self.program.operations]
        if not self.program.basic_blocks:
            return

        block_inputs: list[Definitions] = [{} for _ in self.program.basic_blocks]

        # Initialize
        for index in range(len(self.program.operations)):
            if not self.program.reachable(index):
                continue
            for var in self.program.iter_inputs(index):
                if var.kind in {
                    VarKinds.IndirectAccess,
                    VarKinds.Ram,
                    VarKinds.Register,
                }:
                    for byte in split(var):
                        block_inputs[0][byte] = {EntryIndex}

        # Block effects are computed once, on demand, as (generated defs, killed variables)
        effects: dict[BlockIndex, tuple[Definitions, set[Var]]] = {}

        # Each reachable block is processed at least once, lowest index first:
        # follows the code layout, close to a reverse post-order.
        # The list is sorted, so it is already a valid heap.
        pending = [
            index
            for index, block in enumerate(self.program.basic_blocks)
            if block.reachable
        ]
        queued = set(pending)

        while pending:
            block_index = heapq.heappop(pending)
            queued.remove(block_index)

            if block_index not in effects:
                effects[block_index] = self._compute_effect(
                    self.program.basic_blocks[block_index]
                )
            generated, killed = effects[block_index]
            outputs = self._apply_effect(block_inputs[block_index], generated, killed)

            for successor in self.program.basic_blocks[block_index].succ_blocks:
                if successor < 0:
                    continue
                changed = self._merge_definitions(block_inputs[successor], outputs)
                if changed and successor not in queued:
                    heapq.heappush(pending, successor)
                    queued.add(successor)

        for block, inputs in zip(self.program.basic_blocks, block_inputs):
            if block.reachable:
                self._expand_block(block, inputs)

    def _compute_effect(self, block: BasicBlock) -> tuple[Definitions, set[Var]]:
        generated: Definitions = {}
        killed: set[Var] = set()
        for index in block.op_indexes:
            op = self.program.operations[index]
            if op.output:
                written = {index}
                for byte in split(wrap(op.output)):
                    generated[byte] = written
                    killed.add(byte)
            elif op.opcode == OpCode.STORE:
                # As we do not analyze which part of memory is written,
                # a write to memory won't kill previous definitions of MEMORY.
                generated.setdefault(MEMORY, set()).add(index)
        return generated, killed

    @staticmethod
    def _apply_effect(
        inputs: Definitions, generated: Definitions, killed: set[Var]
    ) -> Definitions:
        # Sets are never modified in place, so they can be shared
        outputs = dict(generated)
        for var, defs in inputs.items():
            if var in killed:
                continue
            outputs[var] = outputs[var] | defs if var in outputs else defs
        return outputs

    def _expand_block(self, block: BasicBlock, inputs: Definitions):
        """Computes the definitions reaching each operation of the block."""

        current = inputs
        for index in block.op_indexes:
            self.program.reaching_defs[index] = current
            current = apply_operation(current, index, self.program.operations[index])

    @staticmethod
    def _merge_definitions(target: Definitions, source: Definitions) -> bool:
        """Adds source definitions to target ones, returns True if target changed."""

        changed = False
        for var, defs in source.items():
            target_defs = target.get(var)
            if target_defs is None:
                target[var] = set(defs)
                changed = True
            elif not defs <= target_defs:
                target_defs |= defs
                changed = True
        return changed
