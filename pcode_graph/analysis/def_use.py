import heapq

from .pass_base import PassBase
from .program import (
    BasicBlock,
    BlockIndex,
    Definitions,
    EntryIndex,
    Var,
    VarKinds,
    wrap,
    split,
    apply_operation,
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
                if var.kind == VarKinds.Register:
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
