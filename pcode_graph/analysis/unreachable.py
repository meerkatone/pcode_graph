from .pass_base import PassBase
from .program import BlockIndex


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
