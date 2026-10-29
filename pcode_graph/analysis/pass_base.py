from pcode_graph.analysis.program import Program
from pcode_graph.visitor import PcodeVisitor


class PassBase(PcodeVisitor):

    def __init__(self, program: Program) -> None:
        super().__init__()
        self.program = program
