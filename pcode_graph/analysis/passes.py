from .load_store import LoadStoreInserter
from .cfg import CFGBuilder
from .unreachable import UnreachableCodeFinder
from .def_use import ReachingDefBuilder
from .alias_analysis import MemoryAnalyzer

from .program import Program


def run_analyses(program: Program):

    LoadStoreInserter(program)()
    CFGBuilder(program)()
    UnreachableCodeFinder(program)()
    ReachingDefBuilder(program)()
    MemoryAnalyzer(program)()
