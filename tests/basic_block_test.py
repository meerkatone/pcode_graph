from pathlib import Path

import networkx as nx
from networkx.algorithms.isomorphism import categorical_multiedge_match

from pcode_graph.characterization import get_graph_diameters
from pcode_graph.graph import CDG, EdgeKinds, NodeKinds
from pcode_graph.maker import MakerFlags
from pcode_graph.nx_exporter import export_to_networkx
from pcode_graph.pcode import OpCodes
from .helpers import make_graph_from_asm
from .fixtures import arm64, x86_64, log, outdir, ArchContext

BASIC_BLOCK_FLAGS = MakerFlags(basic_blocks=True)
NO_BASIC_BLOCK_FLAGS = MakerFlags(basic_blocks=False)


def nodes_of_kind(graph: CDG, kind: NodeKinds) -> list[int]:
    return [index for index, node in enumerate(graph.nodes) if node.kind == kind]


def nodes_with_opcode(graph: CDG, opcode: OpCodes) -> list[int]:
    return [index for index, node in enumerate(graph.nodes) if node.opcode == opcode]


def edges_of_kind(graph: CDG, kind: EdgeKinds) -> set[tuple[int, int]]:
    return {
        (edge.source_node, edge.destination_node)
        for edge in graph.edges
        if edge.kind == kind
    }


def ordering_edges_of(graph: CDG) -> list[tuple[int, int]]:
    """Intra-block control edges, between side-effect operations."""
    return [
        (edge.source_node, edge.destination_node)
        for edge in graph.edges
        if edge.kind == EdgeKinds.Control
        and edge.operand_number is None
        and graph.nodes[edge.source_node].kind == NodeKinds.Operation
    ]


def are_isomorphic(first: CDG, second: CDG) -> bool:
    return nx.is_isomorphic(
        export_to_networkx(first, False),
        export_to_networkx(second, False),
        node_match=lambda first_node, second_node: first_node == second_node,
        edge_match=categorical_multiedge_match(
            ["kind", "operand_number"], [None, None]
        ),
    )


def test_single_block(outdir: Path, arm64: ArchContext, log):
    graph = make_graph_from_asm(
        """
add x0, x1, x2
mul x1, x0, x0
""",
        arm64,
        outdir,
        flags=BASIC_BLOCK_FLAGS,
    )

    (block,) = nodes_of_kind(graph, NodeKinds.BasicBlock)
    (begin,) = nodes_of_kind(graph, NodeKinds.Begin)
    (end,) = nodes_of_kind(graph, NodeKinds.End)
    operations = nodes_of_kind(graph, NodeKinds.Operation)

    assert edges_of_kind(graph, EdgeKinds.BasicBlockMembership) == {
        (operation, block) for operation in operations
    }
    # No side effect: only entry and exit control edges remain
    assert edges_of_kind(graph, EdgeKinds.Control) == {(begin, block), (block, end)}


def test_simple_basic_blocks(outdir: Path, x86_64: ArchContext):

    g = make_graph_from_asm(
        """
            add rax, rbx
            mul rcx
            jne label
            add rcx, rax
            mul rcx
label:      add rbx, rax
""",
        x86_64,
        outdir,
    )

    # TODO


def test_memory_ordering(outdir: Path, arm64: ArchContext, log):
    graph = make_graph_from_asm(
        """
ldr x0, [x9]
ldr x1, [x10]
str x2, [x8]
ldr x3, [x11]
""",
        arm64,
        outdir,
        flags=BASIC_BLOCK_FLAGS,
    )

    loads = set(nodes_with_opcode(graph, OpCodes.LOAD))
    (store,) = nodes_with_opcode(graph, OpCodes.STORE)
    ordering_edges = ordering_edges_of(graph)

    # Loads are not ordered between them, only with respect to the store
    assert len(ordering_edges) == 3
    assert {
        source for source, destination in ordering_edges if destination == store
    } < loads
    assert {
        destination for source, destination in ordering_edges if source == store
    } < loads


def test_transitive_reduction(outdir: Path, arm64: ArchContext, log):
    graph = make_graph_from_asm(
        """
str x0, [x8]
str x1, [x9]
str x2, [x10]
""",
        arm64,
        outdir,
        flags=BASIC_BLOCK_FLAGS,
    )

    # A chain, without the redundant edge from the first to the last store
    assert len(ordering_edges_of(graph)) == 2


def test_inter_block_edges(outdir: Path, arm64: ArchContext, log):
    graph = make_graph_from_asm(
        """
ldr x0, [x9]
cbz x0, end
blr x16
add x1, x1, 1
end:
ret
""",
        arm64,
        outdir,
        flags=BASIC_BLOCK_FLAGS,
    )

    (cbranch,) = nodes_with_opcode(graph, OpCodes.CBRANCH)
    (callind,) = nodes_with_opcode(graph, OpCodes.CALLIND)
    (return_node,) = nodes_with_opcode(graph, OpCodes.RETURN)
    (external,) = nodes_of_kind(graph, NodeKinds.External)

    block_of = {
        edge.source_node: edge.destination_node
        for edge in graph.edges
        if edge.kind == EdgeKinds.BasicBlockMembership
    }
    assert len(nodes_of_kind(graph, NodeKinds.BasicBlock)) == 4

    numbered_edges = {
        (edge.source_node, edge.operand_number, edge.destination_node)
        for edge in graph.edges
        if edge.kind == EdgeKinds.Control and edge.operand_number is not None
    }
    (add_block,) = {
        block_of[index] for index in nodes_with_opcode(graph, OpCodes.INT_ADD)
    } - {block_of[callind]}
    assert numbered_edges == {
        (cbranch, 0, block_of[return_node]),
        (cbranch, 1, block_of[callind]),
        (callind, 0, external),
        (callind, 1, add_block),
        (return_node, 0, external),
        # Fall-through: from the basic block node itself
        (add_block, 0, block_of[return_node]),
    }


def test_reordering_invariance(outdir: Path, arm64: ArchContext, log):
    original = """
ldr x0, [x9]
ldr x1, [x10]
add x2, x3, x4
str x2, [x8]
mul x5, x0, x1
cbz x5, end
blr x16
end:
ret
"""
    reordered = """
add x2, x3, x4
ldr x1, [x10]
ldr x0, [x9]
mul x5, x0, x1
str x2, [x8]
cbz x5, end
blr x16
end:
ret
"""
    graphs = [
        make_graph_from_asm(asm, arm64, outdir / name, flags=flags)
        for flags in (NO_BASIC_BLOCK_FLAGS, BASIC_BLOCK_FLAGS)
        for name, asm in (("original", original), ("reordered", reordered))
    ]
    original_cdg, reordered_cdg, original_grouped, reordered_grouped = graphs

    assert not are_isomorphic(original_cdg, reordered_cdg)
    assert are_isomorphic(original_grouped, reordered_grouped)


def test_reordering_invariance_x86(outdir: Path, x86_64: ArchContext, log):
    original = """
mov rax, qword ptr [rdi]
imul rcx, rdx
mov qword ptr [rsi], rcx
xor r8, r9
test rax, rax
jz end
call rbx
end:
ret
"""
    reordered = """
xor r8, r9
imul rcx, rdx
mov rax, qword ptr [rdi]
mov qword ptr [rsi], rcx
test rax, rax
jz end
call rbx
end:
ret
"""
    original_graph = make_graph_from_asm(
        original, x86_64, outdir / "original", flags=BASIC_BLOCK_FLAGS
    )
    reordered_graph = make_graph_from_asm(
        reordered, x86_64, outdir / "reordered", flags=BASIC_BLOCK_FLAGS
    )
    assert are_isomorphic(original_graph, reordered_graph)


def test_store_load_order_matters(outdir: Path, arm64: ArchContext, log):
    store_first = make_graph_from_asm(
        "str x0, [x8]\nldr x1, [x9]",
        arm64,
        outdir / "store_first",
        flags=BASIC_BLOCK_FLAGS,
    )
    load_first = make_graph_from_asm(
        "ldr x1, [x9]\nstr x0, [x8]",
        arm64,
        outdir / "load_first",
        flags=BASIC_BLOCK_FLAGS,
    )
    assert not are_isomorphic(store_first, load_first)


def test_diameter_reduction(outdir: Path, arm64: ArchContext, log):
    asm = """
add x0, x1, x2
mul x3, x4, x5
sub x6, x7, x8
eor x9, x10, x11
orr x12, x13, x14
and x15, x16, x17
"""
    flat = make_graph_from_asm(asm, arm64, outdir / "flat", flags=NO_BASIC_BLOCK_FLAGS)
    grouped = make_graph_from_asm(
        asm, arm64, outdir / "grouped", flags=BASIC_BLOCK_FLAGS
    )
    assert get_graph_diameters(grouped).both < get_graph_diameters(flat).both
