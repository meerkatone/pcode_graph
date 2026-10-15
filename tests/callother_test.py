from pcode_graph.graph import EdgeKinds, NodeKinds
from pcode_graph.pcode import OpCodes
from .helpers import make_graph_from_asm
from .fixtures import (
    check_inputs,
    check_ops,
    check_outputs,
    log,
    outdir,
    arm64,
    x86_64,
    ArchContext,
)
from pathlib import Path


def test_brk(outdir: Path, arm64: ArchContext, log):

    make_graph_from_asm(
        """
brk #0x5519
""",
        arm64,
        outdir,
    )


def test_syscall_x86(outdir: Path, x86_64: ArchContext, log):
    graph = make_graph_from_asm(
        """
mov eax, 0x3c
xor edi, edi
syscall
mov ebx, eax
""",
        x86_64,
        outdir,
    )

    check_ops(graph, "INT_XOR", "CALLOTHER")
    
    # For pypcode 4.0.0
    # check_inputs(graph, "EDI", "#0x3c")  
    # check_outputs(graph, "RAX", "RDI", "RBX")

    # new behavior in pypcode 4.0.1
    check_inputs(graph, 'rflags', '#0x9', '#0x3c', 'EDI')
    check_outputs(graph, 'RDI', 'RAX', 'RCX', 'R11', 'RBX')

    (syscall,) = [
        index
        for index, node in enumerate(graph.nodes)
        if node.opcode == OpCodes.CALLOTHER
    ]
    proxy = graph.compute_edge_proxy()

    assert graph.nodes[syscall].user_defined_callee == "syscall"

    # The index of the user-defined operation is skipped, like the target of a CALL
    assert list(proxy[syscall].get_data_predecessors()) == []

    # Like a call to unknown code: operand 0 is the callee, operand 1 the return path
    control_edges = {
        (edge.operand_number, graph.nodes[edge.destination_node].kind)
        for edge in graph.edges
        if edge.source_node == syscall and edge.kind == EdgeKinds.Control
    }
    assert control_edges == {(0, NodeKinds.External), (1, NodeKinds.End)}

    # As for CALLIND, we do not resolve the calling convention:
    # EBX still comes from the syscall number.
    (rbx,) = [
        index
        for index, node in enumerate(graph.nodes)
        if node.kind == NodeKinds.OutputRegister and node.register_name == "RBX"
    ]
    (rbx_pred,) = list(proxy[rbx].get_data_predecessors())
    assert str(graph.nodes[rbx_pred]) == "#0x3c"


def test_svc_immediate_arm(outdir: Path, arm64: ArchContext, log):
    graph = make_graph_from_asm(
        """
mov x8, #93
svc #0x80
mov x1, x0
""",
        arm64,
        outdir,
    )

    check_ops(graph, "CALLOTHER")
    check_inputs(graph, "x0", "#0x5d", "#0x80")
    check_outputs(graph, "x8", "x1")

    (syscall,) = [
        index
        for index, node in enumerate(graph.nodes)
        if node.opcode == OpCodes.CALLOTHER
    ]

    # Only the immediate remains, as first operand
    data_inputs = [
        (str(graph.nodes[edge.source_node]), edge.operand_number)
        for edge in graph.edges
        if edge.destination_node == syscall and edge.kind == EdgeKinds.Data
    ]
    assert data_inputs == [("#0x80", 0)]
    assert len(list(graph.compute_edge_proxy()[syscall].get_control_successors())) == 2


def test_callother_with_output_x86(outdir: Path, x86_64: ArchContext, log):
    graph = make_graph_from_asm("rdtsc", x86_64, outdir)

    check_ops(graph, "CALLOTHER", "SUBPIECE")
    check_outputs(graph, "EAX", "EDX")

    (rdtsc,) = [
        index
        for index, node in enumerate(graph.nodes)
        if node.opcode == OpCodes.CALLOTHER
    ]
    (subpiece,) = [
        index
        for index, node in enumerate(graph.nodes)
        if node.opcode == OpCodes.SUBPIECE
    ]

    # Already part of the dataflow, the operation now also branches to unknown code
    control_edges = {
        (edge.operand_number, edge.destination_node)
        for edge in graph.edges
        if edge.source_node == rdtsc and edge.kind == EdgeKinds.Control
    }
    (external,) = [
        index
        for index, node in enumerate(graph.nodes)
        if node.kind == NodeKinds.External
    ]
    assert control_edges == {(0, external), (1, subpiece)}


def test_int_0x80_x86(outdir: Path, x86_64: ArchContext, log):
    graph = make_graph_from_asm("int 0x80", x86_64, outdir)

    check_ops(graph, "CALLOTHER", "CALLIND")
    check_inputs(graph, "#-0x80")

    (swi,) = [
        index
        for index, node in enumerate(graph.nodes)
        if node.opcode == OpCodes.CALLOTHER
    ]
    (callind,) = [
        index
        for index, node in enumerate(graph.nodes)
        if node.opcode == OpCodes.CALLIND
    ]
    (external,) = [
        index
        for index, node in enumerate(graph.nodes)
        if node.kind == NodeKinds.External
    ]

    proxy = graph.compute_edge_proxy()

    # The CALLOTHER computes the target of the CALLIND
    assert list(proxy[callind].get_data_predecessors()) == [swi]

    # Both operations may jump to unknown code, then fall through
    assert set(proxy[swi].get_control_successors()) == {external, callind}
    assert external in set(proxy[callind].get_control_successors())

def test_simd_ext(outdir: Path, arm64: ArchContext, log):
    g = make_graph_from_asm(
        """
ext v0.16b, v1.16b, v2.16b, #3
""",
        arm64,
        outdir,
    )

    check_inputs(g, "q1", "q2", "#0x3", "#0x1")
    check_outputs(g, "q0")
    check_ops(g, "CALLOTHER")
    
    proxy = g.compute_edge_proxy()
    for n, node in enumerate(g.nodes):
        if node.opcode == OpCodes.CALLOTHER:
            assert len(list(proxy[n].get_data_predecessors()) ) == 4
            break
    else:
        raise AssertionError("CALLOTHER node not found")
    