"""Add BasicBlock nodes to CDG.

The resulting graph is invariant to semantic-preserving reorderings of instructions
and should have a smaller diameter:

- Each basic block becomes a dedicated node linked to its operations by Membership edges.
- Inside a basic block, Control edges only remain between operations with side effects
  whose relative order matters (calls and memory accesses), as the transitive reduction
  of their ordering constraints.
- Between basic blocks, Control edges go from the block terminator to the BasicBlock
  node of the destination. Blocks without terminator are linked from their own
  BasicBlock node, as their last operation is arbitrary.
"""

from collections import Counter, defaultdict
from dataclasses import replace
from enum import Enum, auto

from pcode_graph.graph import CDG, Edge, EdgeKinds, Node, NodeIndex, NodeKinds
from pcode_graph.pcode import OpCodes

CALL_OPCODES = frozenset({OpCodes.CALL, OpCodes.CALLIND, OpCodes.CALLOTHER})

# Operations which always end their basic block, whatever their number of successors
TERMINATOR_OPCODES = frozenset(
    {
        OpCodes.BRANCH,
        OpCodes.BRANCHIND,
        OpCodes.CBRANCH,
        OpCodes.CALL,
        OpCodes.CALLIND,
        OpCodes.RETURN,
    }
)


class SideEffect(Enum):
    NONE = auto()
    # Commutes with other reads
    READ = auto()
    # Memory write or call: ordered with respect to any other side effect
    BARRIER = auto()


class BasicBlockGrouper:
    """Rewrites the control flow of a CDG in place, see module documentation."""

    def __init__(self, graph: CDG) -> None:
        assert not any(node.kind == NodeKinds.BasicBlock for node in graph.nodes)
        self.graph = graph
        self.control_successors: defaultdict[NodeIndex, list[NodeIndex]] = defaultdict(
            list
        )
        self.control_predecessor_counts: Counter[NodeIndex] = Counter()

        # Single pass over edges instead of per-node proxies
        for edge in graph.edges:
            if edge.kind == EdgeKinds.Control:
                self.control_successors[edge.source_node].append(edge.destination_node)
                self.control_predecessor_counts[edge.destination_node] += 1

        self.operations = [
            index
            for index, node in enumerate(graph.nodes)
            if node.kind == NodeKinds.Operation
        ]
        self.terminators = {
            index
            for index in self.operations
            if graph.nodes[index].opcode in TERMINATOR_OPCODES
            or len(self.control_successors[index]) > 1
        }
        self.block_node_of: dict[NodeIndex, NodeIndex] = {}
        self.last_operations: set[NodeIndex] = set()

    def next_in_block(self, index: NodeIndex) -> NodeIndex | None:
        successors = self.control_successors[index]
        if index in self.terminators or len(successors) != 1:
            return None
        successor = successors[0]
        if (
            successor == index
            or self.graph.nodes[successor].kind != NodeKinds.Operation
            or self.control_predecessor_counts[successor] != 1
        ):
            return None
        return successor

    def side_effect(self, index: NodeIndex) -> SideEffect:
        opcode = self.graph.nodes[index].opcode
        if opcode in CALL_OPCODES or opcode == OpCodes.STORE:
            return SideEffect.BARRIER
        if opcode == OpCodes.LOAD:
            return SideEffect.READ
        return SideEffect.NONE

    def build_blocks(self) -> list[list[NodeIndex]]:
        """Returns operations grouped by block, in control flow order."""

        next_operations = {
            index: self.next_in_block(index) for index in self.operations
        }
        followers = set(next_operations.values())
        # Leaders first, then any operation of a cycle without entry point
        candidates = [index for index in self.operations if index not in followers]
        candidates += [index for index in self.operations if index in followers]

        blocks: list[list[NodeIndex]] = []
        for leader in candidates:
            if leader in self.block_node_of:
                continue
            block_node = self.graph.add_node(Node(NodeKinds.BasicBlock))
            block: list[NodeIndex] = []
            current: NodeIndex | None = leader
            while current is not None and current not in self.block_node_of:
                block.append(current)
                self.block_node_of[current] = block_node
                current = next_operations[current]
            self.last_operations.add(block[-1])
            blocks.append(block)
        return blocks

    def ordering_edges(self, block: list[NodeIndex]) -> list[Edge]:
        """Transitive reduction of side-effect ordering constraints, in linear time.

        A read only depends on the last barrier. A barrier depends on the reads
        since the last barrier or, if there is none, on the last barrier itself.
        """

        edges: list[Edge] = []
        last_barrier: NodeIndex | None = None
        reads_since_barrier: list[NodeIndex] = []
        for index in block:
            match self.side_effect(index):
                case SideEffect.READ:
                    if last_barrier is not None:
                        edges.append(Edge(last_barrier, index, EdgeKinds.Control, None))
                    reads_since_barrier.append(index)
                case SideEffect.BARRIER:
                    sources = reads_since_barrier or (
                        [] if last_barrier is None else [last_barrier]
                    )
                    edges += [
                        Edge(source, index, EdgeKinds.Control, None)
                        for source in sources
                    ]
                    last_barrier, reads_since_barrier = index, []
        return edges

    def inter_block_edge(self, edge: Edge) -> Edge | None:
        """Redirects a control edge between blocks, returns None for intra-block edges."""

        source = edge.source_node
        if source in self.block_node_of:
            if source not in self.last_operations:
                return None
            if source not in self.terminators:
                source = self.block_node_of[source]
        destination = self.block_node_of.get(
            edge.destination_node, edge.destination_node
        )
        return replace(edge, source_node=source, destination_node=destination)

    def process(self) -> None:
        blocks = self.build_blocks()
        edges = [edge for edge in self.graph.edges if edge.kind != EdgeKinds.Control]
        for block in blocks:
            block_node = self.block_node_of[block[0]]
            edges += [
                Edge(operation, block_node, EdgeKinds.BasicBlockMembership, None)
                for operation in block
            ]
            edges += self.ordering_edges(block)
        for edge in self.graph.edges:
            if edge.kind == EdgeKinds.Control and (
                inter_block_edge := self.inter_block_edge(edge)
            ):
                edges.append(inter_block_edge)
        self.graph.edges = edges


def make_basic_blocks(graph: CDG) -> None:
    """Restructures the given CDG in place around BasicBlock nodes."""

    BasicBlockGrouper(graph).process()
