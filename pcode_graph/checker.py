from pcode_graph.graph import CDG, EdgeKinds, EdgeProxy, NodeIndex, NodeKinds

from pcode_graph.pcode import NUM_OPERANDS, OpCodes

# Node operands are either operation inputs or control successors.
# There is only one successor except for the special successor of CBRANCH, CALL, CALLIND and CALLOTHER.
# Only OutputRegister, Phi and CALLOTHER Operation can have more than 2 inputs.
MAX_OPERANDS = 9


class ValidationError(Exception): ...


def check(condition: bool, message: str | None = None):
    if not condition:
        raise ValidationError(message)


def check_graph(graph: CDG):
    """Checks the coherency of this graph.

    Can be used to verify that the graph remains valid after a set of transformations
    made by a neural network for instance.

    Raises ValidationError if invalid.
    """

    has_basic_blocks = any(node.kind == NodeKinds.BasicBlock for node in graph.nodes)
    has_cfg = False

    # Check edges are valid
    for edge in graph.edges:
        # check(isinstance(edge.source_node, int))
        # check(isinstance(edge.destination_node, int))
        has_cfg |= edge.kind == EdgeKinds.Control
        check(0 <= edge.source_node < len(graph.nodes))
        check(0 <= edge.destination_node < len(graph.nodes))
        check(edge.operand_number is None or 0 <= edge.operand_number < MAX_OPERANDS)
        if edge.kind == EdgeKinds.BasicBlockMembership:
            check(edge.operand_number is None)
            check(graph.nodes[edge.source_node].kind == NodeKinds.Operation)
            check(graph.nodes[edge.destination_node].kind == NodeKinds.BasicBlock)

    # Check nodes attributes are coherent with node kind
    proxy_dict = graph.compute_edge_proxy()

    for index, node in enumerate(graph.nodes):

        if node.value:
            check(node.kind == NodeKinds.Constant)

        check(
            (not node.register_name)
            ^ (node.kind in {NodeKinds.InputRegister, NodeKinds.OutputRegister})
        )

        proxy = proxy_dict[index]
        control_succs = list(proxy.get_control_successors())
        data_preds = list(proxy.get_data_predecessors())

        if node.kind != NodeKinds.Operation:
            check(node.opcode is None)
        else:
            check(
                node.opcode is not None
                and NUM_OPERANDS[node.opcode] in {-1, len(data_preds)}
            )
            # TODO: move this in check dataflow below
            match node.opcode:
                case OpCodes.LOAD:
                    check(len(data_preds) in {1, 2})
                case OpCodes.STORE:
                    # Accessed memory is implicit
                    check(
                        not list(proxy.get_data_successors()),
                        f"Node {index} {node} with data successors",
                    )

            data_numbers: list[int] = []
            control_numbers: list[int] = []
            for edge in proxy.get_in_edges():
                if edge.kind == EdgeKinds.Data:
                    check(edge.operand_number is not None)
                    data_numbers.append(edge.operand_number)  # type: ignore

            for edge in proxy.get_out_edges():
                if edge.kind == EdgeKinds.Control:
                    if edge.operand_number is not None:
                        control_numbers.append(edge.operand_number)
                    else:
                        # Only ordering edges between side-effects are unnumbered
                        check(
                            has_basic_blocks,
                            "Unnumbered control edge without basic block",
                        )
            check(sorted(data_numbers) == list(range(len(data_preds))))
            check(sorted(control_numbers) == list(range(len(control_numbers))))
            if not has_basic_blocks:
                check(
                    len(control_numbers) == len(control_succs),
                    "Unexpected control successor",
                )

        if has_basic_blocks:
            check_basic_block_structure(graph, index, proxy)
        elif has_cfg:
            # Check CFG coherency
            match len(control_succs):
                case 0:
                    check(
                        node.kind
                        in {
                            NodeKinds.External,
                            NodeKinds.End,
                            NodeKinds.InputRegister,
                            NodeKinds.OutputRegister,
                            NodeKinds.Phi,
                            NodeKinds.Constant,
                        },
                        f"Node {index} {node} without successor",
                    )
                case 1:
                    check(node.kind in {NodeKinds.Begin, NodeKinds.Operation})
                    if node.kind == NodeKinds.Operation:
                        check(
                            node.opcode
                            not in {
                                OpCodes.CBRANCH,
                                OpCodes.CALLIND,
                                OpCodes.CALL,
                            },
                            f"Node {index} {node} with only 1 successors",
                        )
                case 2:
                    check(
                        node.kind == NodeKinds.Operation
                        and node.opcode
                        in {
                            OpCodes.CBRANCH,
                            OpCodes.CALLIND,
                            OpCodes.CALL,
                            OpCodes.CALLOTHER,
                        },
                        f"Node {index} {node} with 2 successors",
                    )
                case _:
                    raise ValidationError("Unexpected number of successors")

        # Check dataflow coherency
        match len(data_preds):
            case 0:
                check(
                    node.kind
                    in {
                        NodeKinds.Begin,
                        NodeKinds.InputRegister,
                        NodeKinds.OutputRegister,
                        NodeKinds.Constant,
                        NodeKinds.External,
                        NodeKinds.End,
                        NodeKinds.Operation,  # For example the CALL operation
                        NodeKinds.BasicBlock,
                    },
                    f"Node {index} {node} without data predecessors",
                )
            case 1 | 2:
                check(
                    node.kind
                    in {
                        NodeKinds.Operation,
                        NodeKinds.Phi,
                        NodeKinds.OutputRegister,
                    },
                    f"Node {index} {node} with {len(data_preds)} data predecessors",
                )
            case _:
                if node.kind not in {NodeKinds.Phi, NodeKinds.OutputRegister}:
                    check(
                        (
                            node.kind == NodeKinds.Operation
                            and node.opcode == OpCodes.CALLOTHER
                        ),
                        f"Node {index} {node} with {len(data_preds)} data predecessors",
                    )


def check_basic_block_structure(graph: CDG, index: NodeIndex, proxy: EdgeProxy):
    """Checks the control flow around one node belonging to a graph with basic blocks."""

    node = graph.nodes[index]
    out_edges = list(proxy.get_out_edges())
    in_edges = list(proxy.get_in_edges())
    memberships = [
        edge for edge in out_edges if edge.kind == EdgeKinds.BasicBlockMembership
    ]
    numbered_control_succs = [
        edge
        for edge in out_edges
        if edge.kind == EdgeKinds.Control and edge.operand_number is not None
    ]

    match node.kind:
        case NodeKinds.Operation:
            check(len(memberships) == 1, f"Node {index} {node} not in one basic block")
            if node.opcode in {OpCodes.CBRANCH, OpCodes.CALL, OpCodes.CALLIND}:
                check(
                    len(numbered_control_succs) == 2,
                    f"Node {index} {node} without 2 successors",
                )
            elif node.opcode == OpCodes.CALLOTHER:
                check(len(numbered_control_succs) <= 2)
            else:
                check(len(numbered_control_succs) <= 1)
            for edge in out_edges:
                if edge.kind == EdgeKinds.Control and edge.operand_number is None:
                    check(
                        graph.nodes[edge.destination_node].kind == NodeKinds.Operation
                    )

        case NodeKinds.BasicBlock:
            check(not memberships)
            check(
                any(edge.kind == EdgeKinds.BasicBlockMembership for edge in in_edges),
                f"Empty basic block {index}",
            )
            check(all(edge.kind != EdgeKinds.Data for edge in in_edges + out_edges))
            check(len(numbered_control_succs) <= 1)

        case NodeKinds.Begin:
            check(len(list(proxy.get_control_successors())) == 1)

        case _:
            check(not memberships)
            check(
                not list(proxy.get_control_successors()),
                f"Node {index} {node} with control successors",
            )
