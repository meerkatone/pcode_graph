from pathlib import Path
from loguru import logger
from pyvis.network import Network

from pcode_graph.graph import CDG, EdgeKinds, NodeKinds
from pcode_graph.pcode import JUMP_OPCODES


def draw_graph(
    graph: CDG,
    out_file: Path,
    open_browser: bool = False,
    print_ids: bool = False,
):
    """Creates a HTML representation of the given graph."""

    parent_dir = out_file.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    network = Network(width="1200", height="1200", directed=True)

    # Create nodes
    for index, node in enumerate(graph.nodes):
        style: dict

        match node.kind:
            case NodeKinds.InputRegister | NodeKinds.ReadMemory:
                style = dict(color="#c0c0ff")  # Light Blue
            case NodeKinds.OutputRegister | NodeKinds.WrittenMemory:
                style = dict(color="#7070da")  # Blue
            case NodeKinds.Constant:
                style = dict(color="#ff8080")  # Red
            case NodeKinds.Begin | NodeKinds.End | NodeKinds.External:
                style = dict(color="#60d060", shape="box")
            case NodeKinds.BasicBlock:
                style = dict(color="#c5c5c5", shape="box")
            case NodeKinds.Phi:
                style = dict(color="#c0c0ff", shape="diamond")
            case NodeKinds.Operation:
                if node.opcode in JUMP_OPCODES:
                    # Jumps in light Green
                    style = dict(color="#60d060", shape="box")
                else:
                    # Other operations in Grey
                    style = dict(color="#c5c5c5", shape="box")
            case _:
                raise ValueError(node.kind)

        label = str(node)
        if print_ids:
            label += " " + str(index)
        style["label"] = label
        network.add_node(index, **style)

    edge_colors = {
        EdgeKinds.Data: "#a0a0e0",
        EdgeKinds.Control: "#40c040",
        EdgeKinds.BasicBlockMembership: "#808080",
    }

    # Create edges
    for edge in graph.edges:
        network.add_edge(
            edge.source_node,
            edge.destination_node,
            style={"dashes": edge.kind == EdgeKinds.Control},
            label=str(edge.operand_number) if edge.operand_number is not None else "",
            color=edge_colors[edge.kind],
        )

    # Layout
    network.force_atlas_2based(spring_length=75)

    network.write_html(out_file.as_posix(), open_browser=open_browser)
    logger.info(f"Written {out_file}")
