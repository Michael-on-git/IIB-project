"""
Module → NetworkX MultiDiGraph.

Node kinds:
    ("net", bit_id)   - one per net in the module
    ("gate", name)    - one per cell in the module

Edge directions:
    net  → gate    (input pin)    role="input"
    gate → net     (output pin)   role="output"
    net  ↔ gate    (inout pin)    role="inout", two directed edges

Edge attributes:
    pin  : the cell pin name (e.g. "A", "Y", "CLK")
    role : "input" | "output" | "inout"

Contracts:
    build_graph is read-only with respect to the Module it receives.
    If a gate references a net that is missing from module.nets, a
    placeholder Net node is synthesized in the graph and a warning is
    logged — the model is NOT mutated. A warning on this path always
    indicates a parser bug and should be treated as such during
    development.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator
import logging
from typing import Any, Literal

import networkx as nx

from .models import BitID, Module, Net

log = logging.getLogger(__name__)

Node = tuple[str, Any]  # ('net', BitID) | ('gate', str)
EdgeDirection = Literal["net_to_gate", "gate_to_net", "both"]

__all__ = [
    "Node",
    "net_node",
    "gate_node",
    "is_net",
    "is_gate",
    "build_graph",
    "edge_data",
    "in_edges",
    "node_label",
]


# -----------------------------------------------------------------------------
# Node key constructors & predicates
# -----------------------------------------------------------------------------


def net_node(bit_id: BitID) -> Node:
    """Canonical node key for a net. Use this instead of inline tuples."""
    return ("net", bit_id)


def gate_node(name: str) -> Node:
    """Canonical node key for a gate. Use this instead of inline tuples."""
    return ("gate", name)


def is_net(node: Node) -> bool:
    """True if `node` is a net node key."""
    return node[0] == "net"


def is_gate(node: Node) -> bool:
    """True if `node` is a gate node key."""
    return node[0] == "gate"


# -----------------------------------------------------------------------------
# Graph builder
# -----------------------------------------------------------------------------


def build_graph(module: Module) -> nx.MultiDiGraph:
    """Build the bipartite netlist graph for a module.

    Read-only with respect to `module`. See module docstring for the
    missing-net placeholder behaviour.
    """
    G = nx.MultiDiGraph()

    def _add_net_node(bit_id: BitID, net: Net) -> None:
        G.add_node(
            net_node(bit_id),
            kind="net",
            bit_id=bit_id,
            name=net.name,
            name_source=net.name_source,
            base_name=net.base_name,
            vector_index=net.vector_index,
            is_constant=net.is_constant,
            constant_value=net.constant_value,
            is_port=net.is_port,
            port_direction=net.port_direction,
        )

    def _ensure_net_node(bit_id: BitID) -> None:
        """Add the net node if not already present. Never mutates `module`."""
        key = net_node(bit_id)
        if G.has_node(key):
            return
        net = module.nets.get(bit_id)
        if net is None:
            log.warning(
                "graph: module %r references net %r not in module.nets; "
                "synthesizing placeholder (parser bug?)",
                module.name,
                bit_id,
            )
            net = Net(id=bit_id)
        _add_net_node(bit_id, net)

    # 1. Explicit nets from the module.
    for bit_id, net in module.nets.items():
        _add_net_node(bit_id, net)

    # 2. Gates and their edges. Missing nets encountered here are
    #    synthesized via _ensure_net_node without touching the model.
    for gate_name, gate in module.gates.items():
        g_node = gate_node(gate_name)
        G.add_node(
            g_node,
            kind="gate",
            cell_type=gate.cell_type,
            attributes=gate.attributes,
            parameters=gate.parameters,
        )

        def _connect(
            pin_map: dict[str, list[BitID]],
            role: str,
            direction: EdgeDirection,
        ) -> None:
            for pin, bits in pin_map.items():
                for bit_id in bits:
                    _ensure_net_node(bit_id)
                    n_node = net_node(bit_id)
                    if direction in ("net_to_gate", "both"):
                        G.add_edge(n_node, g_node, pin=pin, role=role)
                    if direction in ("gate_to_net", "both"):
                        G.add_edge(g_node, n_node, pin=pin, role=role)

        _connect(gate.inputs, "input", "net_to_gate")
        _connect(gate.outputs, "output", "gate_to_net")
        _connect(gate.inouts, "inout", "both")

    return G


# -----------------------------------------------------------------------------
# Edge & label helpers
# -----------------------------------------------------------------------------


def edge_data(
    G: nx.MultiDiGraph,
    u: Node,
    v: Node,
    *,
    required: bool = False,
) -> list[dict[str, Any]]:
    """All edge attribute dicts between u and v.

    MultiDiGraph returns G[u][v] as {key: attrs}, not attrs. This helper
    flattens it.
    """
    if not G.has_edge(u, v):
        if required:
            raise KeyError(f"no edge {u} → {v}")
        return []
    return list(G[u][v].values())


def in_edges(
    G: nx.MultiDiGraph,
    node: Node,
    roles: Collection[str] | None = None,
) -> Iterator[tuple[Node, dict[str, Any]]]:
    """Yield (predecessor, edge_attrs) for incoming edges matching `roles`."""
    for pred in G.predecessors(node):
        for attrs in G[pred][node].values():
            if roles is None or attrs.get("role") in roles:
                yield pred, attrs


def node_label(G: nx.MultiDiGraph, node: Node) -> str:
    """Human-readable label for a net or gate node.

    Preference for nets:
        1. "{base_name}[{vector_index}]" when both are set (bus bits,
           populated by the parser)
        2. name (any other named net)
        3. "n{bit_id}" (unnamed nets)

    Gate nodes fall back to their name from the node tuple.

    Works for any node in the graph. Never raises on missing attributes —
    an unannotated graph yields "n{id}" for nets, which is the intended
    debug label.
    """
    data = G.nodes.get(node, {})
    base = data.get("base_name")
    idx = data.get("vector_index")
    if base is not None and idx is not None:
        return f"{base}[{idx}]"
    if data.get("name"):
        return str(data["name"])
    if is_gate(node):
        return str(node[1])
    return f"n{node[1]}"