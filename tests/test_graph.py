# tests/test_graph.py
"""
Graph contract tests.

We build Module objects directly and assert that build_graph produces
the expected node kinds, edge directions, and edge attributes.

We do NOT test:
  - the exact order of nodes or edges (NetworkX does not guarantee it)
  - any analysis beyond the graph's structure
  - real Yosys files (covered by the conformance test)
"""
import logging

import networkx as nx
import pytest

from wordrev.graph import build_graph, edge_data, net_node, gate_node
from wordrev.models import Gate, Module


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def module_with(*gates: Gate, nets: dict | None = None) -> Module:
    """Build a Module with the given gates and nets (nets auto-created)."""
    m = Module("t")
    for g in gates:
        m.gates[g.name] = g
        for bits in (*g.inputs.values(), *g.outputs.values(), *g.inouts.values()):
            for b in bits:
                m.get_net(b)
    for bit_id in (nets or {}):
        m.get_net(bit_id)
    return m


def simple_and(name: str = "g", a=2, b=3, y=4) -> Gate:
    return Gate(
        name=name,
        cell_type="$_AND_",
        inputs={"A": [a], "B": [b]},
        outputs={"Y": [y]},
    )


def edge_signature(G: nx.MultiDiGraph) -> set[tuple]:
    """Order-independent signature of every edge: (u, v, pin, role)."""
    sigs = set()
    for u, v, _key, data in G.edges(keys=True, data=True):
        sigs.add((u, v, data["pin"], data["role"]))
    return sigs


# -----------------------------------------------------------------------------
# Node creation
# -----------------------------------------------------------------------------

def test_nets_become_net_nodes():
    m = module_with(simple_and())
    G = build_graph(m)
    for bit_id in m.nets:
        assert net_node(bit_id) in G.nodes


def test_gates_become_gate_nodes():
    m = module_with(simple_and())
    G = build_graph(m)
    for gate_name in m.gates:
        assert gate_node(gate_name) in G.nodes


def test_net_node_attributes():
    m = module_with(simple_and())
    m.nets[2].is_port = True
    m.nets[2].port_direction = "input"
    m.nets[2].name = "a"
    G = build_graph(m)
    data = G.nodes[net_node(2)]
    assert data["kind"] == "net"
    assert data["bit_id"] == 2
    assert data["name"] == "a"
    assert data["is_port"] is True
    assert data["port_direction"] == "input"


def test_net_node_constant_attributes():
    m = module_with(simple_and(), nets={0: None})
    m.nets[0].is_constant = True
    m.nets[0].constant_value = 0
    m.nets[0].name = "const_0"
    G = build_graph(m)
    data = G.nodes[net_node(0)]
    assert data["is_constant"] is True
    assert data["constant_value"] == 0
    assert data["name"] == "const_0"


def test_gate_node_attributes():
    gate = Gate(
        name="g1",
        cell_type="$_DFF_P_",
        inputs={"D": [2], "CLK": [3]},
        outputs={"Q": [4]},
        attributes={"src": "x.v:1"},
        parameters={"CLK_POLARITY": "1'b1"},
    )
    m = module_with(gate)
    G = build_graph(m)
    data = G.nodes[gate_node("g1")]
    assert data["kind"] == "gate"
    assert data["cell_type"] == "$_DFF_P_"
    assert data["attributes"] == {"src": "x.v:1"}
    assert data["parameters"] == {"CLK_POLARITY": "1'b1"}


# -----------------------------------------------------------------------------
# Edge direction and attributes
# -----------------------------------------------------------------------------

def test_input_edge_points_net_to_gate():
    m = module_with(simple_and())
    G = build_graph(m)
    assert G.has_edge(net_node(2), gate_node("g"))
    assert not G.has_edge(gate_node("g"), net_node(2))


def test_output_edge_points_gate_to_net():
    m = module_with(simple_and())
    G = build_graph(m)
    assert G.has_edge(gate_node("g"), net_node(4))
    assert not G.has_edge(net_node(4), gate_node("g"))


def test_edge_attributes_input():
    m = module_with(simple_and())
    G = build_graph(m)
    data = edge_data(G, net_node(2), gate_node("g"))
    assert len(data) == 1
    assert data[0]["pin"] == "A"
    assert data[0]["role"] == "input"


def test_edge_attributes_output():
    m = module_with(simple_and())
    G = build_graph(m)
    data = edge_data(G, gate_node("g"), net_node(4))
    assert len(data) == 1
    assert data[0]["pin"] == "Y"
    assert data[0]["role"] == "output"


# -----------------------------------------------------------------------------
# Inout handling
# -----------------------------------------------------------------------------

def test_inout_produces_two_directed_edges():
    gate = Gate(
        name="pad",
        cell_type="$_TRI_",
        inputs={"A": [2], "EN": [3]},
        inouts={"Y": [4]},
    )
    m = module_with(gate)
    G = build_graph(m)
    assert G.has_edge(net_node(4), gate_node("pad"))
    assert G.has_edge(gate_node("pad"), net_node(4))

    forward = edge_data(G, net_node(4), gate_node("pad"))
    backward = edge_data(G, gate_node("pad"), net_node(4))
    assert len(forward) == 1 and forward[0]["role"] == "inout"
    assert len(backward) == 1 and backward[0]["role"] == "inout"
    assert forward[0]["pin"] == "Y"
    assert backward[0]["pin"] == "Y"


# -----------------------------------------------------------------------------
# Multi-edges (two pins, same net)
# -----------------------------------------------------------------------------

def test_two_pins_on_same_net_preserved_as_multi_edges():
    """y = a & a — synthesis occasionally emits this. Both A and B pins
    feed from net 2; the MultiDiGraph must keep both edges."""
    gate = Gate(
        name="g",
        cell_type="$_AND_",
        inputs={"A": [2], "B": [2]},
        outputs={"Y": [4]},
    )
    m = module_with(gate)
    G = build_graph(m)
    edges = edge_data(G, net_node(2), gate_node("g"))
    assert len(edges) == 2
    assert {e["pin"] for e in edges} == {"A", "B"}
    assert all(e["role"] == "input" for e in edges)


# -----------------------------------------------------------------------------
# Bipartiteness
# -----------------------------------------------------------------------------

def test_all_edges_are_net_to_gate_or_gate_to_net():
    """The graph is strictly bipartite. No net→net, no gate→gate edges."""
    gate = simple_and()
    m = module_with(gate)
    G = build_graph(m)
    for u, v, _key in G.edges(keys=True):
        kinds = {u[0], v[0]}
        assert kinds == {"net", "gate"}, f"non-bipartite edge: {u} → {v}"


def test_every_edge_has_pin_and_role():
    gate = simple_and()
    m = module_with(gate)
    G = build_graph(m)
    for u, v, data in G.edges(data=True):
        assert "pin" in data, f"edge {u} → {v} missing pin"
        assert data["role"] in {"input", "output", "inout"}


# -----------------------------------------------------------------------------
# Determinism and rebuildability
# -----------------------------------------------------------------------------

def test_build_is_deterministic():
    """Rebuilding from the same Module yields the same node/edge sets."""
    gate = simple_and()
    m = module_with(gate)
    G1 = build_graph(m)
    G2 = build_graph(m)

    assert set(G1.nodes) == set(G2.nodes)
    assert edge_signature(G1) == edge_signature(G2)


def test_graph_does_not_mutate_module():
    """build_graph must not modify the Module it reads."""
    gate = simple_and()
    m = module_with(gate)
    before = (dict(m.nets), dict(m.gates), dict(m.ports))
    build_graph(m)
    after = (dict(m.nets), dict(m.gates), dict(m.ports))
    assert before == after


# -----------------------------------------------------------------------------
# Defensive path: gate references net missing from module.nets
# -----------------------------------------------------------------------------

def _module_with_missing_net() -> Module:
    """Module whose gate references a net (99) not in module.nets."""
    m = Module("t")
    m.gates["g"] = Gate(
        name="g",
        cell_type="$_BUF_",
        inputs={"A": [99]},
        outputs={"Y": [100]},
    )
    return m


def test_missing_net_creates_placeholder_node():
    m = _module_with_missing_net()
    G = build_graph(m)
    assert net_node(99) in G.nodes
    assert net_node(100) in G.nodes
    assert G.nodes[net_node(99)]["kind"] == "net"


def test_missing_net_does_not_mutate_module():
    m = _module_with_missing_net()
    before = set(m.nets)
    build_graph(m)
    assert set(m.nets) == before, (
        "build_graph must not create nets in the Module"
    )


def test_missing_net_warns(caplog):
    m = _module_with_missing_net()
    with caplog.at_level(logging.WARNING, logger="wordrev.graph"):
        build_graph(m)
    messages = [r.message for r in caplog.records]
    assert any("references net 99" in msg for msg in messages), (
        f"expected a warning about net 99; got: {messages}"
    )


def test_missing_net_warns_once_per_net(caplog):
    """A net referenced from two pins must produce exactly one warning."""
    m = Module("t")
    m.gates["g"] = Gate(
        name="g",
        cell_type="$_AND_",
        inputs={"A": [99], "B": [99]},
        outputs={"Y": [100]},
    )
    with caplog.at_level(logging.WARNING, logger="wordrev.graph"):
        build_graph(m)
    net99_warnings = [
        r for r in caplog.records if "references net 99" in r.message
    ]
    assert len(net99_warnings) == 1


# -----------------------------------------------------------------------------
# edge_data helper
# -----------------------------------------------------------------------------

def test_edge_data_returns_list():
    m = module_with(simple_and())
    G = build_graph(m)
    result = edge_data(G, net_node(2), gate_node("g"))
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["pin"] == "A"


def test_edge_data_missing_returns_empty_by_default():
    m = module_with(simple_and())
    G = build_graph(m)
    assert edge_data(G, net_node(99), gate_node("g")) == []


def test_edge_data_missing_raises_when_required():
    m = module_with(simple_and())
    G = build_graph(m)
    with pytest.raises(KeyError):
        edge_data(G, net_node(99), gate_node("g"), required=True)


def test_edge_data_required_true_on_existing_edge():
    m = module_with(simple_and())
    G = build_graph(m)
    result = edge_data(G, net_node(2), gate_node("g"), required=True)
    assert len(result) == 1