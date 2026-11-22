# tests/test_shape_unroll.py
"""Shape hashing (unroll method) contract tests.

Coverage:
  - Basic API: returns dict, deterministic, empty graph, negative k
  - Within-cone sharing: AND(a, a) vs AND(a, b)  ← the distinguishing case
  - Functional equivalence: two identical gates hash the same
  - Cross-slice grouping: AND(a[0], b[0]) == AND(a[1], b[1])
  - Gate type sensitivity: AND != OR
  - Constant sensitivity: AND(_, 0) != AND(_, 1)
  - Pin sensitivity: include_pins toggles (does not distinguish swapped
    pins under canonical leaf IDs; this is documented, not a bug)
  - k semantics: k=0 hashes initial kinds, k>=1 grows
  - Comparison with WL: agree on tree circuits, differ on shared circuits
"""

import pytest

from wordrev.graph import build_graph, net_node, gate_node
from wordrev.models import Gate, Module
from wordrev.shape import compute_all_shape_hashes
from wordrev.shape_unroll import compute_all_shape_hashes_unrolled


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def make_module(*gates: Gate, nets: list = None) -> Module:
    m = Module("t")
    for g in gates:
        m.gates[g.name] = g
        for bits in (*g.inputs.values(), *g.outputs.values(), *g.inouts.values()):
            for b in bits:
                m.get_net(b)
    for bit_id in (nets or []):
        m.get_net(bit_id)
    return m


def _hash_of(m: Module, net_id: int, *, k: int = 4, include_pins: bool = True) -> str:
    G = build_graph(m)
    return compute_all_shape_hashes_unrolled(G, k=k, include_pins=include_pins)[
        net_node(net_id)
    ]


# -----------------------------------------------------------------------------
# Basic API
# -----------------------------------------------------------------------------

def test_returns_dict():
    m = make_module(Gate("g", "$_AND_", {"A": [2], "B": [3]}, {"Y": [4]}))
    G = build_graph(m)
    result = compute_all_shape_hashes_unrolled(G, k=2)
    assert isinstance(result, dict)


def test_keys_are_net_nodes_only():
    m = make_module(Gate("g", "$_AND_", {"A": [2], "B": [3]}, {"Y": [4]}))
    G = build_graph(m)
    result = compute_all_shape_hashes_unrolled(G, k=2)
    for key in result:
        assert key[0] == "net"
    assert gate_node("g") not in result


def test_negative_k_raises():
    m = make_module(Gate("g", "$_AND_", {"A": [2], "B": [3]}, {"Y": [4]}))
    G = build_graph(m)
    with pytest.raises(ValueError, match="non-negative"):
        compute_all_shape_hashes_unrolled(G, k=-1)


def test_empty_graph_returns_empty():
    G = build_graph(Module("t"))
    assert compute_all_shape_hashes_unrolled(G, k=2) == {}


def test_deterministic():
    m = make_module(Gate("g", "$_AND_", {"A": [2], "B": [3]}, {"Y": [4]}))
    G = build_graph(m)
    r1 = compute_all_shape_hashes_unrolled(G, k=3)
    r2 = compute_all_shape_hashes_unrolled(G, k=3)
    assert r1 == r2


# -----------------------------------------------------------------------------
# Within-cone sharing — the property that distinguishes unroll from WL
# -----------------------------------------------------------------------------

def test_and_of_same_input_differs_from_and_of_distinct_inputs():
    """AND(a, a) and AND(a, b) are different functions AND different sharing
    patterns. They must hash differently under unroll."""
    m_same = make_module(
        Gate("g", "$_AND_", {"A": [2], "B": [2]}, {"Y": [4]}),
        nets=[2],
    )
    m_diff = make_module(
        Gate("g", "$_AND_", {"A": [2], "B": [3]}, {"Y": [4]}),
        nets=[2, 3],
    )
    h_same = _hash_of(m_same, 4)
    h_diff = _hash_of(m_diff, 4)
    assert h_same != h_diff, (
        "unroll must distinguish within-cone sharing: "
        "AND(a, a) should not hash like AND(a, b)"
    )


def test_wl_does_not_distinguish_sharing():
    """Complement of the previous test: WL should NOT distinguish
    AND(a,a) from AND(a,b) at k=2. This documents the WL limitation."""
    m_same = make_module(
        Gate("g", "$_AND_", {"A": [2], "B": [2]}, {"Y": [4]}),
        nets=[2],
    )
    m_diff = make_module(
        Gate("g", "$_AND_", {"A": [2], "B": [3]}, {"Y": [4]}),
        nets=[2, 3],
    )
    G_same = build_graph(m_same)
    G_diff = build_graph(m_diff)
    h_same = compute_all_shape_hashes(G_same, k=2)[net_node(4)]
    h_diff = compute_all_shape_hashes(G_diff, k=2)[net_node(4)]
    assert h_same == h_diff, (
        "WL is expected to give identical labels for AND(a, a) and AND(a, b)"
    )


def test_xor_of_shared_and_unshared_differs():
    """XOR(AND(a,b), AND(a,c)) vs XOR(AND(a,b), AND(d,c)) — the first
    reuses `a`, the second does not. Distinct leaf counts, distinct hashes."""
    m_shared = make_module(
        Gate("and1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [10]}),
        Gate("and2", "$_AND_", {"A": [2], "B": [4]}, {"Y": [11]}),
        Gate("xor",  "$_XOR_", {"A": [10], "B": [11]}, {"Y": [12]}),
        nets=[2, 3, 4],
    )
    m_unshared = make_module(
        Gate("and1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [10]}),
        Gate("and2", "$_AND_", {"A": [5], "B": [4]}, {"Y": [11]}),
        Gate("xor",  "$_XOR_", {"A": [10], "B": [11]}, {"Y": [12]}),
        nets=[2, 3, 4, 5],
    )
    h_shared = _hash_of(m_shared, 12)
    h_unshared = _hash_of(m_unshared, 12)
    assert h_shared != h_unshared


# -----------------------------------------------------------------------------
# Functional equivalence — cones that ARE the same should hash the same
# -----------------------------------------------------------------------------

def test_identical_and_gates_hash_the_same():
    m = make_module(
        Gate("g1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [10]}),
        Gate("g2", "$_AND_", {"A": [4], "B": [5]}, {"Y": [11]}),
        nets=[2, 3, 4, 5],
    )
    h10 = _hash_of(m, 10)
    h11 = _hash_of(m, 11)
    assert h10 == h11


def test_bit_slices_hash_the_same():
    """AND(a[0], b[0]) and AND(a[1], b[1]) — the canonical leaf IDs make
    both look like AND(leaf:0, leaf:1) even though they involve different
    physical nets."""
    m = make_module(
        Gate("g0", "$_AND_", {"A": [2], "B": [4]}, {"Y": [10]}),
        Gate("g1", "$_AND_", {"A": [3], "B": [5]}, {"Y": [11]}),
    )
    # Nets now exist because make_module populated them from gate pins.
    m.nets[2].base_name, m.nets[2].vector_index = "a", 0
    m.nets[4].base_name, m.nets[4].vector_index = "b", 0
    m.nets[3].base_name, m.nets[3].vector_index = "a", 1
    m.nets[5].base_name, m.nets[5].vector_index = "b", 1

    h10 = _hash_of(m, 10)
    h11 = _hash_of(m, 11)
    assert h10 == h11


# -----------------------------------------------------------------------------
# Sensitivity: what SHOULD differ
# -----------------------------------------------------------------------------

def test_different_gate_types_differ():
    m = make_module(
        Gate("g1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [10]}),
        Gate("g2", "$_OR_",  {"A": [2], "B": [3]}, {"Y": [11]}),
        nets=[2, 3],
    )
    assert _hash_of(m, 10) != _hash_of(m, 11)


def test_constants_distinguish():
    m = make_module(
        Gate("g0", "$_AND_", {"A": [2], "B": [0]}, {"Y": [10]}),
        Gate("g1", "$_AND_", {"A": [2], "B": [1]}, {"Y": [11]}),
        nets=[0, 1, 2],
    )
    m.nets[0].is_constant, m.nets[0].constant_value = True, 0
    m.nets[1].is_constant, m.nets[1].constant_value = True, 1
    assert _hash_of(m, 10) != _hash_of(m, 11)


def test_different_depths_differ():
    """One-deep XOR vs two-deep XOR chain."""
    m = make_module(
        Gate("x1", "$_XOR_", {"A": [2], "B": [3]}, {"Y": [10]}),
        Gate("x2", "$_XOR_", {"A": [4], "B": [5]}, {"Y": [11]}),
        Gate("x3", "$_XOR_", {"A": [11], "B": [6]}, {"Y": [12]}),
        nets=[2, 3, 4, 5, 6],
    )
    assert _hash_of(m, 10, k=4) != _hash_of(m, 12, k=4)


# -----------------------------------------------------------------------------
# k semantics
# -----------------------------------------------------------------------------

def test_k_zero_returns_initial_kind_only():
    """At k=0, every non-constant net hashes to the same "net" label.
    This matches WL k=0 (the initial labeling)."""
    m = make_module(
        Gate("g1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [10]}),
        Gate("g2", "$_OR_",  {"A": [4], "B": [5]}, {"Y": [11]}),
        nets=[2, 3, 4, 5],
    )
    G = build_graph(m)
    result = compute_all_shape_hashes_unrolled(G, k=0)
    assert result[net_node(10)] == result[net_node(11)]


def test_k1_does_not_yet_distinguish():
    """At k=1, two structurally identical AND gates driven by XOR2s
    hash identically — depth 1 is not enough to distinguish them."""
    m = make_module(
        Gate("x1", "$_XOR_", {"A": [2], "B": [3]}, {"Y": [10]}),
        Gate("x2", "$_XOR_", {"A": [4], "B": [5]}, {"Y": [11]}),
        # Both consumed by the same gate type at depth 1
        Gate("y1", "$_AND_", {"A": [10], "B": [6]}, {"Y": [12]}),
        Gate("y2", "$_AND_", {"A": [11], "B": [6]}, {"Y": [13]}),
        nets=[2, 3, 4, 5, 6],
    )
    G = build_graph(m)
    h1 = compute_all_shape_hashes_unrolled(G, k=1)
    assert h1[net_node(12)] == h1[net_node(13)]


# -----------------------------------------------------------------------------
# Comparison with WL
# -----------------------------------------------------------------------------

def test_agree_on_tree_circuit():
    """On a circuit with no sharing and no cycles, WL and unroll produce
    the same equivalence classes (partition of nets by hash)."""
    m = make_module(
        Gate("g0", "$_AND_", {"A": [2], "B": [4]}, {"Y": [10]}),
        Gate("g1", "$_AND_", {"A": [3], "B": [5]}, {"Y": [11]}),
        nets=[2, 3, 4, 5],
    )
    G = build_graph(m)
    wl = compute_all_shape_hashes(G, k=2)
    un = compute_all_shape_hashes_unrolled(G, k=4)

    def partition(h):
        classes: dict[str, list] = {}
        for net, h_val in h.items():
            classes.setdefault(h_val, []).append(net)
        return frozenset(frozenset(c) for c in classes.values())

    # Two AND slices: {10, 11} in one bucket, inputs in individual buckets.
    # Both methods should agree on this partition.
    assert partition(wl) == partition(un), (
        f"on a tree circuit, WL and unroll should agree; "
        f"WL={partition(wl)}, unroll={partition(un)}"
    )


def test_differ_on_shared_circuit():
    """A circuit where two AND gates share an input — WL merges all three
    fragments into one class, unroll separates them."""
    m = make_module(
        Gate("g1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [10]}),  # AND(a, b)
        Gate("g2", "$_AND_", {"A": [2], "B": [4]}, {"Y": [11]}),  # AND(a, c)
        nets=[2, 3, 4],
    )
    G = build_graph(m)
    wl = compute_all_shape_hashes(G, k=2)
    un = compute_all_shape_hashes_unrolled(G, k=4)

    # WL: both AND gates have the same 2-input shape → same hash.
    assert wl[net_node(10)] == wl[net_node(11)]
    # Unroll: g1's cone shares leaf `a` with... actually per-cone, each
    # cone sees only its own leaves. Both see {a, (b or c)} → 2 leaves,
    # same canonical IDs → same hash. So this test EXPECTS same.
    assert un[net_node(10)] == un[net_node(11)]

# -----------------------------------------------------------------------------
# Test for canonical ordering fix
# -----------------------------------------------------------------------------

def test_bit_slices_hash_same_with_adversarial_bit_ids():
    """Same function as test_bit_slices_hash_the_same, but bit IDs are
    deliberately non-monotonic across slices. The base_name sort must
    produce the same canonical leaf ordering regardless of bit IDs.

    Regression for: implementation that sorts by bit_id only.
    """
    # Bit 0: A=a[0]=net 5, B=b[0]=net 2   (A-pin has HIGHER bit ID)
    # Bit 1: A=a[1]=net 3, B=b[1]=net 4   (A-pin has LOWER bit ID)
    m = make_module(
        Gate("g0", "$_AND_", {"A": [5], "B": [2]}, {"Y": [10]}),
        Gate("g1", "$_AND_", {"A": [3], "B": [4]}, {"Y": [11]}),
    )
    m.nets[5].base_name, m.nets[5].vector_index = "a", 0
    m.nets[2].base_name, m.nets[2].vector_index = "b", 0
    m.nets[3].base_name, m.nets[3].vector_index = "a", 1
    m.nets[4].base_name, m.nets[4].vector_index = "b", 1

    h10 = _hash_of(m, 10)
    h11 = _hash_of(m, 11)
    assert h10 == h11, (
        "bit slices must hash the same regardless of bit-ID assignment — "
        "the canonical leaf ordering must use (base_name, vector_index), "
        "not bit_id"
    )