# tests/test_shape.py
import pytest
from wordrev.models import Module, Gate
from wordrev.graph import build_graph, net_node, gate_node
from wordrev.shape import compute_all_shape_hashes, shape_hash, backward_cone


# -----------------------------------------------------------------------------
# Fixtures & Helpers
# -----------------------------------------------------------------------------

def create_mock_module() -> Module:
    """
    Constructs a known module hierarchy:
    
    [const 0] ---> (gate $and1: A) 
                         |---------> [net 2] ---> (gate $or1: A) ---> [net 3]
    [net 1 (IN)] -> (gate $and1: B)               [net 4 (IN)] -> (gate $or1: B)
    """
    mod = Module(name="top")
    
    # Nets
    c0 = mod.get_net(0)
    c0.is_constant, c0.constant_value = True, 0
    
    n1 = mod.get_net(1)
    n1.name, n1.is_port, n1.port_direction = "in_a", True, "input"
    
    n2 = mod.get_net(2)
    n2.name = "w1"
    
    n3 = mod.get_net(3)
    n3.name, n3.is_port, n3.port_direction = "out_y", True, "output"
    
    n4 = mod.get_net(4)
    n4.name, n4.is_port, n4.port_direction = "in_b", True, "input"
    
    # Gates
    g1 = Gate(name="$and1", cell_type="$and", inputs={"A": [0], "B": [1]}, outputs={"Y": [2]})
    g2 = Gate(name="$or1", cell_type="$or", inputs={"A": [2], "B": [4]}, outputs={"Y": [3]})
    
    mod.gates[g1.name] = g1
    mod.gates[g2.name] = g2
    
    return mod


# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------

def test_leaf_and_constant_initialization():
    """Verify that primary inputs and constant nets retain fixed baseline labels."""
    mod = create_mock_module()
    G = build_graph(mod)

    hashes_k0 = compute_all_shape_hashes(G, k=0)
    hashes_k2 = compute_all_shape_hashes(G, k=2)

    # Leaf nets (no driving gate) must maintain invariant hashes across rounds
    assert hashes_k0[net_node(0)] == hashes_k2[net_node(0)]  # Constant 0
    assert hashes_k0[net_node(1)] == hashes_k2[net_node(1)]  # Primary input 1
    assert hashes_k0[net_node(4)] == hashes_k2[net_node(4)]  # Primary input 4

    # Constants and plain primary inputs should have distinct base labels
    assert hashes_k0[net_node(0)] != hashes_k0[net_node(1)]

    # All non-constant leaves must hash identically to each other.
    # If this fails, leaves are being given unique labels (e.g. net names),
    # which breaks grouping at the leaf boundary.
    assert hashes_k0[net_node(1)] == hashes_k0[net_node(4)]

def test_bipartite_wl_propagation():
    """Verify that shape hashes capture deeper upstream structure as k grows."""
    mod = create_mock_module()
    G = build_graph(mod)

    h_k0 = compute_all_shape_hashes(G, k=0)
    h_k1 = compute_all_shape_hashes(G, k=1)
    h_k2 = compute_all_shape_hashes(G, k=2)

    # k=0: every non-constant net is just "net".
    assert h_k0[net_node(1)] == h_k0[net_node(2)] == h_k0[net_node(3)] == h_k0[net_node(4)]

    # k=1: net 2 (driven by $and1) is distinguished from leaf primary
    # inputs 1 and 4, because its hash now incorporates the driver gate.
    assert h_k1[net_node(2)] != h_k1[net_node(1)]
    assert h_k1[net_node(2)] != h_k1[net_node(4)]

    # k=1: net 3 (driven by $or1) is distinguished from leaves and from
    # net 2 (different driver gate types).
    assert h_k1[net_node(3)] != h_k1[net_node(1)]
    assert h_k1[net_node(3)] != h_k1[net_node(2)]

    # Distinctions persist at k=2 (WL monotonicity: once distinguished,
    # always distinguished at deeper k). Hash *values* change between
    # rounds — that is expected — but the *equivalence relation* refines
    # monotonically.
    assert h_k2[net_node(2)] != h_k2[net_node(1)]
    assert h_k2[net_node(3)] != h_k2[net_node(2)]

def test_isomorphic_cone_matching():
    """Two independent nets driven by identical sub-graphs must produce identical hashes."""
    mod = Module(name="iso_test")
    
    # Branch A: IN1 -> NOT -> Net 10
    mod.get_net(1)
    mod.gates["g1"] = Gate(name="g1", cell_type="$not", inputs={"A": [1]}, outputs={"Y": [10]})
    
    # Branch B: IN2 -> NOT -> Net 20 (structurally identical to Branch A)
    mod.get_net(2)
    mod.gates["g2"] = Gate(name="g2", cell_type="$not", inputs={"A": [2]}, outputs={"Y": [20]})
    
    G = build_graph(mod)
    hashes = compute_all_shape_hashes(G, k=2)
    
    assert hashes[net_node(10)] == hashes[net_node(20)]


def test_pin_sensitivity_toggle():
    """Swapping pin attachments should produce different hashes iff include_pins=True."""
    # Module A: Constant 0 to Pin A, Net 2 to Pin B
    mod_a = Module(name="mod_a")
    c0_a = mod_a.get_net(1)
    c0_a.is_constant, c0_a.constant_value = True, 0
    mod_a.get_net(2)
    mod_a.get_net(10)
    mod_a.gates["g"] = Gate(name="g", cell_type="$sub", inputs={"A": [1], "B": [2]}, outputs={"Y": [10]})
    
    # Module B: Net 2 to Pin A, Constant 0 to Pin B
    mod_b = Module(name="mod_b")
    c0_b = mod_b.get_net(1)
    c0_b.is_constant, c0_b.constant_value = True, 0
    mod_b.get_net(2)
    mod_b.get_net(10)
    mod_b.gates["g"] = Gate(name="g", cell_type="$sub", inputs={"A": [2], "B": [1]}, outputs={"Y": [10]})
    
    Ga = build_graph(mod_a)
    Gb = build_graph(mod_b)
    
    # Pin sensitive (default): different pin mappings yield different hashes
    hash_a_pin = shape_hash(Ga, net_node(10), k=1, include_pins=True)
    hash_b_pin = shape_hash(Gb, net_node(10), k=1, include_pins=True)
    assert hash_a_pin != hash_b_pin
    
    # Pin insensitive: permutation invariant
    hash_a_nopin = shape_hash(Ga, net_node(10), k=1, include_pins=False)
    hash_b_nopin = shape_hash(Gb, net_node(10), k=1, include_pins=False)
    assert hash_a_nopin == hash_b_nopin


def test_inout_net_is_not_a_leaf():
    """An inout connection must drive refinement of the net's hash.

    If the inout edge is treated as a driver (which it should be), the
    net's label changes at k=1 because it now incorporates the driving
    gate's type. If someone removes 'inout' from the role filter in
    _has_gate_driver or the net update loop, the net silently becomes a
    leaf, its label stays 'net' forever, and this test catches it."""
    mod = Module(name="inout_mod")
    mod.get_net(1)  # primary input
    mod.get_net(2)  # pad net (inout)
    mod.gates["io_gate"] = Gate(
        name="io_gate",
        cell_type="$iobuf",
        inputs={"I": [1]},
        inouts={"IO": [2]},
    )
    G = build_graph(mod)

    h_k0 = compute_all_shape_hashes(G, k=0)
    h_k1 = compute_all_shape_hashes(G, k=1)

    assert h_k0[net_node(2)] != h_k1[net_node(2)], (
        "inout-driven net did not refine at k=1 — it may be treated as a leaf"
    )


def test_backward_cone_traversal():
    """Verify depth-bounded cone traversal counts gate transitions correctly."""
    mod = create_mock_module()
    G = build_graph(mod)
    
    # Net 3 -> Gate $or1 (depth 1) -> Net 2 (depth 1) -> Gate $and1 (depth 2) -> Net 0/1 (depth 2)
    cone_k1 = backward_cone(G, net_node(3), k=1)
    nodes_k1 = {node for node, depth in cone_k1}
    
    assert net_node(3) in nodes_k1
    assert gate_node("$or1") in nodes_k1
    assert net_node(2) in nodes_k1
    assert gate_node("$and1") not in nodes_k1
    
    cone_k2 = backward_cone(G, net_node(3), k=2)
    nodes_k2 = {node for node, depth in cone_k2}
    
    assert gate_node("$and1") in nodes_k2
    assert net_node(0) in nodes_k2


def test_invalid_parameters():
    """Ensure proper exception handling for negative k or invalid node requests."""
    mod = create_mock_module()
    G = build_graph(mod)
    
    with pytest.raises(ValueError, match="k must be non-negative"):
        compute_all_shape_hashes(G, k=-1)
        
    with pytest.raises(ValueError, match="shape_hash requires a net node"):
        shape_hash(G, gate_node("$and1"), k=1)

def test_partition_stability():
    """WL partitions refine monotonically: distinctions never collapse.

    Note: hash *values* do not stabilize in standard WL, because each
    round rehashes with the previous round's label as the base. What
    stabilizes is the *equivalence relation* — which nets have equal
    hashes. Once two nets are distinguished at round k, they remain
    distinguished at every round k' > k. This is the actual convergence
    property of 1-WL, and the reason k can be a bounded constant.
    """
    mod = create_mock_module()
    G = build_graph(mod)

    nets = [net_node(i) for i in range(5)]
    h = {k: compute_all_shape_hashes(G, k=k) for k in range(6)}

    for k in range(5):
        for a in nets:
            for b in nets:
                if h[k][a] != h[k][b]:
                    assert h[k + 1][a] != h[k + 1][b], (
                        f"{a} and {b} distinguished at k={k} but "
                        f"collapsed at k={k + 1}"
                    )

def test_partition_reaches_fixed_point():
    """The partition into equivalence classes stabilizes after enough
    rounds. This is the real reason k can be a bounded constant: beyond
    some K, running more rounds produces no new distinctions. This test
    finds K for the mock module and verifies stability beyond it."""
    mod = create_mock_module()
    G = build_graph(mod)

    nets = [net_node(i) for i in range(5)]

    def partition_at(k):
        h = compute_all_shape_hashes(G, k=k)
        classes: dict[str, list] = {}
        for net in nets:
            classes.setdefault(h[net], []).append(net)
        return frozenset(frozenset(c) for c in classes.values())

    for k in range(10):
        if partition_at(k) == partition_at(k + 1):
            for j in range(k, 10):
                assert partition_at(k) == partition_at(j), (
                    f"partition stable at k={k} but changed at k={j}"
                )
            return
    pytest.fail("partition never stabilized within 10 rounds")

def test_shape_hash_matches_batch():
    """shape_hash is a convenience wrapper; it must agree with the batch
    API for every net. Guards against divergent implementations."""
    mod = create_mock_module()
    G = build_graph(mod)

    batch = compute_all_shape_hashes(G, k=2)
    for net in (net_node(0), net_node(1), net_node(2), net_node(3), net_node(4)):
        assert shape_hash(G, net, k=2) == batch[net], (
            f"shape_hash and compute_all_shape_hashes disagree on {net}"
        )

def test_shape_hash_matches_batch_no_pins():
    mod = create_mock_module()
    G = build_graph(mod)

    batch = compute_all_shape_hashes(G, k=2, include_pins=False)
    for net in (net_node(2), net_node(3)):
        assert shape_hash(G, net, k=2, include_pins=False) == batch[net]

def test_pin_toggle_on_commutative_gate():
    """AND(a, b) and AND(b, a) differ with include_pins=True but match
    with include_pins=False. This is the case that matters in practice:
    synthesis may swap input pins on commutative gates, and pin-blind
    hashing is what recovers the equivalence."""
    # Module A: AND(A=const0, B=net2)
    mod_a = Module(name="mod_a")
    c0_a = mod_a.get_net(1)
    c0_a.is_constant, c0_a.constant_value = True, 0
    mod_a.get_net(2)
    mod_a.get_net(10)
    mod_a.gates["g"] = Gate(
        name="g", cell_type="$_AND_",
        inputs={"A": [1], "B": [2]}, outputs={"Y": [10]},
    )

    # Module B: AND(A=net2, B=const0) — pins swapped
    mod_b = Module(name="mod_b")
    c0_b = mod_b.get_net(1)
    c0_b.is_constant, c0_b.constant_value = True, 0
    mod_b.get_net(2)
    mod_b.get_net(10)
    mod_b.gates["g"] = Gate(
        name="g", cell_type="$_AND_",
        inputs={"A": [2], "B": [1]}, outputs={"Y": [10]},
    )

    Ga = build_graph(mod_a)
    Gb = build_graph(mod_b)

    # Pin-sensitive: swapped inputs → different hashes
    assert shape_hash(Ga, net_node(10), k=1, include_pins=True) != \
           shape_hash(Gb, net_node(10), k=1, include_pins=True), \
        "pin-sensitive hashing should distinguish swapped commutative inputs"

    # Pin-insensitive: swapped inputs → identical hashes
    assert shape_hash(Ga, net_node(10), k=1, include_pins=False) == \
           shape_hash(Gb, net_node(10), k=1, include_pins=False), \
        "pin-blind hashing should treat swapped commutative inputs as equal"

def test_net_side_always_includes_output_pin() -> None:
    module = Module(name="dff_multi_output")
    for bit_id in (1, 2, 3, 4, 5):
        module.get_net(bit_id)
    module.gates["ff0"] = Gate(
        name="ff0",
        cell_type="$_DFF_PN0_",
        inputs={"C": [1], "D": [2], "R": [3]},
        outputs={"Q": [4], "QN": [5]},
    )

    G = build_graph(module)
    q_hash = shape_hash(G, net_node(4), include_pins=False)
    qn_hash = shape_hash(G, net_node(5), include_pins=False)

    assert q_hash != qn_hash