# tests/test_aggregate.py
"""Word aggregation contract tests.

Each public and non-trivial private function is tested in its own section.
Helpers build minimal modules with a known structure; no fixture files.

Coverage:
  - Word dataclass
  - _net_sort_key (bit-ID fallback ordering)
  - _bit_sort_key (vector-index ordering, bus grouping, mixed fallback)
  - _cluster_candidates (identical gates, split signatures, min_width)
  - _cone_sizes (monotonicity, cap enforcement)
  - _structural_order (cone-size ordering, vector-index tiebreak)
  - _carry_chain_order (dependent bits, independent bits, singleton)
  - order_bits (returns kind, dispatches correctly)
  - propagate_words (parallel gates, mismatched pins, mismatched cells,
    ambiguity rejection, confidence decay, multi-round, deduplication)
  - resolve_overlaps (confidence, width tiebreak, disjointness)
  - aggregate_words (empty, no propagation, precomputed inputs, adder,
    disjointness, source/confidence well-formedness)
"""

from wordrev.aggregate import (
    Word,
    _bit_sort_key,
    _carry_chain_order,
    _cluster_candidates,
    _cone_sizes,
    _net_sort_key,
    _structural_order,
    aggregate_words,
    order_bits,
    propagate_words,
    resolve_overlaps,
)
from wordrev.graph import build_graph, net_node
from wordrev.models import Gate, Module
from wordrev.shape import compute_all_shape_hashes
from wordrev.slice import compute_functional_signatures


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


def _set_vector(net, base: str, idx: int) -> None:
    """Simulate parser output: assign a vector name to a Net."""
    net.name = f"{base}[{idx}]"
    net.base_name = base
    net.vector_index = idx


def ripple_adder_module() -> Module:
    """3-bit ripple-carry adder with cin tied to 0.

    Nets: a0=1 b0=2 a1=3 b1=4 a2=5 b2=6
          sum[0..2]        = 10, 11, 12
          intermediate XOR = 20, 30
          carries          = 100, 101
    """
    return make_module(
        Gate("xor0",  "$_XOR_", {"A": [1], "B": [2]}, {"Y": [10]}),
        Gate("and0",  "$_AND_", {"A": [1], "B": [2]}, {"Y": [100]}),
        Gate("xor1a", "$_XOR_", {"A": [3], "B": [4]}, {"Y": [20]}),
        Gate("xor1b", "$_XOR_", {"A": [20], "B": [100]}, {"Y": [11]}),
        Gate("and1",  "$_AND_", {"A": [20], "B": [100]}, {"Y": [101]}),
        Gate("xor2a", "$_XOR_", {"A": [5], "B": [6]}, {"Y": [30]}),
        Gate("xor2b", "$_XOR_", {"A": [30], "B": [101]}, {"Y": [12]}),
    )


# -----------------------------------------------------------------------------
# Word dataclass
# -----------------------------------------------------------------------------

def test_word_width():
    assert Word(bits=[net_node(i) for i in (1, 2, 3)]).width == 3
    assert Word(bits=[net_node(1)]).width == 1
    assert Word(bits=[]).width == 0


def test_word_defaults():
    w = Word(bits=[net_node(1)])
    assert w.confidence == 1.0
    assert w.source == "cluster_carry"
    assert w.shape_hash is None
    assert w.signature is None


# -----------------------------------------------------------------------------
# _net_sort_key  (bit-ID fallback)
# -----------------------------------------------------------------------------

def test_net_sort_key_integer_order():
    nodes = [net_node(10), net_node(2), net_node(1)]
    assert sorted(nodes, key=_net_sort_key) == [
        net_node(1), net_node(2), net_node(10),
    ]


def test_net_sort_key_mixed_types():
    """String IDs sort after all integers, lexicographically."""
    nodes = [net_node("x"), net_node(5), net_node("a")]
    assert sorted(nodes, key=_net_sort_key) == [
        net_node(5), net_node("a"), net_node("x"),
    ]


# -----------------------------------------------------------------------------
# _bit_sort_key  (semantic ordering: vector index > bit ID)
# -----------------------------------------------------------------------------

def test_bit_sort_key_falls_back_without_vector_fields():
    """When the parser has not populated base_name/vector_index, ordering
    falls through to bit ID."""
    m = make_module(nets=[1, 2, 10])
    G = build_graph(m)
    # No vector fields set on any Net
    assert _bit_sort_key(net_node(2), G) == (1, "", _net_sort_key(net_node(2)))
    assert _bit_sort_key(net_node(10), G) == (1, "", _net_sort_key(net_node(10)))


def test_bit_sort_key_orders_by_vector_index():
    """When vector fields are present, index determines order regardless
    of bit ID."""
    m = make_module(nets=[1, 2])
    _set_vector(m.nets[1], "a", 3)
    _set_vector(m.nets[2], "a", 0)
    G = build_graph(m)
    # a[0] (net 2) sorts before a[3] (net 1), even though net 1 < net 2.
    assert _bit_sort_key(net_node(2), G) < _bit_sort_key(net_node(1), G)


def test_bit_sort_key_groups_by_base_name():
    """Bits of the same bus are adjacent; bits of different buses are not."""
    m = make_module(nets=[1, 2, 3, 4])
    _set_vector(m.nets[1], "a", 1)
    _set_vector(m.nets[2], "b", 0)
    _set_vector(m.nets[3], "a", 0)
    _set_vector(m.nets[4], "b", 1)
    G = build_graph(m)
    ordered = sorted(
        [net_node(1), net_node(2), net_node(3), net_node(4)],
        key=lambda n: _bit_sort_key(n, G),
    )
    # Bus `a` first (index 0 then 1), then bus `b` (0 then 1).
    assert ordered == [net_node(3), net_node(1), net_node(2), net_node(4)]


# -----------------------------------------------------------------------------
# _cluster_candidates
# -----------------------------------------------------------------------------

def test_cluster_two_identical_and_gates():
    m = make_module(
        Gate("g1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [10]}),
        Gate("g2", "$_AND_", {"A": [4], "B": [5]}, {"Y": [11]}),
    )
    G = build_graph(m)
    hashes = compute_all_shape_hashes(G, k=2)
    sigs = compute_functional_signatures(G)
    clusters = _cluster_candidates(hashes, sigs, min_width=2)
    assert len(clusters) == 1
    members, shape, sig = clusters[0]
    assert members == frozenset({net_node(10), net_node(11)})
    assert sig == (2, 0b1000)


def test_cluster_different_signatures_do_not_cluster():
    m = make_module(
        Gate("g1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [10]}),
        Gate("g2", "$_OR_",  {"A": [4], "B": [5]}, {"Y": [11]}),
    )
    G = build_graph(m)
    hashes = compute_all_shape_hashes(G, k=2)
    sigs = compute_functional_signatures(G)
    assert _cluster_candidates(hashes, sigs, min_width=2) == []


def test_cluster_respects_min_width():
    m = make_module(Gate("g", "$_AND_", {"A": [2], "B": [3]}, {"Y": [10]}))
    G = build_graph(m)
    hashes = compute_all_shape_hashes(G, k=2)
    sigs = compute_functional_signatures(G)
    assert _cluster_candidates(hashes, sigs, min_width=2) == []
    assert len(_cluster_candidates(hashes, sigs, min_width=1)) == 1


def test_cluster_one_per_qualifying_signature():
    """Four gates in one shape bucket, split 2 AND + 2 OR, must yield
    two clusters — one per signature."""
    m = make_module(
        Gate("a1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [10]}),
        Gate("a2", "$_AND_", {"A": [4], "B": [5]}, {"Y": [11]}),
        Gate("o1", "$_OR_",  {"A": [6], "B": [7]}, {"Y": [12]}),
        Gate("o2", "$_OR_",  {"A": [8], "B": [9]}, {"Y": [13]}),
    )
    G = build_graph(m)
    hashes = compute_all_shape_hashes(G, k=2)
    sigs = compute_functional_signatures(G)
    clusters = _cluster_candidates(hashes, sigs, min_width=2)
    by_sig = {sig: members for members, _, sig in clusters}
    assert by_sig[(2, 0b1000)] == frozenset({net_node(10), net_node(11)})
    assert by_sig[(2, 0b1110)] == frozenset({net_node(12), net_node(13)})


# -----------------------------------------------------------------------------
# _cone_sizes
# -----------------------------------------------------------------------------

def test_cone_sizes_increase_with_depth():
    m = make_module(
        Gate("inv", "$_NOT_", {"A": [2]}, {"Y": [3]}),
        Gate("buf", "$_BUF_", {"A": [3]}, {"Y": [4]}),
    )
    G = build_graph(m)
    sizes = _cone_sizes(G, [net_node(3), net_node(4)])
    assert sizes[net_node(3)] < sizes[net_node(4)]


def test_cone_sizes_capped():
    gates = []
    prev = 2
    for i in range(20):
        out = 100 + i
        gates.append(Gate(f"inv{i}", "$_NOT_", {"A": [prev]}, {"Y": [out]}))
        prev = out
    m = make_module(*gates)
    G = build_graph(m)
    sizes = _cone_sizes(G, [net_node(prev)], cap=10)
    assert sizes[net_node(prev)] <= 10


# -----------------------------------------------------------------------------
# _structural_order
# -----------------------------------------------------------------------------

def test_structural_order_by_cone_size():
    m = make_module(
        Gate("inv", "$_NOT_", {"A": [2]}, {"Y": [3]}),
        Gate("buf", "$_BUF_", {"A": [3]}, {"Y": [4]}),
    )
    G = build_graph(m)
    cluster = frozenset({net_node(3), net_node(4)})
    sizes = _cone_sizes(G, list(cluster))
    ordered = _structural_order(G, cluster, sizes)
    assert ordered == [net_node(3), net_node(4)]


def test_structural_order_uses_vector_index_as_tiebreak():
    """When cone sizes tie, vector index orders — not bit ID."""
    m = make_module(nets=[1, 2])
    _set_vector(m.nets[1], "a", 3)
    _set_vector(m.nets[2], "a", 0)
    G = build_graph(m)
    cluster = frozenset({net_node(1), net_node(2)})
    # Give both equal cone sizes to force the tiebreak.
    sizes = {net_node(1): 5, net_node(2): 5}
    ordered = _structural_order(G, cluster, sizes)
    assert ordered == [net_node(2), net_node(1)]  # a[0] before a[3]


# -----------------------------------------------------------------------------
# _carry_chain_order
# -----------------------------------------------------------------------------

def test_carry_chain_orders_dependent_bits():
    m = make_module(
        Gate("and0", "$_AND_", {"A": [2], "B": [3]}, {"Y": [100]}),
        Gate("and1", "$_AND_", {"A": [4], "B": [100]}, {"Y": [101]}),
    )
    G = build_graph(m)
    cluster = frozenset({net_node(100), net_node(101)})
    ordered = _carry_chain_order(G, cluster)
    assert ordered == [net_node(100), net_node(101)]


def test_carry_chain_returns_none_for_independent_bits():
    """Parallel sum bits are not in each other's cones — no chain."""
    m = ripple_adder_module()
    G = build_graph(m)
    cluster = frozenset({net_node(11), net_node(12)})
    assert _carry_chain_order(G, cluster) is None


def test_carry_chain_returns_none_for_singleton():
    m = make_module(Gate("g", "$_AND_", {"A": [2], "B": [3]}, {"Y": [10]}))
    G = build_graph(m)
    assert _carry_chain_order(G, frozenset({net_node(10)})) is None


def test_carry_chain_ties_broken_by_vector_index():
    """Two bits with equal carry depth in a chain: vector index breaks the tie."""
    # Net 10 feeds both inv0 (-> net 100) and inv1 (-> net 101).
    # In cluster {100, 101, 10}:
    #   net 10 has carry depth 0 (in cones of 100 and 101).
    #   net 100 and 101 both have carry depth 1 (depend on 10, but not on each other).
    # Nets 100 and 101 tie in carry depth (1), so vector_index decides their order!
    m = make_module(
        Gate("inv0", "$_NOT_", {"A": [10]}, {"Y": [100]}),
        Gate("inv1", "$_NOT_", {"A": [10]}, {"Y": [101]}),
    )
    _set_vector(m.nets[100], "x", 5)
    _set_vector(m.nets[101], "x", 2)
    _set_vector(m.nets[10],  "x", 0)
    
    G = build_graph(m)
    cluster = frozenset({net_node(10), net_node(100), net_node(101)})
    
    ordered = _carry_chain_order(G, cluster)
    # LSB is 10 (depth 0). For depth 1, x[2] (net 101) must sort before x[5] (net 100).
    assert ordered == [net_node(10), net_node(101), net_node(100)]


# -----------------------------------------------------------------------------
# order_bits
# -----------------------------------------------------------------------------

def test_order_bits_returns_carry_kind():
    m = make_module(
        Gate("and0", "$_AND_", {"A": [2], "B": [3]}, {"Y": [100]}),
        Gate("and1", "$_AND_", {"A": [4], "B": [100]}, {"Y": [101]}),
    )
    G = build_graph(m)
    cluster = frozenset({net_node(100), net_node(101)})
    ordered, kind = order_bits(G, cluster)
    assert kind == "carry"
    assert ordered == [net_node(100), net_node(101)]


def test_order_bits_returns_structural_kind():
    m = make_module(
        Gate("g1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [10]}),
        Gate("g2", "$_AND_", {"A": [4], "B": [5]}, {"Y": [11]}),
    )
    G = build_graph(m)
    cluster = frozenset({net_node(10), net_node(11)})
    _, kind = order_bits(G, cluster)
    assert kind == "structural"


def test_order_bits_structural_path_uses_vector_index():
    """Two parallel AND gates with no chain; vector index orders the result."""
    m = make_module(
        Gate("g1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [10]}),
        Gate("g2", "$_AND_", {"A": [4], "B": [5]}, {"Y": [11]}),
    )
    _set_vector(m.nets[10], "q", 1)
    _set_vector(m.nets[11], "q", 0)
    G = build_graph(m)
    cluster = frozenset({net_node(10), net_node(11)})
    ordered, kind = order_bits(G, cluster)
    assert kind == "structural"
    assert ordered == [net_node(11), net_node(10)]  # q[0] before q[1]


# -----------------------------------------------------------------------------
# propagate_words
# -----------------------------------------------------------------------------

def test_propagate_through_parallel_buffers():
    m = make_module(
        Gate("buf0", "$_BUF_", {"A": [1]}, {"Y": [10]}),
        Gate("buf1", "$_BUF_", {"A": [2]}, {"Y": [11]}),
    )
    G = build_graph(m)
    result = propagate_words(G, [Word(bits=[net_node(1), net_node(2)])])
    propagated = [w for w in result if w.source == "propagated"]
    assert len(propagated) == 1
    assert propagated[0].bits == [net_node(10), net_node(11)]


def test_propagate_decays_confidence():
    m = make_module(
        Gate("buf0", "$_BUF_", {"A": [1]}, {"Y": [10]}),
        Gate("buf1", "$_BUF_", {"A": [2]}, {"Y": [11]}),
    )
    G = build_graph(m)
    result = propagate_words(G, [Word(bits=[net_node(1), net_node(2)], confidence=1.0)])
    propagated = [w for w in result if w.source == "propagated"]
    assert abs(propagated[0].confidence - 0.9) < 1e-9


def test_propagate_stops_at_mismatched_pins():
    """Bits feeding different pins of the same gate do not propagate."""
    m = make_module(
        Gate("xor", "$_XOR_", {"A": [1], "B": [2]}, {"Y": [20]}),
    )
    G = build_graph(m)
    result = propagate_words(G, [Word(bits=[net_node(1), net_node(2)])])
    assert all(w.source != "propagated" for w in result)


def test_propagate_stops_at_mismatched_cell_types():
    m = make_module(
        Gate("buf", "$_BUF_", {"A": [1]}, {"Y": [10]}),
        Gate("inv", "$_NOT_", {"A": [2]}, {"Y": [11]}),
    )
    G = build_graph(m)
    result = propagate_words(G, [Word(bits=[net_node(1), net_node(2)])])
    assert all(w.source != "propagated" for w in result)


def test_propagate_rejects_ambiguous_matches():
    """A bit feeding two same-type gates on the same pin is ambiguous —
    propagation is rejected rather than picking arbitrarily."""
    m = make_module(
        Gate("b0a", "$_BUF_", {"A": [1]}, {"Y": [10]}),
        Gate("b0b", "$_BUF_", {"A": [1]}, {"Y": [11]}),  # bit 1 fans to two
        Gate("b1",  "$_BUF_", {"A": [2]}, {"Y": [20]}),
    )
    G = build_graph(m)
    result = propagate_words(G, [Word(bits=[net_node(1), net_node(2)])])
    assert all(w.source != "propagated" for w in result)


def test_propagate_multiple_rounds():
    m = make_module(
        Gate("buf0a", "$_BUF_", {"A": [1]}, {"Y": [10]}),
        Gate("buf0b", "$_BUF_", {"A": [10]}, {"Y": [20]}),
        Gate("buf1a", "$_BUF_", {"A": [2]}, {"Y": [11]}),
        Gate("buf1b", "$_BUF_", {"A": [11]}, {"Y": [21]}),
    )
    G = build_graph(m)
    result = propagate_words(G, [Word(bits=[net_node(1), net_node(2)])])
    bit_sets = {frozenset(w.bits) for w in result}
    assert frozenset({net_node(10), net_node(11)}) in bit_sets
    assert frozenset({net_node(20), net_node(21)}) in bit_sets


def test_propagate_deduplicates():
    m = make_module(
        Gate("buf0", "$_BUF_", {"A": [1]}, {"Y": [10]}),
        Gate("buf1", "$_BUF_", {"A": [2]}, {"Y": [11]}),
    )
    G = build_graph(m)
    result = propagate_words(G, [
        Word(bits=[net_node(1), net_node(2)]),
        Word(bits=[net_node(1), net_node(2)]),  # duplicate
    ])
    propagated = [w for w in result if w.source == "propagated"]
    assert len(propagated) == 1


def test_propagate_empty_word_is_noop():
    m = make_module(Gate("g", "$_BUF_", {"A": [1]}, {"Y": [10]}))
    G = build_graph(m)
    result = propagate_words(G, [Word(bits=[])])
    assert len(result) == 1
    assert result[0].bits == []


# -----------------------------------------------------------------------------
# resolve_overlaps
# -----------------------------------------------------------------------------

def test_resolve_prefers_higher_confidence():
    w_hi = Word(bits=[net_node(1), net_node(2)], confidence=1.0)
    w_lo = Word(bits=[net_node(2), net_node(3)], confidence=0.5)
    accepted = resolve_overlaps([w_lo, w_hi])
    assert len(accepted) == 1
    assert accepted[0] is w_hi


def test_resolve_tiebreaks_by_width():
    w_narrow = Word(bits=[net_node(1), net_node(2)], confidence=0.5)
    w_wide   = Word(bits=[net_node(2), net_node(3), net_node(4)], confidence=0.5)
    accepted = resolve_overlaps([w_narrow, w_wide])
    assert len(accepted) == 1
    assert accepted[0] is w_wide


def test_resolve_keeps_disjoint():
    w1 = Word(bits=[net_node(1), net_node(2)])
    w2 = Word(bits=[net_node(3), net_node(4)])
    assert len(resolve_overlaps([w1, w2])) == 2


def test_resolve_empty_input():
    assert resolve_overlaps([]) == []


# -----------------------------------------------------------------------------
# aggregate_words
# -----------------------------------------------------------------------------

def test_aggregate_empty_graph():
    m = make_module(nets=[1, 2])
    G = build_graph(m)
    assert aggregate_words(G) == []


def test_aggregate_no_propagation():
    m = make_module(
        Gate("buf0",  "$_BUF_", {"A": [1]},  {"Y": [10]}),
        Gate("buf1",  "$_BUF_", {"A": [2]},  {"Y": [11]}),
        Gate("out0",  "$_BUF_", {"A": [10]}, {"Y": [20]}),
        Gate("out1",  "$_BUF_", {"A": [11]}, {"Y": [21]}),
    )
    G = build_graph(m)
    words = aggregate_words(G, min_width=1, propagate=False)
    assert all(w.source != "propagated" for w in words)


def test_aggregate_uses_precomputed_inputs():
    """Passing precomputed shape_hashes and signatures yields the same result."""
    m = make_module(
        Gate("g1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [10]}),
        Gate("g2", "$_AND_", {"A": [4], "B": [5]}, {"Y": [11]}),
    )
    G = build_graph(m)
    hashes = compute_all_shape_hashes(G, k=2)
    sigs = compute_functional_signatures(G)
    auto = aggregate_words(G, shape_k=2)
    pre  = aggregate_words(G, shape_hashes=hashes, signatures=sigs)
    assert {tuple(w.bits) for w in auto} == {tuple(w.bits) for w in pre}


def test_aggregate_ripple_adder_recovers_words():
    G = build_graph(ripple_adder_module())
    words = aggregate_words(G, shape_k=2, min_width=2)
    assert len(words) >= 1
    assert all(w.width >= 2 for w in words)


def test_aggregate_words_are_disjoint():
    G = build_graph(ripple_adder_module())
    words = aggregate_words(G, shape_k=2, min_width=2)
    claimed: set = set()
    for w in words:
        bits = set(w.bits)
        assert not (bits & claimed), f"overlap: {bits} with {claimed}"
        claimed |= bits


def test_aggregate_source_and_confidence_well_formed():
    G = build_graph(ripple_adder_module())
    words = aggregate_words(G, shape_k=2, min_width=2)
    for w in words:
        assert w.source in ("cluster_carry", "cluster_structural", "propagated")
        assert 0.0 < w.confidence <= 1.0


def test_aggregate_carry_ordered_word_has_full_confidence():
    """Regression: confidence must reflect ordering kind. A word ordered
    by carry chain must get 1.0 and source 'cluster_carry'."""
    G = build_graph(ripple_adder_module())
    words = aggregate_words(G, shape_k=2, min_width=2)
    carry_words = [w for w in words if w.source == "cluster_carry"]
    for w in carry_words:
        assert w.confidence == 1.0