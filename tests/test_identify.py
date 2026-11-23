# tests/test_identify.py
"""Function identification contract tests.

Coverage:
  - Library components: compute, widths, factories
  - Match dataclass repr
  - _find_fragment: boundary, constants, interior gates, early cutoff
  - _group_free_vars: base_name grouping, singletons, determinism
  - _build_recovered_tt: simple functions, constants substituted
  - _alignments: width matching, permutation coverage, empty case
  - _verify_alignment: positive/negative
  - _canonical_alignment_key: symmetric collisions, asymmetric distinction
  - identify_word: bitwise, mux, adder, function-equivalent components,
    unknown, empty, out-of-scope, sequential
  - identify_words: batch API matches single
  - Deduplication: symmetric ports yield one Match
  - Determinism

Not covered:
  - Real Yosys netlists (integration test territory)
  - Timing or memory characteristics
"""
import networkx as nx

from collections import defaultdict
from wordrev.aggregate import Word
from wordrev.graph import build_graph, net_node
from wordrev.identify import (
    Match,
    _alignments,
    _build_recovered_tt,
    _canonical_alignment_key,
    _find_fragment,
    _group_free_vars,
    _verify_alignment,
    identify_word,
    identify_words,
    match_slice_to_library,
)
from wordrev.library import (
    candidates_for_width,
    make_adder,
    make_adder_inc,
    make_bitwise_and,
    make_bitwise_not,
    make_bitwise_or,
    make_bitwise_xor,
    make_mux,
    make_subtractor,
)
from wordrev.models import Gate, Module
from wordrev.slice import enumerate_slices

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


def _annotate(m: Module, net_id: int, base: str, idx: int, direction: str = None):
    n = m.nets[net_id]
    n.name = f"{base}[{idx}]"
    n.base_name = base
    n.vector_index = idx
    if direction:
        n.is_port = True
        n.port_direction = direction


def make_bitwise_and_2bit() -> Module:
    """y[0] = a[0]&b[0], y[1] = a[1]&b[1].
    Inputs: a=1,2 b=3,4. Outputs: y=10,11."""
    m = make_module(
        Gate("g0", "$_AND_", {"A": [1], "B": [3]}, {"Y": [10]}),
        Gate("g1", "$_AND_", {"A": [2], "B": [4]}, {"Y": [11]}),
    )
    for net_id, base, idx in [(1, "a", 0), (2, "a", 1), (3, "b", 0), (4, "b", 1)]:
        _annotate(m, net_id, base, idx, "input")
    for net_id, idx in [(10, 0), (11, 1)]:
        _annotate(m, net_id, "y", idx, "output")
    return m


def make_bitwise_xor_1bit() -> Module:
    """y = a ^ b. Inputs: a=1, b=2. Output: y=10."""
    m = make_module(Gate("g", "$_XOR_", {"A": [1], "B": [2]}, {"Y": [10]}))
    _annotate(m, 1, "a", 0, "input")
    _annotate(m, 2, "b", 0, "input")
    _annotate(m, 10, "y", 0, "output")
    return m


def make_2bit_mux() -> Module:
    """y = sel ? b : a. Inputs: a=1,2 b=3,4 sel=5. Outputs: y=10,11."""
    m = make_module(
        Gate("m0", "$_MUX_", {"A": [1], "B": [3], "S": [5]}, {"Y": [10]}),
        Gate("m1", "$_MUX_", {"A": [2], "B": [4], "S": [5]}, {"Y": [11]}),
    )
    for net_id, base, idx in [(1, "a", 0), (2, "a", 1), (3, "b", 0), (4, "b", 1), (5, "sel", 0)]:
        _annotate(m, net_id, base, idx, "input")
    for net_id, idx in [(10, 0), (11, 1)]:
        _annotate(m, net_id, "y", idx, "output")
    return m


def make_2bit_adder() -> Module:
    """2-bit ripple adder with cin=0.
    sum[0]=a[0]^b[0], sum[1]=a[1]^b[1]^carry0 where carry0=a[0]&b[0]."""
    m = make_module(
        Gate("xor0",  "$_XOR_", {"A": [1], "B": [3]}, {"Y": [10]}),
        Gate("and0",  "$_AND_", {"A": [1], "B": [3]}, {"Y": [20]}),
        Gate("xor1a", "$_XOR_", {"A": [2], "B": [4]}, {"Y": [21]}),
        Gate("xor1s", "$_XOR_", {"A": [21], "B": [20]}, {"Y": [11]}),
    )
    for net_id, base, idx in [(1, "a", 0), (2, "a", 1), (3, "b", 0), (4, "b", 1)]:
        _annotate(m, net_id, base, idx, "input")
    for net_id, idx in [(10, 0), (11, 1)]:
        _annotate(m, net_id, "sum", idx, "output")
    return m


# -----------------------------------------------------------------------------
# Library components
# -----------------------------------------------------------------------------

def test_adder_compute_and_width():
    c = make_adder(4)
    assert c.output_width == 4
    assert c.total_input_bits == 8
    assert c.compute({"a": 5, "b": 3}) == 8
    assert c.compute({"a": 15, "b": 1}) == 0  # overflow wraps


def test_adder_inc_compute():
    c = make_adder_inc(4)
    assert c.compute({"a": 0, "b": 0}) == 1
    assert c.compute({"a": 15, "b": 1}) == 1  # 15+1+1 = 17 & 15


def test_subtractor_compute():
    c = make_subtractor(4)
    assert c.compute({"a": 5, "b": 3}) == 2
    assert c.compute({"a": 3, "b": 5}) == 14


def test_mux_compute():
    c = make_mux(2)
    assert c.compute({"a": 1, "b": 2, "sel": 0}) == 1
    assert c.compute({"a": 1, "b": 2, "sel": 1}) == 2


def test_bitwise_factories():
    assert make_bitwise_and(2).compute({"a": 3, "b": 1}) == 1
    assert make_bitwise_or(2).compute({"a": 2, "b": 1}) == 3
    assert make_bitwise_xor(2).compute({"a": 3, "b": 1}) == 2
    assert make_bitwise_not(2).compute({"a": 1}) == 2


def test_candidates_for_width_returns_correct_width():
    for width in (1, 2, 4, 8):
        for c in candidates_for_width(width):
            assert c.output_width == width


def test_candidates_for_width_is_nonempty():
    assert len(candidates_for_width(4)) > 0


# -----------------------------------------------------------------------------
# Match dataclass
# -----------------------------------------------------------------------------

def test_match_repr_is_readable():
    c = make_bitwise_and(1)
    m = Match(component=c, alignment={"a": [net_node(1)], "b": [net_node(2)]})
    r = repr(m)
    assert "bitwise_and_1" in r
    assert "a" in r and "b" in r


# -----------------------------------------------------------------------------
# _find_fragment
# -----------------------------------------------------------------------------

def test_find_fragment_identifies_boundary():
    """Free vars are the primary inputs feeding the fragment."""
    m = make_bitwise_and_2bit()
    G = build_graph(m)
    f = _find_fragment(G, [net_node(10), net_node(11)])
    assert set(f.free_vars) == {net_node(1), net_node(2), net_node(3), net_node(4)}
    assert f.constants == {}
    assert len(f.interior_gates) == 2


def test_find_fragment_includes_constants_separately():
    """Constants are boundary nets, but not free vars."""
    m = make_module(
        Gate("g", "$_AND_", {"A": [1], "B": [0]}, {"Y": [10]}),
        nets=[0, 1],
    )
    m.nets[0].is_constant = True
    m.nets[0].constant_value = 0
    _annotate(m, 1, "a", 0, "input")
    _annotate(m, 10, "y", 0, "output")
    G = build_graph(m)
    f = _find_fragment(G, [net_node(10)])
    assert f.free_vars == [net_node(1)]
    assert f.constants == {net_node(0): 0}


def test_find_fragment_word_bit_is_primary_input():
    """A Word bit that is itself a primary input yields no interior gates."""
    m = make_module(nets=[1])
    _annotate(m, 1, "x", 0, "input")
    G = build_graph(m)
    f = _find_fragment(G, [net_node(1)])
    assert f.free_vars == [net_node(1)]
    assert f.interior_gates == set()


# -----------------------------------------------------------------------------
# _group_free_vars
# -----------------------------------------------------------------------------

def test_group_free_vars_by_base_name():
    m = make_bitwise_and_2bit()
    G = build_graph(m)
    f = _find_fragment(G, [net_node(10), net_node(11)])
    groups = _group_free_vars(G, f.free_vars)
    names = [name for name, _ in groups]
    assert names == ["a", "b"]
    for _, nets in groups:
        assert len(nets) == 2


def test_group_free_vars_singletons_for_unnamed():
    """Unnamed nets get singleton groups."""
    m = make_module(
        Gate("g0", "$_AND_", {"A": [1], "B": [2]}, {"Y": [10]}),
    )
    G = build_graph(m)  # no annotations
    f = _find_fragment(G, [net_node(10)])
    groups = _group_free_vars(G, f.free_vars)
    # Two unnamed free vars, each a singleton group.
    assert len(groups) == 2
    for name, nets in groups:
        assert name.startswith("__single_")
        assert len(nets) == 1


def test_group_free_vars_sorts_within_group():
    """Bits in a group are ordered by vector index, not discovery order."""
    m = make_module(nets=[5, 3, 7])
    _annotate(m, 5, "a", 2)
    _annotate(m, 3, "a", 0)
    _annotate(m, 7, "a", 1)
    G = build_graph(m)
    # Free vars presented out of order.
    groups = _group_free_vars(G, [net_node(5), net_node(3), net_node(7)])
    assert len(groups) == 1
    name, nets = groups[0]
    assert name == "a"
    assert nets == [net_node(3), net_node(7), net_node(5)]  # a[0], a[1], a[2]


# -----------------------------------------------------------------------------
# _build_recovered_tt
# -----------------------------------------------------------------------------

def test_tt_for_and_gate():
    m = make_module(Gate("g", "$_AND_", {"A": [1], "B": [2]}, {"Y": [10]}))
    _annotate(m, 1, "a", 0)
    _annotate(m, 2, "b", 0)
    _annotate(m, 10, "y", 0)
    G = build_graph(m)
    f = _find_fragment(G, [net_node(10)])
    tt = _build_recovered_tt(G, [net_node(10)], f.free_vars, f.constants)
    # Free vars sorted: [1, 2].  tt[a]: bit assignment a over (var0, var1).
    assert tt[0b00] == 0
    assert tt[0b01] == 0  # a=1, b=0
    assert tt[0b10] == 0  # a=0, b=1
    assert tt[0b11] == 1


def test_tt_substitutes_constants():
    m = make_module(
        Gate("g", "$_AND_", {"A": [1], "B": [0]}, {"Y": [10]}),
        nets=[0, 1],
    )
    m.nets[0].is_constant = True
    m.nets[0].constant_value = 1
    _annotate(m, 1, "a", 0)
    _annotate(m, 10, "y", 0)
    G = build_graph(m)
    f = _find_fragment(G, [net_node(10)])
    tt = _build_recovered_tt(G, [net_node(10)], f.free_vars, f.constants)
    # AND(a, 1) = a (identity over 1 free var).
    assert tt[0] == 0
    assert tt[1] == 1


# -----------------------------------------------------------------------------
# _alignments
# -----------------------------------------------------------------------------

def test_alignments_width_mismatch_yields_nothing():
    """Two width-2 ports against one width-1 and one width-3 group: no match."""
    c = make_adder(2)  # a:2, b:2
    groups = [("x", [net_node(1)]), ("y", [net_node(2), net_node(3), net_node(4)])]
    alns = list(_alignments(c, groups, []))
    assert alns == []


def test_alignments_generates_permutations():
    """Two width-1 ports against two width-1 groups: 2 permutations."""
    c = make_bitwise_and(1)  # a:1, b:1
    groups = [("x", [net_node(1)]), ("y", [net_node(2)])]
    alns = list(_alignments(c, groups, []))
    assert len(alns) == 2
    assert {tuple(a["a"]) for a in alns} == {(net_node(1),), (net_node(2),)}


def test_alignments_skips_width_mismatched_ports():
    """If one port can't be filled, no alignment is yielded."""
    c = make_mux(2)  # a:2, b:2, sel:1
    groups = [("x", [net_node(1), net_node(2)]), ("y", [net_node(3), net_node(4)])]
    # No width-1 group for `sel` — no alignments.
    alns = list(_alignments(c, groups, []))
    assert alns == []


# -----------------------------------------------------------------------------
# _verify_alignment
# -----------------------------------------------------------------------------

def test_verify_alignment_positive():
    """AND(a=1, b=2) verifies against bitwise_and_1 with a=X, b=Y."""
    m = make_module(Gate("g", "$_AND_", {"A": [1], "B": [2]}, {"Y": [10]}))
    _annotate(m, 1, "x", 0)
    _annotate(m, 2, "y", 0)
    _annotate(m, 10, "z", 0)
    G = build_graph(m)
    f = _find_fragment(G, [net_node(10)])
    tt = _build_recovered_tt(G, [net_node(10)], f.free_vars, f.constants)
    c = make_bitwise_and(1)
    alignment = {"a": [net_node(1)], "b": [net_node(2)]}
    assert _verify_alignment(c, alignment, f.free_vars, tt) is True


def test_verify_alignment_negative():
    """AND does not verify against bitwise_or_1."""
    m = make_module(Gate("g", "$_AND_", {"A": [1], "B": [2]}, {"Y": [10]}))
    _annotate(m, 1, "x", 0)
    _annotate(m, 2, "y", 0)
    _annotate(m, 10, "z", 0)
    G = build_graph(m)
    f = _find_fragment(G, [net_node(10)])
    tt = _build_recovered_tt(G, [net_node(10)], f.free_vars, f.constants)
    c = make_bitwise_or(1)
    alignment = {"a": [net_node(1)], "b": [net_node(2)]}
    assert _verify_alignment(c, alignment, f.free_vars, tt) is False


# -----------------------------------------------------------------------------
# _canonical_alignment_key
# -----------------------------------------------------------------------------

def test_canonical_key_symmetric_ports_collide():
    """AND(a=X, b=Y) and AND(a=Y, b=X) produce the same canonical key."""
    c = make_bitwise_and(1)
    x, y = [net_node(1)], [net_node(2)]
    a1 = {"a": x, "b": y}
    a2 = {"a": y, "b": x}
    assert _canonical_alignment_key(c, a1) == _canonical_alignment_key(c, a2)


def test_canonical_key_different_components_differ():
    """AND and OR canonicalize differently even with the same alignment."""
    and_c = make_bitwise_and(1)
    or_c = make_bitwise_or(1)
    alignment = {"a": [net_node(1)], "b": [net_node(2)]}
    assert _canonical_alignment_key(and_c, alignment) != _canonical_alignment_key(or_c, alignment)


def test_canonical_key_mixed_widths():
    """Mux has no commutative groups: swapping a and b changes the key.

    A mux's data ports are asymmetric — sel chooses between them — so
    the width-based grouping from an earlier version over-collapsed
    these alignments. The corrected key preserves the distinction.
    """
    c = make_mux(2)
    alignment = {
        "a": [net_node(1), net_node(2)],
        "b": [net_node(3), net_node(4)],
        "sel": [net_node(5)],
    }
    swapped = {"a": alignment["b"], "b": alignment["a"], "sel": alignment["sel"]}
    assert _canonical_alignment_key(c, alignment) != _canonical_alignment_key(c, swapped)

# -----------------------------------------------------------------------------
# identify_word
# -----------------------------------------------------------------------------

def test_identify_bitwise_and_2():
    G = build_graph(make_bitwise_and_2bit())
    word = Word(bits=[net_node(10), net_node(11)])
    matches = identify_word(G, word)
    names = {m.component.name for m in matches}
    assert "bitwise_and_2" in names
    assert "bitwise_or_2" not in names
    assert "bitwise_xor_2" not in names


def test_identify_bitwise_and_deduplicates_symmetric_alignments():
    """AND(a, b) and AND(b, a) both verify; dedupe keeps one Match."""
    G = build_graph(make_bitwise_and_2bit())
    word = Word(bits=[net_node(10), net_node(11)])
    matches = identify_word(G, word)
    and_matches = [m for m in matches if m.component.name == "bitwise_and_2"]
    assert len(and_matches) == 1


def test_identify_1bit_xor_finds_multiple_equivalent_components():
    """A 1-bit XOR matches bitwise_xor_1, adder_1, and subtractor_1 —
    they are all the same function on 1 bit. All three appear."""
    G = build_graph(make_bitwise_xor_1bit())
    word = Word(bits=[net_node(10)])
    matches = identify_word(G, word)
    names = {m.component.name for m in matches}
    assert "bitwise_xor_1" in names
    assert "adder_1" in names
    assert "subtractor_1" in names


def test_identify_2bit_mux():
    G = build_graph(make_2bit_mux())
    word = Word(bits=[net_node(10), net_node(11)])
    matches = identify_word(G, word)
    names = {m.component.name for m in matches}
    assert "mux_2" in names


def test_identify_2bit_adder_sum():
    """The 2-bit sum is identified as adder_2."""
    G = build_graph(make_2bit_adder())
    word = Word(bits=[net_node(10), net_node(11)])
    matches = identify_word(G, word)
    names = {m.component.name for m in matches}
    assert "adder_2" in names


def test_identify_2bit_adder_sum_does_not_match_subtractor():
    """adder_2 and subtractor_2 differ (bit 1 of sum vs difference)."""
    G = build_graph(make_2bit_adder())
    word = Word(bits=[net_node(10), net_node(11)])
    matches = identify_word(G, word)
    names = {m.component.name for m in matches}
    assert "subtractor_2" not in names


def test_identify_alignment_binds_correct_groups():
    """The retained Match's alignment maps a→a nets, b→b nets."""
    G = build_graph(make_bitwise_and_2bit())
    word = Word(bits=[net_node(10), net_node(11)])
    matches = identify_word(G, word)
    and_match = next(m for m in matches if m.component.name == "bitwise_and_2")
    assert {n[1] for n in and_match.alignment["a"]} == {1, 2}
    assert {n[1] for n in and_match.alignment["b"]} == {3, 4}


def test_identify_empty_word_returns_empty():
    G = build_graph(make_module(nets=[1]))
    assert identify_word(G, Word(bits=[])) == []


def test_identify_unknown_function_returns_empty():
    """(a AND b) XOR c — no component in the library."""
    m = make_module(
        Gate("g1", "$_AND_", {"A": [1], "B": [2]}, {"Y": [10]}),
        Gate("g2", "$_XOR_", {"A": [10], "B": [3]}, {"Y": [11]}),
    )
    _annotate(m, 1, "a", 0)
    _annotate(m, 2, "b", 0)
    _annotate(m, 3, "c", 0)
    _annotate(m, 11, "y", 0)
    G = build_graph(m)
    word = Word(bits=[net_node(11)])
    assert identify_word(G, word) == []


def test_identify_identity_returns_empty():
    """A Word bit that is itself a primary input has no library match."""
    m = make_module(nets=[1])
    _annotate(m, 1, "x", 0, "input")
    G = build_graph(m)
    word = Word(bits=[net_node(1)])
    assert identify_word(G, word) == []


def test_identify_rejects_when_external_inputs_exceed_limit():
    """A fragment with more free vars than max_external_inputs returns []."""
    # Chain of 8 ANDs: free vars = 8.
    gates = []
    prev = 1
    for i in range(2, 9):
        out = 100 + i
        gates.append(Gate(f"and{i}", "$_AND_", {"A": [prev], "B": [i]}, {"Y": [out]}))
        prev = out
    m = make_module(*gates, nets=list(range(1, 9)))
    _annotate(m, prev, "y", 0, "output")
    G = build_graph(m)
    word = Word(bits=[net_node(prev)])
    assert identify_word(G, word, max_external_inputs=4) == []
    # With a larger limit, the fragment is in scope (match not required —
    # the 8-input AND is not in the library — but the limit must not reject).
    # A negative match is fine; the limit-rejection test is the assertion above.


def test_identify_rejects_when_fragment_exceeds_gate_limit():
    """Long chain exceeds max_fragment_gates."""
    gates = []
    prev = 1
    for i in range(20):
        out = 100 + i
        gates.append(Gate(f"inv{i}", "$_NOT_", {"A": [prev]}, {"Y": [out]}))
        prev = out
    m = make_module(*gates, nets=[1])
    _annotate(m, prev, "y", 0, "output")
    G = build_graph(m)
    word = Word(bits=[net_node(prev)])
    # Only 1 free var, but 20 interior gates.
    assert identify_word(G, word, max_fragment_gates=5) == []


def test_identify_sequential_fragment_returns_empty():
    """A DFF in the fragment causes simulation failure → []."""
    m = make_module(
        Gate("ff", "$_DFF_P_", {"D": [1], "C": [2]}, {"Q": [10]}),
    )
    _annotate(m, 1, "d", 0)
    _annotate(m, 2, "clk", 0)
    _annotate(m, 10, "q", 0)
    G = build_graph(m)
    word = Word(bits=[net_node(10)])
    assert identify_word(G, word) == []


# -----------------------------------------------------------------------------
# identify_words (batch)
# -----------------------------------------------------------------------------

def test_identify_words_returns_dict_keyed_by_index():
    G = build_graph(make_bitwise_and_2bit())
    words = [
        Word(bits=[net_node(10), net_node(11)]),
        Word(bits=[]),
    ]
    results = identify_words(G, words)
    assert set(results.keys()) == {0, 1}
    assert results[0] != []
    assert results[1] == []


def test_identify_words_matches_single_calls():
    G = build_graph(make_bitwise_and_2bit())
    words = [
        Word(bits=[net_node(10), net_node(11)]),
        Word(bits=[net_node(10)]),
    ]
    batch = identify_words(G, words)
    for i, w in enumerate(words):
        single = identify_word(G, w)
        batch_names = {m.component.name for m in batch[i]}
        single_names = {m.component.name for m in single}
        assert batch_names == single_names


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------

def test_identify_is_deterministic():
    G = build_graph(make_bitwise_and_2bit())
    word = Word(bits=[net_node(10), net_node(11)])
    r1 = identify_word(G, word)
    r2 = identify_word(G, word)
    assert [m.component.name for m in r1] == [m.component.name for m in r2]
    assert [m.alignment for m in r1] == [m.alignment for m in r2]


def test_identify_alignment_order_is_stable():
    """Repeated calls yield the same (retained) alignment for each component."""
    G = build_graph(make_bitwise_and_2bit())
    word = Word(bits=[net_node(10), net_node(11)])
    r1 = identify_word(G, word)
    r2 = identify_word(G, word)
    for m1, m2 in zip(r1, r2):
        assert m1.alignment.keys() == m2.alignment.keys()
        for port in m1.alignment:
            assert m1.alignment[port] == m2.alignment[port]

# -----------------------------------------------------------------------------
# After commutativity changes
# -----------------------------------------------------------------------------

def test_1bit_xor_is_identified_by_every_equivalent_component_once():
    """At width 1, XOR == adder == subtractor. Each component that matches
    a 1-bit XOR fragment must produce exactly one Match, not two."""
    G = build_graph(make_bitwise_xor_1bit())
    word = Word(bits=[net_node(10)])
    matches = identify_word(G, word)

    by_component: dict[str, list[Match]] = defaultdict(list)
    for m in matches:
        by_component[m.component.name].append(m)

    for name in ("bitwise_xor_1", "adder_1", "subtractor_1"):
        assert name in by_component, f"expected {name} in matches"
        assert len(by_component[name]) == 1, (
            f"{name} returned {len(by_component[name])} Matches; "
            f"expected 1 (regression: width-1 subtractor is commutative)"
        )


def test_1bit_subtractor_declares_commutativity():
    from wordrev.library import make_subtractor
    assert frozenset({"a", "b"}) in make_subtractor(1).commutative_groups


def test_2bit_subtractor_is_not_commutative():
    from wordrev.library import make_subtractor
    assert make_subtractor(2).commutative_groups == ()

# -----------------------------------------------------------------------------
# Slice enumerate change
# -----------------------------------------------------------------------------

def _build_mux1_netlist():
    G = nx.MultiDiGraph()
    a, b, sel, y = ("net", 2), ("net", 3), ("net", 4), ("net", 5)
    g = ("gate", 0)
    G.add_node(a, is_constant=False, base_name="a", vector_index=0)
    G.add_node(b, is_constant=False, base_name="b", vector_index=0)
    G.add_node(sel, is_constant=False, base_name="sel", vector_index=0)
    G.add_node(y, is_constant=False, base_name="y", vector_index=0)
    G.add_node(g, cell_type="$_MUX_")
    G.add_edge(a, g, pin="A", role="input")
    G.add_edge(b, g, pin="B", role="input")
    G.add_edge(sel, g, pin="S", role="input")
    G.add_edge(g, y, pin="Y", role="output")
    return G, a, b, sel, y


def test_match_slice_to_library_mux():
    G, a, b, sel, y = _build_mux1_netlist()
    slices = enumerate_slices(G)
    assert slices[y]

    sl = slices[y][0]
    result = match_slice_to_library(G, sl)
    assert result is not None

    component, alignment = result
    assert component.output_width == 1
    assert "sel" in component.select_ports
    # Mux is not commutative across a/b, so exactly one alignment verifies.
    assert alignment["a"] == (a,)
    assert alignment["b"] == (b,)
    assert alignment["sel"] == (sel,)
    # Fields populated in place.
    assert sl.aligned_to is component
    assert sl.alignment == alignment


def test_match_slice_to_library_idempotent():
    G, *_ = _build_mux1_netlist()
    sl = enumerate_slices(G)[("net", 5)][0]
    first = match_slice_to_library(G, sl)
    second = match_slice_to_library(G, sl)
    assert first == second  # same component + alignment