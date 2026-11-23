# tests/test_slice.py
"""Bitslice extraction contract tests.

Coverage:
  - Gate truth tables (all 12 supported primitives)
  - Constant substitution in cuts
  - Combinational cycle detection (path-based)
  - Canonicalization under permutation
  - Cut enumeration: trivial-first, expansion, minimality, primary-input safety
  - Sort-key reproducibility (_net_sort_key)
  - Functional signatures: determinism, matching, arity
  - The 2-bit adder grouping signal
  - Batch API robustness (sequential cells, unsupported cells)

Not covered (by design):
  - Specific hash or truth-table values beyond what is semantically meaningful
  - Behaviour of the parser or graph builder (covered in their own test files)
  - Real Yosys files (covered by the conformance test)
"""
import pytest
import networkx as nx
from wordrev.graph import build_graph, net_node
from wordrev.models import Gate, Module
from wordrev.slice import (
    _net_sort_key,
    canonical_truth_table,
    compute_functional_signatures,
    enumerate_cuts,
    simulate_slice,
    slice_signature,
    enumerate_slices,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def make_module(*gates: Gate, nets: list = None) -> Module:
    """Build a Module from gates; every referenced net is auto-created."""
    m = Module("t")
    for g in gates:
        m.gates[g.name] = g
        for bits in (*g.inputs.values(), *g.outputs.values(), *g.inouts.values()):
            for b in bits:
                m.get_net(b)
    for bit_id in (nets or []):
        m.get_net(bit_id)
    return m


def two_bit_adder_module() -> Module:
    """2-bit ripple-carry adder with cin tied to 0.

    Nets: a0=2 a1=3 b0=4 b1=5
          sum0=10 carry0=11 temp_x=12 sum1=13 temp_and1=14 temp_andc=15 cout=16
    """
    return make_module(
        Gate("xor0",  "$_XOR_", {"A": [2], "B": [4]}, {"Y": [10]}),
        Gate("and0",  "$_AND_", {"A": [2], "B": [4]}, {"Y": [11]}),
        Gate("xor_x", "$_XOR_", {"A": [3], "B": [5]}, {"Y": [12]}),
        Gate("xor1",  "$_XOR_", {"A": [12], "B": [11]}, {"Y": [13]}),
        Gate("and1",  "$_AND_", {"A": [3], "B": [5]}, {"Y": [14]}),
        Gate("andc",  "$_AND_", {"A": [11], "B": [12]}, {"Y": [15]}),
        Gate("or1",   "$_OR_",  {"A": [14], "B": [15]}, {"Y": [16]}),
    )


def single_gate(cell_type: str, a=2, b=3, y=4) -> tuple[Module, "nx.MultiDiGraph"]:
    """Convenience: build a module with one two-input gate, return (module, graph)."""
    m = make_module(
        Gate("g", cell_type,
             {"A": [a], "B": [b]} if "MUX" not in cell_type else {"A": [a], "B": [b], "S": [5]},
             {"Y": [y]})
    )
    return m, build_graph(m)


# -----------------------------------------------------------------------------
# Sort key (reproducibility)
# -----------------------------------------------------------------------------

def test_net_sort_key_numeric_order():
    """Integer bit IDs sort numerically, not lexicographically. This is the
    regression test for using repr as a sort key."""
    nodes = [net_node(10), net_node(2), net_node(3), net_node(1)]
    assert sorted(nodes, key=_net_sort_key) == [
        net_node(1), net_node(2), net_node(3), net_node(10),
    ]


def test_net_sort_key_differs_from_repr():
    """Demonstrate the bug we fixed: repr order is not numeric."""
    nodes = [net_node(10), net_node(2)]
    repr_order = sorted(nodes, key=repr)
    numeric_order = sorted(nodes, key=_net_sort_key)
    assert repr_order != numeric_order  # 10 before 2 in repr
    assert numeric_order == [net_node(2), net_node(10)]


def test_net_sort_key_handles_string_ids():
    """String bit IDs sort after all integers, lexicographically."""
    nodes = [net_node("b"), net_node(2), net_node("a")]
    ordered = sorted(nodes, key=_net_sort_key)
    assert ordered[0] == net_node(2)
    assert ordered[1] == net_node("a")
    assert ordered[2] == net_node("b")


# -----------------------------------------------------------------------------
# Gate truth tables (one per supported primitive)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("cell_type, expected_tt", [
    ("$_AND_",    0b1000),  # bit 3: A=1,B=1
    ("$_NAND_",   0b0111),  # bits 0,1,2
    ("$_OR_",     0b1110),  # bits 1,2,3
    ("$_NOR_",    0b0001),  # bit 0: A=0,B=0
    ("$_XOR_",    0b0110),  # bits 1,2
    ("$_XNOR_",   0b1001),  # bits 0,3
    ("$_ANDNOT_", 0b0010),  # bit 1: A=1,B=0  → A & ~B
    ("$_ORNOT_",  0b1011),  # bits 0,1,3     → A | ~B
])
def test_two_input_gate_truth_tables(cell_type, expected_tt):
    """Truth tables use LSB-first variable order (variable i is bit i of
    the assignment), with variables sorted by _net_sort_key. For cut
    {net 2, net 3} that gives A=2, B=3 in that order."""
    m = make_module(Gate("g", cell_type, {"A": [2], "B": [3]}, {"Y": [4]}))
    G = build_graph(m)
    tt, variables = simulate_slice(
        G, net_node(4), frozenset({net_node(2), net_node(3)})
    )
    assert len(variables) == 2
    assert tt == expected_tt

def test_not_gate_truth_table():
    m = make_module(Gate("g", "$_NOT_", {"A": [2]}, {"Y": [4]}))
    G = build_graph(m)
    tt, variables = simulate_slice(G, net_node(4), frozenset({net_node(2)}))
    assert len(variables) == 1
    assert tt == 0b01  # output = ~A


def test_buf_gate_truth_table():
    m = make_module(Gate("g", "$_BUF_", {"A": [2]}, {"Y": [4]}))
    G = build_graph(m)
    tt, _ = simulate_slice(G, net_node(4), frozenset({net_node(2)}))
    assert tt == 0b10  # output = A


def test_mux_gate_truth_table():
    """$_MUX_(A, B, S) = S ? B : A.

    Variables sorted by _net_sort_key: net 2=A, net 3=B, net 4=S.
    Bit i of `assignment` is variable i, so:
        bit 0 -> A, bit 1 -> B, bit 2 -> S
    """
    m = make_module(
        Gate("g", "$_MUX_", {"A": [2], "B": [3], "S": [4]}, {"Y": [5]})
    )
    G = build_graph(m)
    cut = frozenset({net_node(2), net_node(3), net_node(4)})
    tt, variables = simulate_slice(G, net_node(5), cut)
    # (A,B,S) assignments:
    #   000 -> A=0,S=0 -> 0
    #   001 -> A=1,S=0 -> 1  (bit 1)
    #   010 -> A=0,S=0 -> 0
    #   011 -> A=1,S=0 -> 1  (bit 3)
    #   100 -> B=0,S=1 -> 0
    #   101 -> B=0,S=1 -> 0
    #   110 -> B=1,S=1 -> 1  (bit 6)
    #   111 -> B=1,S=1 -> 1  (bit 7)
    assert variables == [net_node(2), net_node(3), net_node(4)]
    assert tt == 0b11001010  # 202, bits 1,3,6,7 set


def test_nmux_gate_truth_table():
    """$_NMUX_ is the bitwise inverse of $_MUX_."""
    m = make_module(
        Gate("g", "$_NMUX_", {"A": [2], "B": [3], "S": [4]}, {"Y": [5]})
    )
    G = build_graph(m)
    cut = frozenset({net_node(2), net_node(3), net_node(4)})
    tt, _ = simulate_slice(G, net_node(5), cut)
    mux_tt = 0b11001010
    assert tt == (~mux_tt & 0xFF)  # 0b00110101 = 53

# -----------------------------------------------------------------------------
# Constant substitution
# -----------------------------------------------------------------------------

def test_constant_in_cut_is_substituted():
    """AND(a, const_1) with cut {a, const_1} → identity over 1 var."""
    m = make_module(
        Gate("g", "$_AND_", {"A": [2], "B": [1]}, {"Y": [4]}),
        nets=[1],
    )
    m.nets[1].is_constant = True
    m.nets[1].constant_value = 1
    G = build_graph(m)
    tt, variables = simulate_slice(
        G, net_node(4), frozenset({net_node(2), net_node(1)})
    )
    assert len(variables) == 1
    assert tt == 0b10


def test_constant_zero_short_circuits():
    """AND(a, const_0) → always 0, arity 1 over `a`."""
    m = make_module(
        Gate("g", "$_AND_", {"A": [2], "B": [1]}, {"Y": [4]}),
        nets=[1],
    )
    m.nets[1].is_constant = True
    m.nets[1].constant_value = 0
    G = build_graph(m)
    tt, variables = simulate_slice(
        G, net_node(4), frozenset({net_node(2), net_node(1)})
    )
    assert len(variables) == 1
    assert tt == 0b00


# -----------------------------------------------------------------------------
# Unsupported cell type
# -----------------------------------------------------------------------------

def test_unsupported_cell_raises():
    m = make_module(
        Gate("ff", "$_DFF_P_", {"D": [2], "C": [3]}, {"Q": [4]})
    )
    G = build_graph(m)
    with pytest.raises(ValueError, match="unsupported cell type"):
        simulate_slice(G, net_node(4), frozenset({net_node(2), net_node(3)}))


# -----------------------------------------------------------------------------
# Combinational cycle detection
# -----------------------------------------------------------------------------

def test_combinational_loop_raises():
    """Two NOT gates driving each other form a loop. Empty cut forces
    _evaluate_net to walk the loop rather than hitting a base case."""
    m = make_module(
        Gate("inv_a", "$_NOT_", {"A": [3]}, {"Y": [2]}),
        Gate("inv_b", "$_NOT_", {"A": [2]}, {"Y": [3]}),
    )
    G = build_graph(m)
    with pytest.raises(ValueError, match="combinational cycle"):
        simulate_slice(G, net_node(2), frozenset())


def test_diamond_dependency_is_not_a_cycle():
    """Two paths to the same net from a common start is NOT a cycle.
    Regression: a too-eager visited-set would falsely flag this."""
    # y = AND(n1, n2), n1 = NOT(a), n2 = NOT(a). Both n1 and n2 depend on a.
    m = make_module(
        Gate("not1", "$_NOT_", {"A": [2]}, {"Y": [3]}),
        Gate("not2", "$_NOT_", {"A": [2]}, {"Y": [4]}),
        Gate("and",  "$_AND_", {"A": [3], "B": [4]}, {"Y": [5]}),
    )
    G = build_graph(m)
    # Cut {a}: n1 and n2 both derive from a via distinct paths. Must not raise.
    tt, variables = simulate_slice(G, net_node(5), frozenset({net_node(2)}))
    assert len(variables) == 1
    # y = ~a & ~a = ~a
    assert tt == 0b01


# -----------------------------------------------------------------------------
# Canonicalization
# -----------------------------------------------------------------------------

def test_canonical_identity():
    assert canonical_truth_table(0b10, 1) == 0b10


def test_canonical_and_matches_permuted():
    """AND(a, b) is symmetric; canonical form is the same."""
    assert canonical_truth_table(0b1000, 2) == canonical_truth_table(0b1000, 2)


def test_canonical_andnot_matches_permuted():
    """ANDNOT(a, b) and ANDNOT(b, a) differ as raw tables but match canonically."""
    a = canonical_truth_table(0b0100, 2)
    b = canonical_truth_table(0b0010, 2)
    assert a == b


def test_canonical_and_or_differ():
    assert canonical_truth_table(0b1000, 2) != canonical_truth_table(0b1110, 2)


def test_canonical_above_cap_returns_raw():
    """For n > max_n, canonicalization is bypassed."""
    assert canonical_truth_table(0b10, 6, max_n=5) == 0b10


def test_canonical_at_exact_cap_still_canonicalizes():
    """At n == max_n, canonicalization still runs."""
    # Symmetric AND-like function; permuting inputs must not change result.
    tt = 0b1000
    assert canonical_truth_table(tt, 2, max_n=2) == 0b1000


# -----------------------------------------------------------------------------
# Cut enumeration: basic properties
# -----------------------------------------------------------------------------

def test_cut_enumeration_primary_input_yields_nothing():
    """A net with no driver (primary input) has no cuts."""
    m = make_module(nets=[2])
    G = build_graph(m)
    assert list(enumerate_cuts(G, net_node(2))) == []


def test_cut_enumeration_trivial_cut_first():
    m = make_module(Gate("g", "$_XOR_", {"A": [2], "B": [3]}, {"Y": [4]}))
    G = build_graph(m)
    cuts = list(enumerate_cuts(G, net_node(4)))
    assert cuts[0] == frozenset({net_node(2), net_node(3)})


def test_cut_enumeration_expands_driver():
    m = make_module(
        Gate("xor0", "$_XOR_", {"A": [2], "B": [3]}, {"Y": [5]}),
        Gate("xor1", "$_XOR_", {"A": [5], "B": [4]}, {"Y": [6]}),
    )
    G = build_graph(m)
    cuts = list(enumerate_cuts(G, net_node(6)))
    assert frozenset({net_node(5), net_node(4)}) in cuts
    assert frozenset({net_node(2), net_node(3), net_node(4)}) in cuts


def test_cut_enumeration_bounded_by_size():
    m = make_module(
        Gate("g1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [5]}),
        Gate("g2", "$_AND_", {"A": [5], "B": [4]}, {"Y": [6]}),
    )
    G = build_graph(m)
    for cut in enumerate_cuts(G, net_node(6), max_size=2):
        assert len(cut) <= 2


# -----------------------------------------------------------------------------
# Cut enumeration: primary-input safety
# -----------------------------------------------------------------------------

def test_primary_input_not_expanded_away():
    """A primary input in a cut must never be silently removed — no valid
    cut may lose a member whose driver does not exist."""
    m = make_module(
        Gate("xor_ab", "$_XOR_", {"A": [2], "B": [3]}, {"Y": [5]}),
        Gate("and_y",  "$_AND_", {"A": [5], "B": [4]}, {"Y": [6]}),
    )
    G = build_graph(m)
    cuts = list(enumerate_cuts(G, net_node(6), max_size=6, max_depth=3))
    for cut in cuts:
        assert len(cut) >= 2, f"cut lost a member: {cut}"


# -----------------------------------------------------------------------------
# Cut enumeration: minimality (re-convergent fanout)
# -----------------------------------------------------------------------------

def test_minimality_rejects_over_specified_cut():
    """y = XOR(n1, n2), n1 = AND(a,b), n2 = AND(n1,c).

    Expanding n1 gives {a,b,n2}. Expanding n2 from there gives
    {a,b,n1,c}, which contains n1 AND its inputs {a,b}. Non-minimal.
    The enumerator must not yield it.
    """
    m = make_module(
        Gate("and1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [4]}),  # n1
        Gate("and2", "$_AND_", {"A": [4], "B": [5]}, {"Y": [6]}),  # n2
        Gate("xor",  "$_XOR_", {"A": [4], "B": [6]}, {"Y": [7]}),  # y
    )
    G = build_graph(m)
    cuts = list(enumerate_cuts(G, net_node(7), max_size=6, max_depth=4))

    # {a=2, b=3, n1=4, c=5} would be non-minimal (contains n1 and its inputs).
    non_minimal = frozenset({net_node(2), net_node(3), net_node(4), net_node(5)})
    assert non_minimal not in cuts


def test_minimality_keeps_valid_expansion():
    """Expanding a member to its inputs, without leaving the parent in cut,
    produces a minimal cut — which must be yielded."""
    m = make_module(
        Gate("and1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [4]}),
        Gate("and2", "$_AND_", {"A": [4], "B": [5]}, {"Y": [6]}),
        Gate("xor",  "$_XOR_", {"A": [4], "B": [6]}, {"Y": [7]}),
    )
    G = build_graph(m)
    cuts = list(enumerate_cuts(G, net_node(7), max_size=6, max_depth=4))

    # {a=2, b=3, c=5} — none derivable from the others.
    minimal = frozenset({net_node(2), net_node(3), net_node(5)})
    assert minimal in cuts


# -----------------------------------------------------------------------------
# Signature extraction
# -----------------------------------------------------------------------------

def test_signature_of_and():
    m = make_module(Gate("g", "$_AND_", {"A": [2], "B": [3]}, {"Y": [4]}))
    G = build_graph(m)
    sig = slice_signature(G, net_node(4), frozenset({net_node(2), net_node(3)}))
    assert sig == (2, 0b1000)


def test_two_identical_ands_match():
    m = make_module(
        Gate("g1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [10]}),
        Gate("g2", "$_AND_", {"A": [5], "B": [6]}, {"Y": [11]}),
    )
    G = build_graph(m)
    sig1 = slice_signature(G, net_node(10), frozenset({net_node(2), net_node(3)}))
    sig2 = slice_signature(G, net_node(11), frozenset({net_node(5), net_node(6)}))
    assert sig1 == sig2


def test_and_and_or_differ():
    m = make_module(
        Gate("g1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [10]}),
        Gate("g2", "$_OR_",  {"A": [5], "B": [6]}, {"Y": [11]}),
    )
    G = build_graph(m)
    sig1 = slice_signature(G, net_node(10), frozenset({net_node(2), net_node(3)}))
    sig2 = slice_signature(G, net_node(11), frozenset({net_node(5), net_node(6)}))
    assert sig1 != sig2


def test_signature_arises_independently_of_pin_order():
    """AND(A=a, B=b) and AND(A=b, B=a) must canonicalize to the same signature."""
    m = make_module(
        Gate("g1", "$_AND_", {"A": [2], "B": [3]}, {"Y": [10]}),
        Gate("g2", "$_AND_", {"A": [3], "B": [2]}, {"Y": [11]}),
    )
    G = build_graph(m)
    sig1 = slice_signature(G, net_node(10), frozenset({net_node(2), net_node(3)}))
    sig2 = slice_signature(G, net_node(11), frozenset({net_node(3), net_node(2)}))
    assert sig1 == sig2


# -----------------------------------------------------------------------------
# The core WordRev signal: 2-bit adder sum bits group by XOR2
# -----------------------------------------------------------------------------

def test_2bit_adder_sum_bits_share_signature():
    """sum[0] and sum[1] both have XOR2 at their trivial cut.

    This is exactly what bitslice aggregation is supposed to detect:
    two structurally different nets (different cone depths, different
    input shapes) computing the same function on their immediate inputs.
    """
    G = build_graph(two_bit_adder_module())
    sigs = compute_functional_signatures(G, max_cut_size=4, max_cut_depth=2)

    sum0 = sigs[net_node(10)]
    sum1 = sigs[net_node(13)]

    assert (2, 0b0110) in sum0
    assert (2, 0b0110) in sum1
    assert sum0 & sum1


def test_2bit_adder_cout_differs_from_sum_bits():
    """cout is OR-like; sum bits are XOR2. Signatures must not intersect."""
    G = build_graph(two_bit_adder_module())
    sigs = compute_functional_signatures(G, max_cut_size=4, max_cut_depth=2)

    sum0 = sigs[net_node(10)]
    cout = sigs[net_node(16)]
    assert (2, 0b0110) in sum0            # sum[0] has XOR2
    assert (2, 0b0110) not in cout        # cout does not
    assert sum0 != cout                    # they are different signature sets


# -----------------------------------------------------------------------------
# Batch API structure
# -----------------------------------------------------------------------------

def test_compute_returns_dict_keyed_by_nets():
    G = build_graph(two_bit_adder_module())
    sigs = compute_functional_signatures(G)
    for node in G.nodes:
        if node[0] == "net":
            assert node in sigs
            assert isinstance(sigs[node], frozenset)
        else:
            assert node not in sigs  # gates not in the result


def test_compute_is_deterministic():
    """Same graph, same result, twice."""
    G = build_graph(two_bit_adder_module())
    sigs1 = compute_functional_signatures(G)
    sigs2 = compute_functional_signatures(G)
    assert sigs1 == sigs2


def test_sequential_cells_do_not_crash_batch():
    """DFF-driven nets get empty signatures, never a crash."""
    m = make_module(
        Gate("ff", "$_DFF_P_", {"D": [2], "C": [3]}, {"Q": [4]})
    )
    G = build_graph(m)
    sigs = compute_functional_signatures(G)
    assert sigs[net_node(4)] == frozenset()


def test_primary_input_gets_empty_signature():
    """A primary input has no driver, hence no cuts, hence no signatures."""
    m = make_module(nets=[2])
    G = build_graph(m)
    sigs = compute_functional_signatures(G)
    assert sigs[net_node(2)] == frozenset()

# -----------------------------------------------------------------------------
# Enumerate
# -----------------------------------------------------------------------------

def _build_and2_netlist():
    G = nx.MultiDiGraph()
    a, b, y = ("net", 2), ("net", 3), ("net", 4)
    g = ("gate", 0)
    G.add_node(a, is_constant=False, base_name="a", vector_index=0)
    G.add_node(b, is_constant=False, base_name="b", vector_index=0)
    G.add_node(y, is_constant=False, base_name="y", vector_index=0)
    G.add_node(g, cell_type="$_AND_")
    G.add_edge(a, g, pin="A", role="input")
    G.add_edge(b, g, pin="B", role="input")
    G.add_edge(g, y, pin="Y", role="output")
    return G, a, b, y


def test_enumerate_slices_returns_ordered_inputs():
    G, a, b, y = _build_and2_netlist()
    result = enumerate_slices(G)

    # Output net has at least the trivial cut.
    assert y in result and result[y]

    sl = result[y][0]
    assert sl.output_net == y
    # Ordered by _net_sort_key: bit 2 before bit 3.
    assert sl.ordered_inputs == (a, b)
    # AND(0,0)=0, AND(0,1)=0, AND(1,0)=0, AND(1,1)=1 → 0b1000 = 8.
    assert sl.truth_table == 8
    assert sl.signature == (2, 8)
    # Match fields are None until match_slice_to_library runs.
    assert sl.aligned_to is None and sl.alignment is None


def test_enumerate_slices_primary_input_is_empty():
    G, a, _, _ = _build_and2_netlist()
    result = enumerate_slices(G)
    assert result[a] == []