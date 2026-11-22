# wordrev/slice.py
"""
Bitslice extraction: cut enumeration, slice simulation, functional signatures.

For each net, we enumerate valid *minimal* cuts — sets of nets that fully
determine the net's value and are not redundant (no cut member is derivable
from the others). For each cut, we simulate the subcircuit between cut and
net to obtain a truth table over the cut's non-constant members. The truth
table is canonicalized under input permutation so that two slices computing
the same function from permuted inputs match.

Supported cell types are the Yosys gate primitives emitted by `techmap`:
    $_AND_, $_NAND_, $_OR_, $_NOR_, $_XOR_, $_XNOR_,
    $_ANDNOT_, $_ORNOT_, $_NOT_, $_BUF_, $_MUX_, $_NMUX_

Design notes:

  - Cycle detection is path-based: a net may be re-entered from sibling
    branches (diamonds), but a back-edge along the current path raises
    ValueError. A cyclic netlist cannot be sliced combinationally.

  - Re-convergent fanout (a net whose inputs are also present in the cut)
    would produce semantically over-specified signatures. `_is_minimal_cut`
    rejects those cuts during enumeration. `slice_signature` assumes the
    cut is minimal; callers that construct cuts manually are responsible
    for this precondition.

  - Sort order of cut members uses `_net_sort_key`, which sorts integer
    bit IDs numerically. This keeps exploration reproducible across Yosys
    runs that assign different bit IDs to structurally isomorphic cones.
"""
from __future__ import annotations

import itertools
from typing import Any

import networkx as nx

from .graph import Node


# -----------------------------------------------------------------------------
# Node sorting
# -----------------------------------------------------------------------------

def _net_sort_key(node: Node) -> tuple:
    """Stable, semantically meaningful sort key for net/gate nodes.

    Integer bit IDs sort numerically (2 before 10). String bit IDs sort
    lexicographically after all integers. The specific order is not
    important for correctness — what matters is that it is *independent
    of bit-ID assignment*, so two structurally isomorphic cones enumerate
    the same cut set regardless of which integers Yosys happened to pick.
    """
    kind, bit_id = node
    if isinstance(bit_id, int) and not isinstance(bit_id, bool):
        return (kind, 0, bit_id)
    return (kind, 1, str(bit_id))


# -----------------------------------------------------------------------------
# Gate function table
# -----------------------------------------------------------------------------

_GATE_FUNCS: dict[str, Any] = {
    "$_AND_":    lambda v: v["A"] & v["B"],
    "$_NAND_":   lambda v: 1 - (v["A"] & v["B"]),
    "$_OR_":     lambda v: v["A"] | v["B"],
    "$_NOR_":    lambda v: 1 - (v["A"] | v["B"]),
    "$_XOR_":    lambda v: v["A"] ^ v["B"],
    "$_XNOR_":   lambda v: 1 - (v["A"] ^ v["B"]),
    "$_ANDNOT_": lambda v: v["A"] & (1 - v["B"]),
    "$_ORNOT_":  lambda v: v["A"] | (1 - v["B"]),
    "$_NOT_":    lambda v: 1 - v["A"],
    "$_BUF_":    lambda v: v["A"],
    "$_MUX_":    lambda v: v["B"] if v["S"] else v["A"],
    "$_NMUX_":   lambda v: 1 - (v["B"] if v["S"] else v["A"]),
}


def _apply_cell(cell_type: str, inputs: dict[str, int]) -> int:
    fn = _GATE_FUNCS.get(cell_type)
    if fn is None:
        raise ValueError(f"unsupported cell type: {cell_type}")
    return fn(inputs)


# -----------------------------------------------------------------------------
# Graph helpers (netlist-level)
# -----------------------------------------------------------------------------

def _driver_gate(G: nx.MultiDiGraph, net: Node) -> Node | None:
    """The gate that drives `net` (via output or inout pin), or None."""
    for pred in G.predecessors(net):
        if pred[0] != "gate":
            continue
        for attrs in G[pred][net].values():
            if attrs["role"] in ("output", "inout"):
                return pred
    return None


def _driver_inputs(G: nx.MultiDiGraph, net: Node) -> frozenset[Node]:
    """Input nets of the gate driving `net`. Empty if the net has no driver."""
    gate = _driver_gate(G, net)
    if gate is None:
        return frozenset()
    inputs: set[Node] = set()
    for pred in G.predecessors(gate):
        if pred[0] != "net":
            continue
        for attrs in G[pred][gate].values():
            if attrs["role"] == "input":
                inputs.add(pred)
    return frozenset(inputs)


# -----------------------------------------------------------------------------
# Slice evaluation with cycle protection
# -----------------------------------------------------------------------------

def _evaluate_net(
    G: nx.MultiDiGraph,
    net: Node,
    values: dict[Node, int],
    visited: set[Node] | None = None,
) -> int:
    """Evaluate `net` given concrete values for cut members, with cycle detection.

    `visited` tracks the *current* recursion path, not all nets ever seen.
    Sibling branches may legitimately re-enter the same net; only a back-edge
    along the current path is a cycle.

    Only `role="input"` edges are followed. Inout edges are deliberately
    ignored — following them would create a false cycle whenever an inout
    net appears in its own backward cone (e.g. a bidirectional pad). When
    a cell with inout pins is added to `_GATE_FUNCS`, inout traversal
    should be designed explicitly rather than conflated with input
    traversal here.
    """
    if net in values:
        return values[net]

    data = G.nodes[net]
    if data.get("is_constant"):
        return int(data["constant_value"])

    if visited is None:
        visited = set()
    if net in visited:
        raise ValueError(f"combinational cycle detected involving net {net!r}")
    visited.add(net)

    gate = _driver_gate(G, net)
    if gate is None:
        raise ValueError(f"net {net!r} has no driver and is not in cut")

    input_vals: dict[str, int] = {}
    for pred in G.predecessors(gate):
        if pred[0] != "net":
            continue
        for attrs in G[pred][gate].values():
            if attrs["role"] == "input":
                input_vals[attrs["pin"]] = _evaluate_net(G, pred, values, visited)

    visited.remove(net)
    return _apply_cell(G.nodes[gate]["cell_type"], input_vals)


def simulate_slice(
    G: nx.MultiDiGraph,
    net: Node,
    cut: frozenset[Node],
) -> tuple[int, list[Node]]:
    """Simulate the slice defined by (net, cut).

    Returns (truth_table, variables):
        truth_table: integer whose bit `a` is the output for input
                     assignment `a` (variable i of `variables` is bit i).
        variables:   sorted list of non-constant cut members, in variable order.

    Constants in the cut are substituted and excluded from `variables`.

    Precondition: `cut` must be a *minimal* cut for `net` (no member derivable
    from the others). `enumerate_cuts` yields only minimal cuts; direct
    callers of this function are responsible for this precondition.

    Raises ValueError if the slice contains an unsupported cell type or a
    combinational cycle.
    """
    constants: dict[Node, int] = {}
    variables: list[Node] = []
    for member in sorted(cut, key=_net_sort_key):
        data = G.nodes[member]
        if data.get("is_constant"):
            constants[member] = int(data["constant_value"])
        else:
            variables.append(member)

    n = len(variables)
    tt = 0
    for assignment in range(1 << n):
        values = dict(constants)
        for i, var in enumerate(variables):
            values[var] = (assignment >> i) & 1
        if _evaluate_net(G, net, values):
            tt |= 1 << assignment
    return tt, variables


# -----------------------------------------------------------------------------
# Canonical truth table
# -----------------------------------------------------------------------------

def _permute_tt(tt: int, perm: tuple[int, ...]) -> int:
    """Apply permutation to truth table: `perm[i]` is the new position of var i."""
    n = len(perm)
    result = 0
    for old in range(1 << n):
        if not (tt >> old) & 1:
            continue
        new = 0
        for old_pos in range(n):
            if (old >> old_pos) & 1:
                new |= 1 << perm[old_pos]
        result |= 1 << new
    return result


def canonical_truth_table(tt: int, n: int, *, max_n: int = 5) -> int:
    """Permutation-invariant form of `tt`. For n > max_n, returns raw `tt`."""
    if n <= 1 or n > max_n:
        return tt
    return min(_permute_tt(tt, p) for p in itertools.permutations(range(n)))


# -----------------------------------------------------------------------------
# Cut enumeration with minimality filtering
# -----------------------------------------------------------------------------

def _is_minimal_cut(G: nx.MultiDiGraph, cut: frozenset[Node]) -> bool:
    """Return True if no member of cut can be derived from other members of cut."""
    for member in cut:
        if G.nodes[member].get("is_constant"):
            continue
        driver_inputs = _driver_inputs(G, member)
        if driver_inputs and driver_inputs.issubset(cut):
            return False
    return True


def enumerate_cuts(
    G: nx.MultiDiGraph,
    net: Node,
    *,
    max_size: int = 6,
    max_depth: int = 4,
):
    """Yield valid minimal cuts for `net`, from trivial to expanded.

    The first yielded cut is the trivial cut (immediate driver inputs).
    Expansion replaces one non-constant member at a time with its driver
    inputs, bounded by `max_size` and `max_depth`. Non-minimal cuts
    (re-convergent fanout) are filtered out before yielding.

    Yields nothing for primary inputs and any net whose gate has no inputs.
    """
    initial = _driver_inputs(G, net)
    if not initial:
        return
    if len(initial) > max_size:
        yield initial
        return

    frontier: list[tuple[frozenset[Node], int]] = [(initial, 0)]
    seen: set[frozenset[Node]] = {initial}

    while frontier:
        cut, depth = frontier.pop()
        if _is_minimal_cut(G, cut):
            yield cut

        if depth >= max_depth:
            continue

        for member in sorted(cut, key=_net_sort_key):
            if G.nodes[member].get("is_constant"):
                continue
            expanded_members = _driver_inputs(G, member)
            if not expanded_members:
                continue

            new_cut = (cut - {member}) | expanded_members
            if len(new_cut) > max_size:
                continue
            if new_cut in seen:
                continue

            seen.add(new_cut)
            frontier.append((new_cut, depth + 1))


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def slice_signature(
    G: nx.MultiDiGraph,
    net: Node,
    cut: frozenset[Node],
    *,
    max_canonical_n: int = 5,
) -> tuple[int, int]:
    """(n_inputs, canonical_truth_table) for one specific slice.

    Precondition: `cut` must be a *minimal* cut for `net`. `enumerate_cuts`
    yields only minimal cuts, so the standard pipeline is safe. Direct
    callers that construct cuts by hand must ensure minimality — a
    non-minimal cut produces a well-defined but semantically over-specified
    signature (redundant variables make the truth table arity larger than
    the function's true essential arity).

    Raises ValueError if the slice contains an unsupported cell type or a
    combinational cycle.
    """
    tt, variables = simulate_slice(G, net, cut)
    n = len(variables)
    return (n, canonical_truth_table(tt, n, max_n=max_canonical_n))


def compute_functional_signatures(
    G: nx.MultiDiGraph,
    *,
    max_cut_size: int = 6,
    max_cut_depth: int = 4,
    max_canonical_n: int = 5,
) -> dict[Node, frozenset[tuple[int, int]]]:
    """For every net, the set of (n_inputs, canonical_tt) across valid cuts.

    Nets with no driver (primary inputs) and nets whose slices all raise
    (sequential cells, unsupported cells, cycles) get an empty frozenset.
    Callers should interpret empty as "no combinational signature available,"
    which is the correct semantics for a sequential or boundary net.
    """
    result: dict[Node, frozenset[tuple[int, int]]] = {}
    for node in G.nodes:
        if node[0] != "net":
            continue
        sigs: set[tuple[int, int]] = set()
        for cut in enumerate_cuts(
            G, node, max_size=max_cut_size, max_depth=max_cut_depth
        ):
            try:
                sig = slice_signature(G, node, cut, max_canonical_n=max_canonical_n)
            except ValueError:
                # Unsupported cell or cycle detected; skip this cut cleanly.
                continue
            sigs.add(sig)
        result[node] = frozenset(sigs)
    return result