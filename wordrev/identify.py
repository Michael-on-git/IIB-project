# wordrev/identify.py
"""
Function identification: match a recovered Word against a library of
known word-level operations.

Algorithm:
    1. Boundary. Compute the backward cone of the Word's bits and stop
       at primary inputs and constants. The resulting "free variables"
       and "constants" define the fragment's external interface.
    2. Grouping. Group free variables by their base_name and vector_index,
       so a[0..3] is one group, b[0..3] is another, and anonymous single
       bits form singleton groups. Group widths must match the library
       component's port widths for an alignment to be tried.
    3. Truth table. Enumerate all 2^k free-variable assignments, simulate
       the fragment, and record the Word's output for each assignment.
    4. Match. For each library component with the right output width and
       total input bits, try all valid group→port alignments. A component
       matches an alignment iff its truth table equals the recovered one.
    5. Deduplication. Symmetric/commutative port permutations (e.g. a=X, b=Y
       vs. a=Y, b=X on an adder) are canonicalized so each distinct physical
       operation yields exactly one Match.
    
    Returns a list of Matches (possibly empty) — canonicalized so that
    commutative port permutations of the same component yield a single
    Match. Functionally equivalent components in the library (e.g. a 1-bit
    XOR and a 1-bit adder) each produce their own Match; the caller
    chooses which to report based on domain knowledge or priority order.

Limitations:
    - Enumerates 2^k truth tables, k = free-variable count. Practical up
      to k ≈ 12. Larger fragments return no match — this is deliberate.
    - Does not reason about side inputs from gate sharing. Designs where
      synthesis has merged logic across the fragment and outside it will
      produce truth tables that do not match any library component. This
      is exactly the case the paper's QBF extension handles; it is out of
      scope for this mockup.
    - Sequential operations are out of scope. Fragments containing DFFs
      raise during simulation and produce no match.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import networkx as nx

from .aggregate import Word, _bit_sort_key
from .graph import Node
from .library import Component, candidates_for_width
from .slice import _driver_gate, _evaluate_net

log = logging.getLogger(__name__)

# Fragment size and input-count bounds. Beyond these the truth table is
# too expensive and the caller should not treat "no match" as meaningful.
_MAX_EXTERNAL_INPUTS = 12          # ~4 s per word; interactive-friendly
_MAX_EXTERNAL_INPUTS_BATCH = 14    # ~1 min per word; batch use
_MAX_EXTERNAL_INPUTS_HARD = 16     # ~20 min per word; overnight runs
_MAX_FRAGMENT_GATES = 500


# -----------------------------------------------------------------------------
# Match result
# -----------------------------------------------------------------------------

@dataclass
class Match:
    """A confirmed match between a recovered Word and a library component.

    `alignment` maps each library port name to the list of boundary Nets
    that feed it, in LSB-first order. So for an adder, `alignment["a"]`
    is the ordered list of nets feeding the library's `a` port.
    """
    component: Component
    alignment: dict[str, list[Node]]

    def __repr__(self) -> str:
        ports = {p: [n[1] for n in nets] for p, nets in self.alignment.items()}
        return f"Match({self.component.name}, ports={ports})"


# -----------------------------------------------------------------------------
# Boundary discovery
# -----------------------------------------------------------------------------

@dataclass
class _Fragment:
    free_vars: list[Node]        # ordered, non-constant boundary nets
    constants: dict[Node, int]  # constant boundary nets -> fixed value
    interior_gates: set[Node]    # gates strictly inside the fragment


def _find_fragment(
    G: nx.MultiDiGraph,
    word_bits: list[Node],
    *,
    max_gates: int | None = None,
) -> _Fragment | None:
    """Compute the fragment producing `word_bits`.

    Walks backward from every Word bit, stopping at nets with no driver
    or nets explicitly marked as constants. Those stopped nets are the boundary.
    Everything between the boundary and the Word's bits is the interior.
    """
    boundary: set[Node] = set()
    interior: set[Node] = set()
    visited: set[Node] = set()
    frontier = list(word_bits)

    while frontier:
        net = frontier.pop()
        if net in visited:
            continue
        visited.add(net)

        gate = _driver_gate(G, net)
        if G.nodes[net].get("is_constant") or gate is None:
            boundary.add(net)
            continue

        interior.add(gate)
        if max_gates is not None and len(interior) > max_gates:
            return None
        for pred in G.predecessors(gate):
            if pred[0] == "net":
                frontier.append(pred)

    free_vars: list[Node] = []
    constants: dict[Node, int] = {}
    for net in boundary:
        data = G.nodes[net]
        if data.get("is_constant"):
            constants[net] = int(data["constant_value"])
        else:
            free_vars.append(net)
    free_vars.sort(key=lambda n: _bit_sort_key(n, G))

    return _Fragment(free_vars=free_vars, constants=constants, interior_gates=interior)


# -----------------------------------------------------------------------------
# Grouping
# -----------------------------------------------------------------------------

def _group_free_vars(
    G: nx.MultiDiGraph, free_vars: list[Node]
) -> list[tuple[str, list[Node]]]:
    """Group free variables by base_name; singleton groups for anonymous nets."""
    groups: dict[str, list[Node]] = defaultdict(list)
    for net in free_vars:
        base = G.nodes[net].get("base_name") or f"__single_{net[1]}"
        groups[base].append(net)

    out = []
    for name in sorted(groups):
        bits = sorted(groups[name], key=lambda n: _bit_sort_key(n, G))
        out.append((name, bits))
    return out


# -----------------------------------------------------------------------------
# Truth table construction
# -----------------------------------------------------------------------------

def _build_recovered_tt(
    G: nx.MultiDiGraph,
    word_bits: list[Node],
    free_vars: list[Node],
    constants: dict[Node, int],
) -> dict[int, int]:
    """Map every free-variable assignment to the Word's output as an int."""
    k = len(free_vars)
    tt: dict[int, int] = {}
    for a in range(1 << k):
        values = dict(constants)
        for i, net in enumerate(free_vars):
            values[net] = (a >> i) & 1
        out = 0
        for i, bit in enumerate(word_bits):
            if _evaluate_net(G, bit, values):
                out |= 1 << i
        tt[a] = out
    return tt


# -----------------------------------------------------------------------------
# Alignment generation
# -----------------------------------------------------------------------------

def _alignments(
    component: Component,
    groups: list[tuple[str, list[Node]]],
    free_vars: list[Node],
) -> Iterable[dict[str, list[Node]]]:
    """Yield all bijections of component ports to groups with matching widths."""
    group_positions = {name: nets for name, nets in groups}
    group_widths = {name: len(nets) for name, nets in groups}
    ports = list(component.input_ports)

    def backtrack(
        i: int,
        used: set[str],
        current: dict[str, list[Node]],
    ):
        if i == len(ports):
            yield dict(current)
            return
        port_name, port_width = ports[i]
        for gname, gwidth in group_widths.items():
            if gname in used or gwidth != port_width:
                continue
            current[port_name] = group_positions[gname]
            used.add(gname)
            yield from backtrack(i + 1, used, current)
            used.discard(gname)
            del current[port_name]

    yield from backtrack(0, set(), {})


# -----------------------------------------------------------------------------
# Verification & Deduplication
# -----------------------------------------------------------------------------

def _verify_alignment(
    component: Component,
    alignment: dict[str, list[Node]],
    free_vars: list[Node],
    recovered_tt: dict[int, int],
) -> bool:
    """True iff the library component, aligned as given, matches recovered_tt."""
    position = {net: i for i, net in enumerate(free_vars)}
    port_positions = {
        port: [position[net] for net in nets]
        for port, nets in alignment.items()
    }
    k = len(free_vars)
    for a in range(1 << k):
        port_vals: dict[str, int] = {}
        for port, positions in port_positions.items():
            v = 0
            for i, pos in enumerate(positions):
                v |= ((a >> pos) & 1) << i
            port_vals[port] = v
        if component.compute(port_vals) != recovered_tt[a]:
            return False
    return True

def _node_key(n: Node) -> tuple:
    kind, bit_id = n
    if isinstance(bit_id, int) and not isinstance(bit_id, bool):
        return (kind, 0, bit_id)
    return (kind, 1, str(bit_id))

def _canonical_alignment_key(
    component: Component, alignment: dict[str, list[Node]]
) -> tuple:
    """Canonical form that collapses commutative port permutations only.

    Two alignments are equivalent iff they differ only by a permutation
    of ports within a commutative group. Alignments that differ on
    asymmetric ports (mux data vs select, subtractor minuend vs
    subtrahend) produce distinct keys.

    Examples:
        AND(a=X, b=Y) and AND(a=Y, b=X)  -> same key  (AND is commutative)
        SUB(a=X, b=Y) and SUB(a=Y, b=X)  -> different keys (a-b != b-a)
        MUX(a=X, b=Y, sel=S) vs MUX(a=Y, b=X, sel=S) -> different keys
    """
    used_ports: set[str] = set()
    parts: list[tuple] = []

    # Sort groups by their canonical port set so group ordering is
    # deterministic across runs.
    for group in sorted(
        component.commutative_groups, key=lambda g: tuple(sorted(g))
    ):
        # Sort ONLY the net assignments, not (port, nets) pairs. This is
        # what makes symmetric permutations collapse.
        assignments = sorted(
            tuple(alignment[port])
            for port in group
            if port in alignment
        )
        parts.append(("comm", tuple(sorted(group)), tuple(assignments)))
        used_ports |= group

    # Non-commutative ports: preserve which port got which nets.
    for port in sorted(alignment):
        if port not in used_ports:
            parts.append(("single", port, tuple(alignment[port])))

    return (component.name, tuple(parts))


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def identify_word(
    G: nx.MultiDiGraph,
    word: Word,
    *,
    max_external_inputs: int = _MAX_EXTERNAL_INPUTS,
    max_fragment_gates: int = _MAX_FRAGMENT_GATES,
) -> list[Match]:
    """Identify the operation producing `word`.

    Returns a list of Matches (possibly empty) — canonicalized so that
    commutative port permutations yield a single unique Match per component.
    """
    if not word.bits:
        return []

    fragment = _find_fragment(G, word.bits, max_gates=max_fragment_gates)
    if fragment is None:
        log.debug("identify: %r exceeds gate limit", word)
        return []
    if len(fragment.free_vars) > max_external_inputs:
        log.debug(
            "identify: %r has %d free vars > %d; skipping",
            word, len(fragment.free_vars), max_external_inputs,
        )
        return []
    if len(fragment.interior_gates) > max_fragment_gates:
        log.debug(
            "identify: %r has %d interior gates > %d; skipping",
            word, len(fragment.interior_gates), max_fragment_gates,
        )
        return []

    groups = _group_free_vars(G, fragment.free_vars)
    k = len(fragment.free_vars)

    try:
        recovered_tt = _build_recovered_tt(
            G, word.bits, fragment.free_vars, fragment.constants
        )
    except ValueError as exc:
        log.debug("identify: simulation failed for %r: %s", word, exc)
        return []

    matches: list[Match] = []
    seen_canonical_keys: set[tuple] = set()

    for component in candidates_for_width(word.width):
        if component.total_input_bits != k:
            continue
        for alignment in _alignments(component, groups, fragment.free_vars):
            if _verify_alignment(
                component, alignment, fragment.free_vars, recovered_tt
            ):
                key = _canonical_alignment_key(component, alignment)
                if key not in seen_canonical_keys:
                    seen_canonical_keys.add(key)
                    matches.append(Match(component=component, alignment=alignment))
    return matches


def identify_words(
    G: nx.MultiDiGraph,
    words: list[Word],
    **kwargs,
) -> dict[int, list[Match]]:
    """Identify every Word, keyed by index in the input list."""
    return {i: identify_word(G, w, **kwargs) for i, w in enumerate(words)}