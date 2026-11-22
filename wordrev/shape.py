# wordrev/shape.py
"""
Shape hashing: k-bounded backward structural fingerprints for nets.

Uses Weisfeiler-Lehman (WL) style iterative refinement on the bipartite
graph, restricted to backward (predecessor) traversal. After k rounds,
each net's label captures its k-bounded backward cone up to isomorphism
(within hash collisions).

Leaf nodes (primary inputs, constants, and any net with no driver) keep
their initial label forever. They are the structural "atoms" of a cone.
"""
from __future__ import annotations

import hashlib

import networkx as nx

from .graph import Node

# Initial label for non-constant nets.
NET_LABEL = "net"

# Prefix for constant net labels; the constant value is appended.
CONST_PREFIX = "const_"


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def compute_all_shape_hashes(
    G: nx.MultiDiGraph,
    k: int = 3,
    *,
    include_pins: bool = True,
) -> dict[Node, str]:
    """Compute the k-bounded backward shape hash for every net in `G`.

    Parameters
    ----------
    G :
        Bipartite graph from `build_graph`.
    k :
        Number of WL refinement rounds. Each round is two bipartite
        transitions (gate update, then net update), so k rounds see
        approximately 2*k hops back in the netlist. Do not confuse with
        the paper's k, which counts gate hops only; a paper-k=3 maps to
        roughly this-k=2. If goal is to reproduce WordRev's exact
        grouping on a benchmark, use k=2 as the starting point and
        calibrate against the WISCAD reference files.
    include_pins :
        If True, the hash distinguishes which input pin a signal feeds.
        Net-side hashes always include the output pin name (e.g., Q vs QN
        on a flip-flop), since different output pins compute different
        functions. So include_pins=False means "invariant under input-pin
        permutation," not "fully pin-agnostic."

    Returns
    -------
    A dict mapping each net node key to a 32-hex-char hash string.
    Gates are not included — only nets have shape hashes.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    # --- Initial labels ---
    labels: dict[Node, str] = {}
    is_leaf: dict[Node, bool] = {}

    for node in G.nodes:
        kind = node[0]
        if kind == "net":
            data = G.nodes[node]
            if data.get("is_constant"):
                labels[node] = f"{CONST_PREFIX}{data.get('constant_value')}"
            else:
                labels[node] = NET_LABEL
            is_leaf[node] = not _has_gate_driver(G, node)
        else:  # gate
            labels[node] = str(G.nodes[node]["cell_type"])
            is_leaf[node] = False

    # --- k rounds of bipartite WL refinement ---
    for _ in range(k):
        # 1. Update Gate labels using upstream input Nets
        new_gate_labels: dict[Node, str] = {}
        for node in G.nodes:
            if node[0] != "gate":
                continue
            
            neighbor_pairs: list[tuple[str, str]] = []
            for pred in G.predecessors(node):
                if pred[0] != "net":
                    continue
                for edge_attrs in G[pred][node].values():
                    if edge_attrs.get("role") in ("input", "inout"):
                        pin = edge_attrs.get("pin", "") if include_pins else ""
                        neighbor_pairs.append((pin, labels[pred]))
            
            neighbor_pairs.sort()
            new_gate_labels[node] = _hash_label(labels[node], neighbor_pairs)

        # Apply gate label updates
        labels.update(new_gate_labels)

        # 2. Update Net labels using driving Gates
        new_net_labels: dict[Node, str] = {}
        for node in G.nodes:
            if node[0] != "net":
                continue
            if is_leaf[node]:
                new_net_labels[node] = labels[node]
                continue

            neighbor_pairs = []
            for pred in G.predecessors(node):
                if pred[0] != "gate":
                    continue
                for edge_attrs in G[pred][node].values():
                    if edge_attrs.get("role") in ("output", "inout"):
                        pin = edge_attrs.get("pin", "") if include_pins else ""
                        neighbor_pairs.append((pin, labels[pred]))

            neighbor_pairs.sort()
            new_net_labels[node] = _hash_label(labels[node], neighbor_pairs)

        # Apply net label updates
        labels.update(new_net_labels)

    # Return shape hashes for net nodes only
    return {node: labels[node] for node in G.nodes if node[0] == "net"}


def shape_hash(
    G: nx.MultiDiGraph,
    net: Node,
    k: int = 3,
    *,
    include_pins: bool = True,
) -> str:
    """Convenience wrapper: shape hash for a single net.

    Recomputes all labels on every call. Use `compute_all_shape_hashes`
    when you need more than one net: the batch version is O(k * |E|)
    for any number of nets, while this wrapper is O(k * |E|) per call.
    """
    if net[0] != "net":
        raise ValueError(f"shape_hash requires a net node, got {net!r}")
    return compute_all_shape_hashes(G, k, include_pins=include_pins)[net]


def backward_cone(
    G: nx.MultiDiGraph,
    start: Node,
    k: int,
) -> set[tuple[Node, int]]:
    """Return all (node, gate_depth) pairs backward-reachable from `start`.

    Depth counts gate transitions: `start` is depth 0, driving gates are
    depth 1, nets driving those gates are depth 1, the gates driving those
    nets are depth 2, etc.
    """
    seen: set[tuple[Node, int]] = set()
    frontier: list[tuple[Node, int]] = [(start, 0)]

    while frontier:
        node, depth = frontier.pop()
        if depth > k or (node, depth) in seen:
            continue
        seen.add((node, depth))

        for pred in G.predecessors(node):
            # Increment depth only when transitioning net -> gate
            next_depth = depth + 1 if node[0] == "net" and pred[0] == "gate" else depth
            if next_depth <= k:
                frontier.append((pred, next_depth))

    return seen


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _has_gate_driver(G: nx.MultiDiGraph, net: Node) -> bool:
    """True if some gate output or inout drives this net."""
    for pred in G.predecessors(net):
        if pred[0] == "gate":
            for attrs in G[pred][net].values():
                if attrs.get("role") in ("output", "inout"):
                    return True
    return False


def _hash_label(base: str, neighbors: list[tuple[str, str]]) -> str:
    """Hash a node's label with its sorted neighbor labels via BLAKE2b."""
    payload = base + "|" + "|".join(f"{pin}={label}" for pin, label in neighbors)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()