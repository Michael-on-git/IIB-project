# wordrev/shape_unroll.py
"""
Explicit k-bounded backward-cone shape hashing (original WordRev style).

Alternative to shape.compute_all_shape_hashes (WL-based). Builds the
backward-reachable subgraph within k steps and hashes its structure as
a depth-stratified node histogram plus edge histogram.

Difference from WL:
    WL refines labels iteratively and captures the "k-hop neighborhood
    up to label equivalence." It cannot distinguish sharing from
    non-sharing: AND(a, a) and AND(a, b) both look like "gate with two
    net-labeled inputs." The explicit method retains node uniqueness,
    so AND(a, a) has 3 unique nodes at depth 2 (a appears once, two
    edges to AND) while AND(a, b) has 4 (a and b are distinct).

    For cycle-containing netlists, WL converges to a fixed point;
    the explicit method uses (node, depth) occurrences so a node on a
    cycle appears at every depth up to k. These are genuinely different
    semantics; the explicit method reflects the paper's "unroll the
    loops by duplicating gates" description more literally.

Signature-compatible with compute_all_shape_hashes so it can be
substituted via the `shape_method` parameter of run_pipeline.
"""
from __future__ import annotations

import hashlib
from collections import deque

import networkx as nx

from .graph import Node


def compute_all_shape_hashes_unrolled(
    G: nx.MultiDiGraph,
    k: int = 3,
    *,
    include_pins: bool = True,
) -> dict[Node, str]:
    """k-bounded backward-cone hash for every net in G."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    return {
        node: _cone_hash(G, node, k, include_pins)
        for node in G.nodes
        if node[0] == "net"
    }


def _node_kind(G: nx.MultiDiGraph, node: Node) -> str:
    data = G.nodes[node]
    if data.get("is_constant"):
        return f"const:{data.get('constant_value')}"
    if node[0] == "gate":
        return f"gate:{data.get('cell_type')}"
    return "net"


def _canonical_net_key(G: nx.MultiDiGraph, node: Node) -> tuple:
    """Stable sort key for canonical leaf ID assignment.

    Preference:
        1. (base_name, vector_index) for named vector bits
        2. (kind, 0, bit_id) for named-but-not-vector nets
        3. (kind, 1, str(bit_id)) for string bit IDs

    Using base_name + vector_index means two bit slices (a[0],b[0])
    and (a[1],b[1]) have the same relative sort order *within their
    cone*, regardless of which bit IDs Yosys happened to assign. This
    is what makes the canonical leaf ID scheme robust to bit-ID
    assignment — otherwise the test happens to pass only when bit IDs
    are monotonically assigned per slice.
    """
    data = G.nodes[node]
    base = data.get("base_name")
    idx = data.get("vector_index")
    if base is not None and idx is not None:
        return (0, base, idx)
    kind, bit_id = node
    if isinstance(bit_id, int) and not isinstance(bit_id, bool):
        return (1, kind, 0, bit_id)
    return (2, kind, 1, str(bit_id))


def _has_driver(G: nx.MultiDiGraph, net: Node) -> bool:
    for pred in G.predecessors(net):
        if pred[0] == "gate":
            return True
    return False


def _cone_hash(G, start, k, include_pins):
    """Canonical k-bounded cone hash with sharing preserved.

    Two cones that differ only in *which* nets feed them (bit slices)
    hash the same. Two cones that differ in *how many distinct leaves*
    feed a gate (AND(a,a) vs AND(a,b)) hash differently.
    """
    # Pass 1: identify leaves within the cone (primary inputs reachable in k hops)
    leaves: set[Node] = set()
    seen: set[tuple[Node, int]] = {(start, 0)}
    frontier = deque([(start, 0)])
    while frontier:
        node, depth = frontier.popleft()
        if depth >= k:
            continue
        for pred in G.predecessors(node):
            key = (pred, depth + 1)
            if key in seen:
                continue
            seen.add(key)
            frontier.append(key)
            if pred[0] != "net":
                continue
            if G.nodes[pred].get("is_constant"):
                continue  # constants hashed by value, not canonical ID
            if not _has_driver(G, pred):
                leaves.add(pred)

    # Pass 2: canonical IDs for leaves
    leaf_ids = {
    leaf: i
    for i, leaf in enumerate(
        sorted(leaves, key=lambda n: _canonical_net_key(G, n))
    )
}

    # Pass 3: recursive hash using leaf IDs
    cache: dict[tuple[Node, int], str] = {}

    def _hash(node: Node, depth: int) -> str:
        if node[0] == "net":
            data = G.nodes[node]
            if data.get("is_constant"):
                return f"const:{data.get('constant_value')}"
            if node in leaf_ids:
                return f"leaf:{leaf_ids[node]}"
        if depth >= k:
            return f"kbound:{_node_kind(G, node)}"
        key = (node, depth)
        if key in cache:
            return cache[key]
        kind = _node_kind(G, node)
        child_sigs = []
        for pred in G.predecessors(node):
            for attrs in G[pred][node].values():
                pin = attrs.get("pin", "") if include_pins else ""
                role = attrs.get("role", "")
                child_sigs.append((pin, role, _hash(pred, depth + 1)))
        child_sigs.sort()
        payload = kind + "|" + "|".join(f"{p}:{r}:{h}" for p, r, h in child_sigs)
        result = hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()
        cache[key] = result
        return result

    return _hash(start, 0)