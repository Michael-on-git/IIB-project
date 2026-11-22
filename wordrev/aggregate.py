# wordrev/aggregate.py
"""
Word aggregation: combine shape hashes and slice signatures into ordered
multi-bit words.

A Word is an ordered list of nets, indexed from LSB (bits[0]) to MSB
(bits[-1]). Ordering is what distinguishes a word from an arbitrary set
of nets: sum[0] and sum[3] belong to the same word, but sum[3] is not
"sum[0] + 3" — their positions are determined by carry propagation or
declared vector index, not by bit ID.

Pipeline:
    1. Cluster nets by (shape_hash, functional_signature). Every signature
       that appears at least `min_width` times within a shape bucket yields
       a cluster; smaller groups are dropped.
    2. Order bits within each cluster. Carry-chain depth is tried first
       (semantic for datapaths); if absent, ordering falls back to cone
       size with (base_name, vector_index) as a semantic tiebreak.
    3. Propagate words across parallel bit-sliced gates (one BUF, NOT, or
       single-bit slice of a multi-output gate per bit). Propagated words
       are less trustworthy than clustered ones and should be independently
       verified before being treated as buses.
    4. Resolve overlaps by preferring higher-confidence words.

Confidence:
    cluster_carry      1.0    bits ordered by carry chain (semantic)
    cluster_structural 0.5    bits ordered heuristically (best-effort)
    propagated         0.9^k  propagated k rounds from a cluster word

Vector-index ordering depends on the parser populating `Net.base_name`
and `Net.vector_index`. If those are not set, this module falls back to
bit-ID ordering — deterministic, but arbitrary with respect to the source
declaration.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass

import networkx as nx

from .graph import Node
from .shape import compute_all_shape_hashes
from .slice import compute_functional_signatures

log = logging.getLogger(__name__)

# Cap for cone-size and carry-depth BFS. Full traversal is O(cone) per
# net, which becomes quadratic over a large cluster. Ordering needs only
# a monotone discriminator, so capping at a few hundred nodes is sufficient
# and keeps the total cost linear in cluster size.
_CONE_SIZE_CAP = 500


# -----------------------------------------------------------------------------
# Sorting keys
# -----------------------------------------------------------------------------

def _net_sort_key(node: Node) -> tuple:
    """Stable sort key from the node's bit ID alone.

    Integer IDs sort numerically (2 before 10). String IDs sort
    lexicographically after all integers. Independent of bit-ID
    assignment, so two isomorphic cones produce the same relative order
    regardless of Yosys internals. This is the fallback when no vector
    index is available.
    """
    kind, bit_id = node
    if isinstance(bit_id, int) and not isinstance(bit_id, bool):
        return (kind, 0, bit_id)
    return (kind, 1, str(bit_id))


def _bit_sort_key(node: Node, G: nx.MultiDiGraph) -> tuple:
    """Semantic LSB→MSB sort key.

    Priority:
        1. base_name + vector_index — groups bits of the same bus and
           orders them by declared index (the only signal available when
           no carry chain exists, e.g. data_in[7:0] from a port).
        2. bit_id — fallback for nets with no vector metadata.

    Depends on the parser populating Net.base_name and Net.vector_index
    and the graph builder copying them onto the node. If they are absent,
    the fallback preserves determinism but loses semantic ordering.
    """
    data = G.nodes.get(node, {})
    base = data.get("base_name")
    idx = data.get("vector_index")
    if base is not None and idx is not None:
        return (0, base, idx)
    return (1, "", _net_sort_key(node))


# -----------------------------------------------------------------------------
# Word dataclass
# -----------------------------------------------------------------------------

@dataclass
class Word:
    """An ordered multi-bit grouping, LSB at index 0."""
    bits: list[Node]
    shape_hash: str | None = None
    signature: tuple[int, int] | None = None
    confidence: float = 1.0
    source: str = "cluster_carry"

    @property
    def width(self) -> int:
        return len(self.bits)

    def __repr__(self) -> str:
        return (f"Word(width={self.width}, conf={self.confidence:.2f}, "
                f"src={self.source}, bits={self.bits})")


# -----------------------------------------------------------------------------
# Step 1: Candidate clustering
# -----------------------------------------------------------------------------

def _cluster_candidates(
    shape_hashes: dict[Node, str],
    signatures: dict[Node, frozenset[tuple[int, int]]],
    *,
    min_width: int = 2,
) -> list[tuple[frozenset[Node], str, tuple[int, int]]]:
    """Cluster nets that share BOTH a shape hash AND a functional signature.

    Returns one cluster per signature that appears at least `min_width`
    times within a shape bucket. A net whose signature set contains two
    qualifying signatures appears in two clusters; `resolve_overlaps`
    handles the overlap later.

    Clusters smaller than `min_width` are dropped. Only the qualifying
    signatures are considered — signatures that do not meet `min_width`
    within a shape bucket are silently discarded, which is correct
    because they do not form a word of the minimum size.
    """
    by_shape: dict[str, list[Node]] = defaultdict(list)
    for net, h in shape_hashes.items():
        by_shape[h].append(net)

    clusters: list[tuple[frozenset[Node], str, tuple[int, int]]] = []

    for shape_hash, nets in by_shape.items():
        if len(nets) < min_width:
            continue

        sig_counter: Counter[tuple[int, int]] = Counter()
        for net in nets:
            for sig in signatures.get(net, frozenset()):
                sig_counter[sig] += 1

        for sig, count in sig_counter.items():
            if count < min_width:
                continue
            members = frozenset(
                net for net in nets if sig in signatures.get(net, frozenset())
            )
            if len(members) < min_width:
                continue
            clusters.append((members, shape_hash, sig))

    return clusters


# -----------------------------------------------------------------------------
# Step 2: Bit ordering
# -----------------------------------------------------------------------------

def _cone_sizes(
    G: nx.MultiDiGraph,
    nets: list[Node],
    *,
    cap: int = _CONE_SIZE_CAP,
) -> dict[Node, int]:
    """Backward-cone size for each net, capped at `cap`.

    Returned sizes are a monotone discriminator, not exact sizes: a net
    whose full cone exceeds `cap` reports `cap` but is still correctly
    ordered relative to nets with smaller cones.
    """
    sizes: dict[Node, int] = {}
    for net in nets:
        seen: set[Node] = set()
        frontier = [net]
        while frontier and len(seen) < cap:
            curr = frontier.pop()
            if curr in seen:
                continue
            seen.add(curr)
            frontier.extend(G.predecessors(curr))
        sizes[net] = len(seen)
    return sizes


def _carry_chain_order(
    G: nx.MultiDiGraph,
    cluster: frozenset[Node],
) -> list[Node] | None:
    """Order cluster members by carry-chain dependency depth.

    A carry chain exists when some cluster members appear in other
    members' backward cones. The bit whose cone contains the fewest other
    cluster members is the LSB; the one whose cone contains the most is
    the MSB. Returns None if all members have the same dependency count
    (no chain detected).

    Ties are broken by `_bit_sort_key`, which prefers vector index over
    bit ID — important when two bits have equal carry depth (e.g. a
    signed-carry adder where the top two bits have identical structure).
    """
    if len(cluster) < 2:
        return None

    depths: dict[Node, int] = {}
    for net_a in cluster:
        deps = 0
        seen: set[Node] = set()
        frontier = list(G.predecessors(net_a))
        while frontier and len(seen) < _CONE_SIZE_CAP:
            curr = frontier.pop()
            if curr in seen:
                continue
            seen.add(curr)
            frontier.extend(G.predecessors(curr))
        for other in cluster:
            if other != net_a and other in seen:
                deps += 1
        depths[net_a] = deps

    if len(set(depths.values())) < 2:
        return None

    return sorted(cluster, key=lambda n: (depths[n], _bit_sort_key(n, G)))


def _structural_order(
    G: nx.MultiDiGraph,
    cluster: frozenset[Node],
    cone_sizes: dict[Node, int],
) -> list[Node]:
    """Fallback ordering: by backward-cone size, then by vector index.

    Larger cone = deeper in the logic = typically a higher bit of a word,
    since higher bits depend on lower bits' carry logic. When cone sizes
    tie (parallel bus bits with no intermediate logic), `_bit_sort_key`
    decides — vector index for named buses, bit ID otherwise.
    """
    return sorted(
        cluster,
        key=lambda n: (cone_sizes.get(n, 0), _bit_sort_key(n, G)),
    )


def order_bits(
    G: nx.MultiDiGraph,
    cluster: frozenset[Node],
) -> tuple[list[Node], str]:
    """Order cluster bits LSB→MSB.

    Returns (ordered_bits, ordering_kind) where ordering_kind is 'carry'
    (semantic ordering via carry-chain depth) or 'structural' (heuristic
    fallback). The caller uses the kind to set confidence.

    Cone sizes are computed lazily: if a carry chain is found, they are
    never needed. If not, they are computed once and passed to
    `_structural_order`.
    """
    ordered = _carry_chain_order(G, cluster)
    if ordered is not None:
        return ordered, "carry"
    cone_sizes = _cone_sizes(G, list(cluster))
    return _structural_order(G, cluster, cone_sizes), "structural"


# -----------------------------------------------------------------------------
# Step 3: Word propagation across bit-sliced gates
# -----------------------------------------------------------------------------

def propagate_words(
    G: nx.MultiDiGraph,
    words: list[Word],
    *,
    max_rounds: int = 10,
) -> list[Word]:
    """Extend words through bit-sliced gates.

    For each word, if every bit feeds a gate of the same cell type on the
    same input pin and produces output on the same output pin, the
    corresponding output nets form a propagated word. Both single-output
    gates ($_BUF_, $_NOT_) and multi-output gates (full adders emitting
    SUM and COUT) are handled by matching on (in_pin, cell_type, out_pin).

    Propagated words have lower confidence than clustered words and can be
    false positives: two unrelated control AND gates that happen to share
    a cell type and each receive one bit of a data word will look like a
    propagated word. Stage 5's function identification should verify
    propagated words before treating them as buses.
    """
    all_words = list(words)
    seen_keys: set[tuple[Node, ...]] = {tuple(w.bits) for w in words}

    for _ in range(max_rounds):
        new_words: list[Word] = []

        for word in all_words:
            if not word.bits:
                continue

            bit_options: list[list[tuple[str, str, str, Node]]] = []
            for bit in word.bits:
                options: list[tuple[str, str, str, Node]] = []
                for gate in G.successors(bit):
                    if gate[0] != "gate":
                        continue
                    cell_type = str(G.nodes[gate].get("cell_type", ""))
                    for in_attrs in G[bit][gate].values():
                        if in_attrs.get("role") != "input":
                            continue
                        in_pin = in_attrs["pin"]
                        for out_net in G.successors(gate):
                            if out_net[0] != "net":
                                continue
                            for out_attrs in G[gate][out_net].values():
                                if out_attrs.get("role") == "output":
                                    options.append(
                                        (in_pin, cell_type, out_attrs["pin"], out_net)
                                    )
                bit_options.append(options)

            if not all(bit_options):
                continue

            # Try each anchor (in_pin, cell_type, out_pin) from bit 0.
            for in_pin0, type0, out_pin0, _ in bit_options[0]:
                out_bits: list[Node] = []
                valid = True

                for options in bit_options:
                    matches = [
                        out_net for (ip, t, op, out_net) in options
                        if ip == in_pin0 and t == type0 and op == out_pin0
                    ]
                    if len(matches) != 1:
                        valid = False
                        break
                    out_bits.append(matches[0])

                if valid and len(set(out_bits)) == len(out_bits):
                    key = tuple(out_bits)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        new_words.append(Word(
                            bits=out_bits,
                            shape_hash=None,
                            signature=None,
                            confidence=word.confidence * 0.9,
                            source="propagated",
                        ))

        if not new_words:
            break
        all_words.extend(new_words)

    return all_words


# -----------------------------------------------------------------------------
# Step 4: Overlap resolution
# -----------------------------------------------------------------------------

def resolve_overlaps(words: list[Word]) -> list[Word]:
    """Greedily accept words by (confidence, width), skipping overlaps.

    Words are sorted by confidence descending, then width descending, so a
    carry-ordered wide word beats a structural narrow one when they claim
    the same net. Nets claimed by an accepted word cannot be reused; a
    lower-ranked word that shares any net is dropped.

    Ties are broken by the stable insertion order (Python's sort is stable),
    so `aggregate_words`'s cluster order determines which of two
    equal-confidence, equal-width words survives. That order is
    deterministic given the input data.
    """
    sorted_words = sorted(
        words, key=lambda w: (w.confidence, w.width), reverse=True
    )
    claimed: set[Node] = set()
    accepted: list[Word] = []
    for word in sorted_words:
        bits = set(word.bits)
        if bits & claimed:
            continue
        claimed |= bits
        accepted.append(word)
    return accepted


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def aggregate_words(
    G: nx.MultiDiGraph,
    *,
    shape_hashes: dict[Node, str] | None = None,
    signatures: dict[Node, frozenset[tuple[int, int]]] | None = None,
    min_width: int = 2,
    propagate: bool = True,
    shape_k: int = 3,
) -> list[Word]:
    """Full aggregation pipeline: cluster → order → propagate → resolve.

    If `shape_hashes` or `signatures` are None, they are computed from G.
    Callers can pass precomputed values to amortize the cost across
    multiple runs (e.g. sweeping shape_k without recomputing signatures).
    """
    if shape_hashes is None:
        # Direct callers of this function get WL. The pipeline computes
        # shape hashes via SHAPE_METHODS[shape_method] first, so its
        # choice of method is honoured there.
        shape_hashes = compute_all_shape_hashes(G, k=shape_k)
    if signatures is None:
        signatures = compute_functional_signatures(G)

    clusters = _cluster_candidates(shape_hashes, signatures, min_width=min_width)

    words: list[Word] = []
    for members, shape_hash, sig in clusters:
        ordered, ordering_kind = order_bits(G, members)
        confidence = 1.0 if ordering_kind == "carry" else 0.5
        words.append(Word(
            bits=ordered,
            shape_hash=shape_hash,
            signature=sig,
            confidence=confidence,
            source=f"cluster_{ordering_kind}",
        ))

    if propagate:
        words = propagate_words(G, words)

    return resolve_overlaps(words)