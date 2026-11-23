# wordrev/pipeline.py
"""
End-to-end pipeline: Yosys JSON → recovered word-level description.

Orchestrates six steps, grouped as five conceptual stages in the write-up
(Stage 1 = parse + graph) but kept as two calls here for clarity:

    Step 1  parse      (parser.parse_yosys_json)            Stage 1a
    Step 2  graph      (graph.build_graph)                  Stage 1b
    Step 3  shape      (shape.compute_all_shape_hashes)     Stage 2
    Step 4  slice      (slice.compute_functional_signatures) Stage 3
    Step 5  aggregate  (aggregate.aggregate_words)          Stage 4
    Step 6  identify   (identify.identify_words)            Stage 5

Steps 3–4 results are optional and can be passed in to amortize cost
across multiple aggregations (e.g. sweeping shape_k).

The CLI in wordrev/__main__.py is the primary consumer. Programmatic
callers should use `run_pipeline` directly. Human-readable formatting is
provided by `format_human`; machine-readable formatting by `format_json`.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from .aggregate import Word, aggregate_words
from .graph import Node, build_graph, node_label
from .identify import Match, identify_words
from .parser import parse_yosys_json
from .shape import compute_all_shape_hashes
from .slice import compute_functional_signatures
from .shape_unroll import compute_all_shape_hashes_unrolled

SHAPE_METHODS = {
    "wl": compute_all_shape_hashes,
    "unroll": compute_all_shape_hashes_unrolled,
}

# -----------------------------------------------------------------------------
# Result dataclasses
# -----------------------------------------------------------------------------

@dataclass
class WordRecovery:
    """A recovered Word plus whatever identification matched it."""
    word: Word
    matches: list[Match] = field(default_factory=list)

    @property
    def identified(self) -> bool:
        return bool(self.matches)

    def bit_labels(self, G: nx.MultiDiGraph) -> list[str]:
        """Human-readable labels for the Word's bits, LSB first."""
        return [node_label(G, node) for node in self.word.bits]

    def port_labels(self, G: nx.MultiDiGraph) -> dict[str, list[str]]:
        """For each matched component, the alignment as human-readable labels.

        Requires the graph because vector names live on node attributes,
        not on the Match itself.
        """
        result: dict[str, list[str]] = {}
        for match in self.matches:
            ports: list[str] = []
            for port_name, nets in match.alignment.items():
                bits = [node_label(G, n) for n in nets]
                ports.append(f"{port_name}<-[{', '.join(bits)}]")
            result[match.component.name] = ports
        return result


@dataclass
class RecoveryResult:
    """Everything the pipeline recovered from one netlist."""
    module_name: str
    words: list[WordRecovery]
    graph: nx.MultiDiGraph
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def num_words(self) -> int:
        return len(self.words)

    @property
    def num_identified(self) -> int:
        return sum(1 for w in self.words if w.identified)

    def summary(self) -> str:
        return (
            f"{self.num_words} words recovered, "
            f"{self.num_identified} identified"
        )


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

def run_pipeline(
    netlist_path: str | Path,
    *,
    top: str | None = None,
    shape_k: int = 3,
    min_width: int = 2,
    propagate: bool = True,
    shape_method: str = "wl",
    identify: bool = True,
    max_external_inputs: int = 12,
    max_fragment_gates: int = 500,
    shape_hashes: dict[Node, str] | None = None,
    signatures: dict[Node, frozenset[tuple[int, int]]] | None = None,
) -> RecoveryResult:
    """Run the full pipeline on a Yosys JSON netlist.

    See module docstring for the stage structure and the CLI in
    wordrev/__main__.py for flag semantics. This function does not print.
    """
    path = Path(netlist_path)
    if not path.exists():
        raise FileNotFoundError(f"netlist not found: {path}")

    t0 = time.perf_counter()
    netlist = parse_yosys_json(path)
    t_parse = time.perf_counter() - t0

    if not netlist.modules:
        raise ValueError(f"netlist {path} contains no modules")

    if top is None:
        module_name = next(iter(netlist.modules))
    else:
        if top not in netlist.modules:
            raise KeyError(
                f"top module {top!r} not in netlist; "
                f"available: {list(netlist.modules)}"
            )
        module_name = top

    module = netlist.modules[module_name]

    t0 = time.perf_counter()
    G = build_graph(module)
    t_graph = time.perf_counter() - t0

    t0 = time.perf_counter()
    if shape_hashes is None:
        shape_hashes = SHAPE_METHODS[shape_method](G, k=shape_k)
    if signatures is None:
        signatures = compute_functional_signatures(G)
    t_analysis = time.perf_counter() - t0

    t0 = time.perf_counter()
    words = aggregate_words(
        G,
        shape_hashes=shape_hashes,
        signatures=signatures,
        min_width=min_width,
        propagate=propagate,
        shape_k=shape_k,
    )
    t_aggregate = time.perf_counter() - t0

    t0 = time.perf_counter()
    if identify:
        matches_by_index = identify_words(
            G,
            words,
            max_external_inputs=max_external_inputs,
            max_fragment_gates=max_fragment_gates,
        )
    else:
        matches_by_index = {i: [] for i in range(len(words))}
    t_identify = time.perf_counter() - t0

    recoveries = [
        WordRecovery(word=w, matches=matches_by_index[i])
        for i, w in enumerate(words)
    ]

    return RecoveryResult(
        module_name=module_name,
        words=recoveries,
        graph=G,
        stats={
            "parse_seconds": t_parse,
            "graph_seconds": t_graph,
            "analysis_seconds": t_analysis,
            "aggregate_seconds": t_aggregate,
            "identify_seconds": t_identify,
            "total_seconds": (
                t_parse + t_graph + t_analysis + t_aggregate + t_identify
            ),
            "num_gates": len(module.gates),
            "num_nets": len(module.nets),
            "num_ports": len(module.ports),
        },
    )


# -----------------------------------------------------------------------------
# Human-readable formatting
# -----------------------------------------------------------------------------

def _format_word(wr: WordRecovery, result: RecoveryResult, index: int) -> str:
    """One Word as a multi-line block for human output.

    Handles three cases for the bit-range display:
        width 0  ->  "(empty)"
        width 1  ->  the single label (no "x..x" repetition)
        width ≥2 ->  "{first}..{last}"
    """
    labels = wr.bit_labels(result.graph)
    width = wr.word.width

    if width == 0:
        bit_range = "(empty)"
    elif width == 1:
        bit_range = labels[0]
    else:
        bit_range = f"{labels[0]}..{labels[-1]}"

    lines = [
        f"  [{index}] {bit_range}   "
        f"width={width}  source={wr.word.source}  conf={wr.word.confidence:.2f}",
    ]
    if wr.matches:
        for match in wr.matches:
            port_labels = wr.port_labels(result.graph)[match.component.name]
            lines.append(
                f"      = {match.component.name}  ({' '.join(port_labels)})"
            )
    else:
        lines.append("      (no library match)")
    return "\n".join(lines)

def format_human(result: RecoveryResult, *, include_stats: bool = False) -> str:
    """Human-readable report."""
    stats = result.stats
    header = (
        f"Module: {result.module_name}  "
        f"(gates={stats['num_gates']}, nets={stats['num_nets']}, "
        f"ports={stats['num_ports']})"
    )
    body = [header, ""]

    if not result.words:
        body.append("No words recovered.")
    else:
        body.append(
            f"Words recovered: {len(result.words)} "
            f"({result.num_identified} identified)"
        )
        body.append("")
        for i, wr in enumerate(result.words):
            body.append(_format_word(wr, result, i))
        body.append("")

    if include_stats:
        body.append("Timing:")
        for key in ("parse_seconds", "graph_seconds", "analysis_seconds",
                    "aggregate_seconds", "identify_seconds", "total_seconds"):
            body.append(f"  {key:<22} {stats[key]:.3f}s")

    return "\n".join(body)


# -----------------------------------------------------------------------------
# Machine-readable formatting
# -----------------------------------------------------------------------------

def format_json(result: RecoveryResult) -> str:
    """Machine-readable JSON output.

    Bit ordering is LSB-first, matching the Word and bit_labels ordering.
    Stats floats are rounded to 4 decimals for stable diffs.
    """
    def node_id(node: Node) -> object:
        return node[1]

    words_out = []
    for i, wr in enumerate(result.words):
        word_out = {
            "index": i,
            "width": wr.word.width,
            "source": wr.word.source,
            "confidence": round(wr.word.confidence, 4),
            "bits": [node_id(n) for n in wr.word.bits],
            "bit_labels": wr.bit_labels(result.graph),
            "matches": [],
        }
        for match in wr.matches:
            alignment_out = {}
            for port, nets in match.alignment.items():
                alignment_out[port] = {
                    "nets": [node_id(n) for n in nets],
                    "labels": [node_label(result.graph, n) for n in nets],
                }
            word_out["matches"].append({
                "component": match.component.name,
                "alignment": alignment_out,
            })
        words_out.append(word_out)

    return json.dumps(
        {
            "module": result.module_name,
            "stats": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in result.stats.items()
            },
            "words": words_out,
        },
        indent=2,
    )