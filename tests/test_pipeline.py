# tests/test_pipeline.py
"""End-to-end pipeline tests.

Runs the full Stage 1–5 pipeline on synthetic Yosys-style JSON via
`tmp_path`. No real Yosys dependency; no fixture files.

Coverage:
  - Happy paths: word recovery, identification, stats
  - Error paths: missing file, empty netlist, unknown top
  - Human formatting: header, word blocks, stats, empty word, single-bit word
  - JSON formatting: schema, labels, valid JSON
  - CLI: subprocess invocation, --json, --stats, --no-identify, error codes
  - WordRecovery helpers: bit_labels, port_labels use vector names
  - Baseline: full adder recovers 0 words; 2-bit AND recovers AND word
  - Semantic verification: hand-checked truth tables for AND and mux
  - Structural invariants (parametrized): real nodes, unique bits, width
  - Determinism: in-process and cross-process identical output
  - Pipeline plumbing: identify=False preserves words; precomputed matches auto
  - Confidence semantics: source↔confidence agreement
"""
import json
import subprocess
import sys

import pytest

from wordrev.aggregate import Word
from wordrev.graph import build_graph, net_node
from wordrev.models import Module
from wordrev.parser import parse_yosys_json
from wordrev.pipeline import (
    RecoveryResult,
    WordRecovery,
    format_human,
    format_json,
    run_pipeline,
)
from wordrev.shape import compute_all_shape_hashes
from wordrev.slice import _evaluate_net, compute_functional_signatures


# -----------------------------------------------------------------------------
# Helpers: netlist builders
# -----------------------------------------------------------------------------

def write_json_netlist(path, module_name, ports, cells, netnames=None):
    """Write a minimal Yosys-style JSON file."""
    data = {
        "creator": "test",
        "modules": {
            module_name: {
                "ports": ports,
                "cells": cells,
                "netnames": netnames or {},
            },
        },
    }
    path.write_text(json.dumps(data))
    return path


def make_2bit_and_netlist(path):
    """2-bit bitwise AND with annotated vector names.

    y[0] = a[0]&b[0], y[1] = a[1]&b[1].
    Inputs: a=2,3  b=4,5.  Outputs: y=10,11.

    Note: bit IDs 0 and 1 are reserved by Yosys for constants. Real wire
    IDs start from 2. Using 0/1 here would cause the parser to mark
    those nets is_constant, which breaks clustering.
    """
    ports = {
        "a": {"direction": "input",  "bits": [2, 3]},
        "b": {"direction": "input",  "bits": [4, 5]},
        "y": {"direction": "output", "bits": [10, 11]},
    }
    cells = {
        "g0": {
            "type": "$_AND_",
            "port_directions": {"A": "input", "B": "input", "Y": "output"},
            "connections": {"A": [2], "B": [4], "Y": [10]},
        },
        "g1": {
            "type": "$_AND_",
            "port_directions": {"A": "input", "B": "input", "Y": "output"},
            "connections": {"A": [3], "B": [5], "Y": [11]},
        },
    }
    netnames = {
        "a": {"hide_name": 0, "bits": [2, 3]},
        "b": {"hide_name": 0, "bits": [4, 5]},
        "y": {"hide_name": 0, "bits": [10, 11]},
    }
    return write_json_netlist(path, "and2", ports, cells, netnames)


def make_2bit_mux_netlist(path):
    """y = sel ? b : a. Inputs: a=2,3  b=4,5  sel=6.  Outputs: y=10,11."""
    ports = {
        "a":   {"direction": "input",  "bits": [2, 3]},
        "b":   {"direction": "input",  "bits": [4, 5]},
        "sel": {"direction": "input",  "bits": [6]},
        "y":   {"direction": "output", "bits": [10, 11]},
    }
    cells = {
        "m0": {
            "type": "$_MUX_",
            "port_directions": {"A": "input", "B": "input", "S": "input", "Y": "output"},
            "connections": {"A": [2], "B": [4], "S": [6], "Y": [10]},
        },
        "m1": {
            "type": "$_MUX_",
            "port_directions": {"A": "input", "B": "input", "S": "input", "Y": "output"},
            "connections": {"A": [3], "B": [5], "S": [6], "Y": [11]},
        },
    }
    netnames = {
        "a":   {"hide_name": 0, "bits": [2, 3]},
        "b":   {"hide_name": 0, "bits": [4, 5]},
        "sel": {"hide_name": 0, "bits": [6]},
        "y":   {"hide_name": 0, "bits": [10, 11]},
    }
    return write_json_netlist(path, "mux2", ports, cells, netnames)


def make_full_adder_netlist(path):
    """Single full adder.
    sum = a ^ b ^ cin;  cout = (a & b) | (cin & (a ^ b)).
    Nets: a=2 b=3 cin=4 sum=10 cout=11 axb=12 axb_and_cin=13 a_and_b=14."""
    ports = {
        "a":    {"direction": "input",  "bits": [2]},
        "b":    {"direction": "input",  "bits": [3]},
        "cin":  {"direction": "input",  "bits": [4]},
        "sum":  {"direction": "output", "bits": [10]},
        "cout": {"direction": "output", "bits": [11]},
    }
    cells = {
        "x1": {"type": "$_XOR_", "port_directions": {"A": "input", "B": "input", "Y": "output"},
               "connections": {"A": [2], "B": [3], "Y": [12]}},
        "x2": {"type": "$_XOR_", "port_directions": {"A": "input", "B": "input", "Y": "output"},
               "connections": {"A": [12], "B": [4], "Y": [10]}},
        "a1": {"type": "$_AND_", "port_directions": {"A": "input", "B": "input", "Y": "output"},
               "connections": {"A": [2], "B": [3], "Y": [14]}},
        "a2": {"type": "$_AND_", "port_directions": {"A": "input", "B": "input", "Y": "output"},
               "connections": {"A": [12], "B": [4], "Y": [13]}},
        "o1": {"type": "$_OR_", "port_directions": {"A": "input", "B": "input", "Y": "output"},
               "connections": {"A": [14], "B": [13], "Y": [11]}},
    }
    return write_json_netlist(path, "full_adder", ports, cells, {})


def make_empty_module_netlist(path):
    """A module with one unused input port, for 'no words recovered' test.
    Uses bit ID 2 (bit IDs 0,1 are reserved for constants)."""
    data = {
        "creator": "test",
        "modules": {
            "m": {
                "ports": {"x": {"direction": "input", "bits": [2]}},
                "cells": {},
                "netnames": {},
            },
        },
    }
    path.write_text(json.dumps(data))
    return path

def make_empty_stats(**overrides):
    """Stats dict with all keys present, defaulted to zero.

    Constructing a RecoveryResult by hand requires every stats key that
    format_human / format_json read. Filling them all with zeroes keeps
    those hand-constructed fixtures focused on what they actually test
    (word / label formatting) rather than on stats plumbing.
    """
    base = {
        "num_gates": 0, "num_nets": 0, "num_ports": 0,
        "parse_seconds": 0.0, "graph_seconds": 0.0, "analysis_seconds": 0.0,
        "aggregate_seconds": 0.0, "identify_seconds": 0.0, "total_seconds": 0.0,
    }
    base.update(overrides)
    return base
# -----------------------------------------------------------------------------
# Parametrized fixture: every structural invariant runs on all three circuits
# -----------------------------------------------------------------------------

@pytest.fixture(params=["and2", "mux2", "full_adder"])
def any_circuit(request, tmp_path):
    builder = {
        "and2": make_2bit_and_netlist,
        "mux2": make_2bit_mux_netlist,
        "full_adder": make_full_adder_netlist,
    }[request.param]
    path = builder(tmp_path / f"{request.param}.json")
    return run_pipeline(path, shape_k=2)


# -----------------------------------------------------------------------------
# Pipeline: happy paths
# -----------------------------------------------------------------------------

def test_pipeline_recovers_and_word(tmp_path):
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = run_pipeline(path, shape_k=2, min_width=2)

    assert isinstance(result, RecoveryResult)
    assert result.module_name == "and2"
    assert result.num_words >= 1

    identified = [
        wr for wr in result.words
        if any(m.component.name == "bitwise_and_2" for m in wr.matches)
    ]
    assert identified, f"no AND-2 word in {[w.word for w in result.words]}"


def test_pipeline_stats_are_populated(tmp_path):
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = run_pipeline(path, shape_k=2)
    assert result.stats["num_gates"] == 2
    assert result.stats["num_nets"] >= 6
    assert result.stats["total_seconds"] >= 0


def test_pipeline_top_module_selection(tmp_path):
    path = tmp_path / "two_modules.json"
    data = {
        "creator": "test",
        "modules": {
            "m1": {"ports": {}, "cells": {}, "netnames": {}},
            "m2": {"ports": {}, "cells": {}, "netnames": {}},
        },
    }
    path.write_text(json.dumps(data))
    result = run_pipeline(path, top="m2")
    assert result.module_name == "m2"


def test_pipeline_top_unknown_raises(tmp_path):
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    with pytest.raises(KeyError, match="nonexistent"):
        run_pipeline(path, top="nonexistent")


def test_pipeline_no_identify_returns_empty_matches(tmp_path):
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = run_pipeline(path, shape_k=2, identify=False)
    for wr in result.words:
        assert wr.matches == []


def test_pipeline_no_propagate(tmp_path):
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = run_pipeline(path, shape_k=2, propagate=False)
    assert all(wr.word.source != "propagated" for wr in result.words)


# -----------------------------------------------------------------------------
# Pipeline: error paths
# -----------------------------------------------------------------------------

def test_pipeline_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        run_pipeline("/nonexistent/path.json")


def test_pipeline_empty_netlist_raises(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"creator": "test", "modules": {}}))
    with pytest.raises(ValueError, match="no modules"):
        run_pipeline(path)


# -----------------------------------------------------------------------------
# Human formatting
# -----------------------------------------------------------------------------

def test_format_human_mentions_module_and_words(tmp_path):
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = run_pipeline(path, shape_k=2)
    text = format_human(result)
    assert "Module: and2" in text
    assert "Words recovered:" in text


def test_format_human_with_stats(tmp_path):
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = run_pipeline(path, shape_k=2)
    text = format_human(result, include_stats=True)
    assert "Timing:" in text
    assert "total_seconds" in text


def test_format_human_empty_words(tmp_path):
    """A netlist with nothing to cluster produces 'No words recovered'."""
    path = make_empty_module_netlist(tmp_path / "empty_mod.json")
    result = run_pipeline(path, min_width=2)
    text = format_human(result)
    assert "No words recovered" in text


def test_format_human_handles_empty_word():
    """A WordRecovery with an empty Word must not crash format_human.

    The pipeline never produces such a word today (min_width>=2,
    propagation skips empty sources), but WordRecovery is a public
    dataclass. The formatter must be total over the input type.
    """
    G = build_graph(Module("t"))
    result = RecoveryResult(
        module_name="t",
        words=[WordRecovery(word=Word(bits=[]))],
        graph=G,
        stats=make_empty_stats(),
    )
    text = format_human(result)
    assert "(empty)" in text
    assert "width=0" in text


def test_format_human_handles_single_bit_word():
    """A width-1 word prints its label once, not 'x..x'."""
    m = Module("t")
    m.get_net(5)
    m.nets[5].name = "sum[0]"
    m.nets[5].base_name = "sum"
    m.nets[5].vector_index = 0
    G = build_graph(m)
    result = RecoveryResult(
        module_name="t",
        words=[WordRecovery(word=Word(bits=[net_node(5)]))],
        graph=G,
        stats=make_empty_stats(num_nets=1),
    )
    text = format_human(result)
    assert "sum[0]" in text
    assert "sum[0]..sum[0]" not in text


# -----------------------------------------------------------------------------
# JSON formatting
# -----------------------------------------------------------------------------

def test_format_json_is_parseable(tmp_path):
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = run_pipeline(path, shape_k=2)
    text = format_json(result)
    data = json.loads(text)

    assert data["module"] == "and2"
    assert "words" in data
    assert "stats" in data


def test_format_json_word_structure(tmp_path):
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = run_pipeline(path, shape_k=2)
    data = json.loads(format_json(result))

    for w in data["words"]:
        assert "index" in w
        assert "bits" in w
        assert "matches" in w
        assert isinstance(w["bits"], list)

    and_matches = [
        m for w in data["words"] for m in w["matches"]
        if m["component"] == "bitwise_and_2"
    ]
    assert and_matches


def test_format_json_bit_labels_populated(tmp_path):
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = run_pipeline(path, shape_k=2)
    data = json.loads(format_json(result))
    for w in data["words"]:
        assert len(w["bit_labels"]) == w["width"]
        for label in w["bit_labels"]:
            assert isinstance(label, str)
            assert label


# -----------------------------------------------------------------------------
# WordRecovery helpers
# -----------------------------------------------------------------------------

def test_word_recovery_bit_labels(tmp_path):
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = run_pipeline(path, shape_k=2)
    for wr in result.words:
        labels = wr.bit_labels(result.graph)
        assert len(labels) == wr.word.width
        assert all(isinstance(label, str) and label for label in labels)


def test_word_recovery_identified_flag(tmp_path):
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = run_pipeline(path, shape_k=2)
    for wr in result.words:
        assert wr.identified == bool(wr.matches)


def test_word_recovery_port_labels_use_vector_names(tmp_path):
    """port_labels must use vector names, not raw bit IDs.

    Regression for a bug where a helper that could not see the graph
    produced 'n1' instead of 'a[0]'.
    """
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = run_pipeline(path, shape_k=2)
    and_word = next(
        wr for wr in result.words
        if any(m.component.name == "bitwise_and_2" for m in wr.matches)
    )
    ports = and_word.port_labels(result.graph)
    assert "bitwise_and_2" in ports
    for port_str in ports["bitwise_and_2"]:
        assert "[0]" in port_str or "[1]" in port_str, (
            f"expected vector labels, got: {port_str}"
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def test_cli_runs_successfully(tmp_path):
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = subprocess.run(
        [sys.executable, "-m", "wordrev", str(path), "--shape-k", "2"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Module: and2" in result.stdout


def test_cli_json_output(tmp_path):
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = subprocess.run(
        [sys.executable, "-m", "wordrev", str(path), "--shape-k", "2", "--json"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["module"] == "and2"


def test_cli_missing_file_returns_error_code():
    result = subprocess.run(
        [sys.executable, "-m", "wordrev", "/nonexistent.json"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "error" in result.stderr.lower()


def test_cli_bad_top_returns_error(tmp_path):
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = subprocess.run(
        [sys.executable, "-m", "wordrev", str(path), "--top", "nope"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "nope" in result.stderr


def test_cli_stats_flag(tmp_path):
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = subprocess.run(
        [sys.executable, "-m", "wordrev", str(path),
         "--shape-k", "2", "--stats"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Timing:" in result.stdout


def test_cli_no_identify(tmp_path):
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = subprocess.run(
        [sys.executable, "-m", "wordrev", str(path),
         "--shape-k", "2", "--no-identify"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "no library match" in result.stdout


# -----------------------------------------------------------------------------
# Baseline: correct behaviour on edge cases
# -----------------------------------------------------------------------------

def test_full_adder_recovers_zero_words(tmp_path):
    """A single full adder has no multi-bit structure. Recovering 0 words
    is the correct answer; any word recovered here would be a false positive.

    If a future aggregator change starts producing words on a full adder,
    that is a regression, not an improvement.
    """
    path = make_full_adder_netlist(tmp_path / "fa.json")
    result = run_pipeline(path, shape_k=3)
    assert result.num_words == 0, (
        f"expected 0 words on a full adder, got: "
        f"{[(wr.word.bits, wr.word.source) for wr in result.words]}"
    )


def test_2bit_and_recovers_expected_word(tmp_path):
    """A 2-bit AND recovers a word covering y[0] and y[1], identified as
    bitwise_and_2."""
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = run_pipeline(path, shape_k=2)

    assert result.num_words >= 1
    and_words = [
        wr for wr in result.words
        if any(m.component.name == "bitwise_and_2" for m in wr.matches)
    ]
    assert and_words
    and_wr = and_words[0]
    assert and_wr.word.width == 2
    assert set(and_wr.word.bits) == {net_node(10), net_node(11)}


# -----------------------------------------------------------------------------
# Semantic verification: the pipeline's claim is actually true
# -----------------------------------------------------------------------------

def test_2bit_and_hand_verified(tmp_path):
    """..."""
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = run_pipeline(path, shape_k=2)
    and_wr = next(
        wr for wr in result.words
        if any(m.component.name == "bitwise_and_2" for m in wr.matches)
    )
    match = next(m for m in and_wr.matches if m.component.name == "bitwise_and_2")

    # Alignment binds a to a[0..1] = {2,3}, b to b[0..1] = {4,5}.
    assert {n[1] for n in match.alignment["a"]} == {2, 3}
    assert {n[1] for n in match.alignment["b"]} == {4, 5}

    cases = [
        (1, 1, 1, 0, 0b01),
        (0, 1, 1, 1, 0b10),
        (1, 0, 1, 1, 0b01),
        (1, 1, 1, 1, 0b11),
        (0, 0, 1, 1, 0b00),
    ]
    for a0, a1, b0, b1, expected in cases:
        values = {
            net_node(2): a0, net_node(3): a1,
            net_node(4): b0, net_node(5): b1,
        }
        y0 = _evaluate_net(result.graph, net_node(10), values)
        y1 = _evaluate_net(result.graph, net_node(11), values)
        actual = y0 | (y1 << 1)

        a_val = a0 | (a1 << 1)
        b_val = b0 | (b1 << 1)
        library = match.component.compute({"a": a_val, "b": b_val})

        assert actual == expected
        assert library == expected


def test_2bit_mux_orientation_not_swapped(tmp_path):
    """..."""
    path = make_2bit_mux_netlist(tmp_path / "mux2.json")
    result = run_pipeline(path, shape_k=2)
    mux_wr = next(
        wr for wr in result.words
        if any(m.component.name == "mux_2" for m in wr.matches)
    )
    match = next(m for m in mux_wr.matches if m.component.name == "mux_2")

    cases = [
        (1, 0, 0, 1, 0, 0b01),
        (1, 0, 0, 1, 1, 0b10),
        (0, 1, 1, 0, 0, 0b10),
        (0, 1, 1, 0, 1, 0b01),
    ]
    for a0, a1, b0, b1, sel, expected in cases:
        values = {
            net_node(2): a0, net_node(3): a1,
            net_node(4): b0, net_node(5): b1,
            net_node(6): sel,
        }
        y0 = _evaluate_net(result.graph, net_node(10), values)
        y1 = _evaluate_net(result.graph, net_node(11), values)
        actual = y0 | (y1 << 1)

        a_val = a0 | (a1 << 1)
        b_val = b0 | (b1 << 1)
        library = match.component.compute({"a": a_val, "b": b_val, "sel": sel})

        assert actual == expected
        assert library == expected


# -----------------------------------------------------------------------------
# Structural invariants (parametrized over all circuits)
# -----------------------------------------------------------------------------

def test_every_word_bit_is_a_real_graph_node(any_circuit):
    """A word must reference existing graph nodes."""
    for wr in any_circuit.words:
        for n in wr.word.bits:
            assert n in any_circuit.graph.nodes, (
                f"word bit {n} not in graph"
            )


def test_word_bits_are_unique(any_circuit):
    """A word must not repeat a net."""
    for wr in any_circuit.words:
        assert len(set(wr.word.bits)) == wr.word.width, (
            f"word has duplicate bits: {wr.word.bits}"
        )


def test_alignment_nets_are_real_graph_nodes(any_circuit):
    """Every net in every match alignment must exist in the graph."""
    for wr in any_circuit.words:
        for match in wr.matches:
            for port, nets in match.alignment.items():
                for n in nets:
                    assert n in any_circuit.graph.nodes, (
                        f"alignment port {port} references {n} not in graph"
                    )


def test_alignment_widths_match_library(any_circuit):
    """For every match, the number of nets per port equals the declared
    library port width."""
    for wr in any_circuit.words:
        for match in wr.matches:
            widths = match.component.port_widths()
            for port, nets in match.alignment.items():
                assert len(nets) == widths[port], (
                    f"{match.component.name}.{port} declares width "
                    f"{widths[port]}, alignment has {len(nets)} nets"
                )


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------

def _stable_json(result):
    """JSON format without timing stats, for byte-identical comparisons.

    The output structure (module, words) is deterministic; timing is not.
    Determinism tests assert stability of the former, not the latter.
    """
    data = json.loads(format_json(result))
    data.pop("stats", None)
    return json.dumps(data, sort_keys=True)


def test_output_is_byte_identical_across_runs(tmp_path):
    """Two consecutive runs produce identical structure (timing excluded).

    Catches non-determinism from set iteration or dict ordering in the
    actual recovered data — not the wall-clock time of the run.
    """
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    r1 = run_pipeline(path, shape_k=2)
    r2 = run_pipeline(path, shape_k=2)
    assert _stable_json(r1) == _stable_json(r2)


def test_output_is_byte_identical_across_processes(tmp_path):
    """A subprocess run produces identical structure (timing excluded) to
    an in-process run. Catches PYTHONHASHSEED-driven iteration order
    drift in the recovered data.
    """
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    in_proc = _stable_json(run_pipeline(path, shape_k=2))

    result = subprocess.run(
        [sys.executable, "-m", "wordrev", str(path),
         "--shape-k", "2", "--json"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr

    sub_data = json.loads(result.stdout)
    sub_data.pop("stats", None)
    sub_proc = json.dumps(sub_data, sort_keys=True)
    assert sub_proc == in_proc


def test_pipeline_reuses_precomputed_inputs(tmp_path):
    """Passing precomputed shape_hashes and signatures yields identical
    structural output (timing excluded) to computing them inside the
    pipeline."""
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    netlist = parse_yosys_json(str(path))
    module = netlist.modules["and2"]
    G = build_graph(module)
    hashes = compute_all_shape_hashes(G, k=2)
    sigs = compute_functional_signatures(G)

    auto = run_pipeline(path, shape_k=2)
    pre = run_pipeline(path, shape_hashes=hashes, signatures=sigs, shape_k=2)
    assert _stable_json(auto) == _stable_json(pre)


# -----------------------------------------------------------------------------
# Pipeline plumbing
# -----------------------------------------------------------------------------

def test_identify_false_preserves_word_structure(tmp_path):
    """identify=False drops matches without altering the words."""
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    with_id = run_pipeline(path, shape_k=2, identify=True)
    without_id = run_pipeline(path, shape_k=2, identify=False)

    assert with_id.num_words == without_id.num_words

    for wr_id, wr_no in zip(with_id.words, without_id.words):
        assert wr_id.word.bits == wr_no.word.bits
        assert wr_id.word.source == wr_no.word.source
        assert wr_id.word.confidence == wr_no.word.confidence

    for wr in without_id.words:
        assert wr.matches == []

def test_stats_components_sum_to_total(tmp_path):
    """total_seconds equals the sum of stage seconds.

    Catches a plumbing bug where a new stage is added but forgotten in
    the total, or where a stage is double-counted.
    """
    path = make_2bit_and_netlist(tmp_path / "and2.json")
    result = run_pipeline(path, shape_k=2)
    s = result.stats
    component_sum = (
        s["parse_seconds"]
        + s["graph_seconds"]
        + s["analysis_seconds"]
        + s["aggregate_seconds"]
        + s["identify_seconds"]
    )
    assert abs(s["total_seconds"] - component_sum) < 1e-6, (
        f"total {s['total_seconds']} != sum {component_sum}"
    )


# -----------------------------------------------------------------------------
# Confidence semantics
# -----------------------------------------------------------------------------

def test_source_confidence_consistent(any_circuit):
    """Every word's source and confidence must agree.

    cluster_carry      -> conf 1.0
    cluster_structural -> conf 0.5
    propagated         -> conf strictly in (0, 1) (decayed per round)
    """
    expected = {
        "cluster_carry": 1.0,
        "cluster_structural": 0.5,
    }
    for wr in any_circuit.words:
        source = wr.word.source
        if source in expected:
            assert wr.word.confidence == expected[source], (
                f"{source} word has confidence {wr.word.confidence}, "
                f"expected {expected[source]}"
            )
        elif source == "propagated":
            assert 0.0 < wr.word.confidence < 1.0, (
                f"propagated word has confidence {wr.word.confidence}, "
                f"expected a decayed value in (0, 1)"
            )
        else:
            pytest.fail(f"unknown word source: {source}")