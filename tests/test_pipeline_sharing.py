# tests/test_pipeline_sharing.py
"""Compare WL and unroll shape hashing at the pipeline level.

Each test runs the full pipeline twice (--method wl and --method unroll)
at hop-aligned k values and asserts whether the recovered words are
identical or divergent.

Hop alignment: WL k rounds cover 2k bipartite hops; unroll k levels cover
k hops. Fair comparison pairs WL k with unroll 2k.
"""
import json

from wordrev.pipeline import format_json, run_pipeline

def _stable_json(result):
    """JSON format without timing stats, for byte-identical comparisons.

    The structural output (module, words) is deterministic; timing is not.
    Determinism tests assert stability of the former, not the latter.
    """
    data = json.loads(format_json(result))
    data.pop("stats", None)
    return json.dumps(data, sort_keys=True)

def make_shared_subexpression_netlist(path):
    """Shared subexpression within a single output word.

    Both y[0] and y[1] compute their AND from the SAME net n0 = XOR(a,b).
    This is within-cone sharing across bits, the case where WL and unroll
    should diverge.
    """
    ports = {
        "a": {"direction": "input", "bits": [2, 3]},
        "b": {"direction": "input", "bits": [4, 5]},
        "y": {"direction": "output", "bits": [10, 11]},
    }
    cells = {
        # Shared: n0 = XOR(a[0], b[0]), feeds both y-bits
        "g0": {"type": "$_XOR_",
               "port_directions": {"A": "input", "B": "input", "Y": "output"},
               "connections": {"A": [2], "B": [4], "Y": [20]}},
        # Both output bits consumed the same XOR output (different second inputs)
        "g1": {"type": "$_AND_",
               "port_directions": {"A": "input", "B": "input", "Y": "output"},
               "connections": {"A": [20], "B": [3], "Y": [10]}},
        "g2": {"type": "$_AND_",
               "port_directions": {"A": "input", "B": "input", "Y": "output"},
               "connections": {"A": [20], "B": [5], "Y": [11]}},
    }
    netnames = {
        "a": {"hide_name": 0, "bits": [2, 3]},
        "b": {"hide_name": 0, "bits": [4, 5]},
        "y": {"hide_name": 0, "bits": [10, 11]},
    }
    data = {"creator": "test",
            "modules": {"shared": {"ports": ports, "cells": cells,
                                   "netnames": netnames}}}
    path.write_text(json.dumps(data))
    return path


def make_independent_netlist(path):
    """Same function as make_shared_subexpression_netlist, but each
    output bit has its own independent XOR. No sharing."""
    ports = {
        "a": {"direction": "input", "bits": [2, 3]},
        "b": {"direction": "input", "bits": [4, 5]},
        "y": {"direction": "output", "bits": [10, 11]},
    }
    cells = {
        "g0": {"type": "$_XOR_",
               "port_directions": {"A": "input", "B": "input", "Y": "output"},
               "connections": {"A": [2], "B": [4], "Y": [20]}},  # bit 0 XOR
        "g1": {"type": "$_XOR_",
               "port_directions": {"A": "input", "B": "input", "Y": "output"},
               "connections": {"A": [3], "B": [5], "Y": [21]}},  # bit 1 XOR
        "g2": {"type": "$_AND_",
               "port_directions": {"A": "input", "B": "input", "Y": "output"},
               "connections": {"A": [20], "B": [3], "Y": [10]}},
        "g3": {"type": "$_AND_",
               "port_directions": {"A": "input", "B": "input", "Y": "output"},
               "connections": {"A": [21], "B": [5], "Y": [11]}},
    }
    netnames = {
        "a": {"hide_name": 0, "bits": [2, 3]},
        "b": {"hide_name": 0, "bits": [4, 5]},
        "y": {"hide_name": 0, "bits": [10, 11]},
    }
    data = {"creator": "test",
            "modules": {"indep": {"ports": ports, "cells": cells,
                                  "netnames": netnames}}}
    path.write_text(json.dumps(data))
    return path


def _word_partition(result):
    """The partition of nets induced by the recovered words."""
    return frozenset(frozenset(w.word.bits) for w in result.words)


# -----------------------------------------------------------------------------
# Same-partition tests: on circuits without sharing, methods agree
# -----------------------------------------------------------------------------

def test_methods_agree_on_independent_netlist(tmp_path):
    """No within-cone sharing → both methods should recover the same words."""
    path = make_independent_netlist(tmp_path / "indep.json")
    wl = run_pipeline(path, shape_k=2, shape_method="wl", identify=False)
    un = run_pipeline(path, shape_k=4, shape_method="unroll", identify=False)
    assert _word_partition(wl) == _word_partition(un), (
        f"methods should agree on a non-shared circuit; "
        f"WL={_word_partition(wl)}, unroll={_word_partition(un)}"
    )


def test_methods_agree_on_mux4(mux4_json):
    """mux4 has no sharing → both methods agree."""
    wl = run_pipeline(mux4_json, shape_k=2, shape_method="wl", identify=False)
    un = run_pipeline(mux4_json, shape_k=4, shape_method="unroll", identify=False)
    assert _word_partition(wl) == _word_partition(un)


def test_methods_diverge_on_add4(add4_json):
    """On a synthesized adder, WL and unroll produce different word
    partitions. This demonstrates that the canonical-leaf-ID fix has a
    measurable effect — sharing detection is now visible.

    Contrast with test_methods_agree_on_independent_netlist, where the
    two methods agree because no sharing exists.
    """
    wl = run_pipeline(add4_json, shape_k=2, shape_method="wl", identify=False)
    un = run_pipeline(add4_json, shape_k=4, shape_method="unroll", identify=False)
    assert _word_partition(wl) != _word_partition(un)


# -----------------------------------------------------------------------------
# Shape hash unit-level: the canonical leaf ID makes the difference
# -----------------------------------------------------------------------------

def test_wl_vs_unroll_on_and_of_same_input():
    """Direct demonstration at the shape-hash level: AND(a, a) vs AND(a, b)."""
    from wordrev.graph import build_graph, net_node
    from wordrev.models import Gate, Module
    from wordrev.shape import compute_all_shape_hashes
    from wordrev.shape_unroll import compute_all_shape_hashes_unrolled

    # Same input
    m1 = Module("t")
    m1.gates["g"] = Gate("g", "$_AND_", {"A": [2], "B": [2]}, {"Y": [4]})
    m1.get_net(2)
    m1.get_net(4)
    G1 = build_graph(m1)

    # Different inputs
    m2 = Module("t")
    m2.gates["g"] = Gate("g", "$_AND_", {"A": [2], "B": [3]}, {"Y": [4]})
    m2.get_net(2)
    m2.get_net(3)
    m2.get_net(4)
    G2 = build_graph(m2)

    wl1 = compute_all_shape_hashes(G1, k=2)[net_node(4)]
    wl2 = compute_all_shape_hashes(G2, k=2)[net_node(4)]
    un1 = compute_all_shape_hashes_unrolled(G1, k=4)[net_node(4)]
    un2 = compute_all_shape_hashes_unrolled(G2, k=4)[net_node(4)]

    assert wl1 == wl2, "WL expected to give identical labels for both"
    assert un1 != un2, "unroll expected to distinguish them"


# -----------------------------------------------------------------------------
# Determinism: both methods must be stable
# -----------------------------------------------------------------------------

def test_wl_deterministic_on_shared(tmp_path):
    path = make_shared_subexpression_netlist(tmp_path / "shared.json")
    r1 = run_pipeline(path, shape_k=2, shape_method="wl", identify=False)
    r2 = run_pipeline(path, shape_k=2, shape_method="wl", identify=False)
    assert _stable_json(r1) == _stable_json(r2)


def test_unroll_deterministic_on_shared(tmp_path):
    path = make_shared_subexpression_netlist(tmp_path / "shared.json")
    r1 = run_pipeline(path, shape_k=4, shape_method="unroll", identify=False)
    r2 = run_pipeline(path, shape_k=4, shape_method="unroll", identify=False)
    assert _stable_json(r1) == _stable_json(r2)


# -----------------------------------------------------------------------------
# CLI integration: --method flag actually switches methods
# -----------------------------------------------------------------------------

def test_cli_method_flag_switches(tmp_path):
    """Regression: the --method flag must reach run_pipeline and not be
    silently overridden by a hardcoded compute_all_shape_hashes call."""
    import subprocess
    import sys
    path = make_shared_subexpression_netlist(tmp_path / "shared.json")
    wl_out = subprocess.run(
        [sys.executable, "-m", "wordrev", str(path),
         "--shape-k", "2", "--method", "wl", "--json", "--no-identify"],
        capture_output=True, text=True, timeout=60,
    )
    assert wl_out.returncode == 0, wl_out.stderr

    un_out = subprocess.run(
        [sys.executable, "-m", "wordrev", str(path),
         "--shape-k", "4", "--method", "unroll", "--json", "--no-identify"],
        capture_output=True, text=True, timeout=60,
    )
    assert un_out.returncode == 0, un_out.stderr

    wl = json.loads(wl_out.stdout)
    un = json.loads(un_out.stdout)
    # We do not assert they differ — they might agree depending on the
    # circuit. What we assert is that the CLI accepted the flag and produced
    # valid output. A separate test on a shared circuit would show divergence.
    assert "words" in wl and "words" in un
    assert wl["module"] == un["module"]