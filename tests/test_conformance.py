# tests/test_conformance.py
"""Conformance: does the parser accept real Yosys output?

This is the ONLY test that touches a real file. It exists to catch
schema drift between our synthetic JSON tests and what Yosys actually
emits. If Yosys output changes shape, this test breaks—which is the
signal to update the parser and the synthetic tests together.
"""
from pathlib import Path
import pytest

from wordrev.parser import parse_yosys_json

FIXTURE = Path(__file__).parent / "conformance" / "yosys_output.json"


@pytest.mark.skipif(not FIXTURE.exists(), reason="no conformance fixture")
def test_real_yosys_output_parses():
    netlist = parse_yosys_json(str(FIXTURE))
    assert netlist.modules, "parser returned no modules"

    for module in netlist.modules.values():
        # Invariants the parser promises for any input:
        for gate in module.gates.values():
            assert gate.outputs or gate.inouts, (
                f"{gate.name}: no output or inout pins"
            )
        for net in module.nets.values():
            assert (net.name is None) == (net.name_source.name == "NONE")