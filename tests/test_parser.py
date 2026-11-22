# tests/test_parser.py
"""
Parser contract tests.

We test *behaviour*, not implementation:
  - naming precedence (PORT > PUBLIC > HIDDEN > CONST)
  - constant detection across int/str/bool forms
  - vector decoration with [idx]
  - cell pin classification (input/output/inout)
  - structural invariants (every referenced net exists)

We do NOT test:
  - the value of any hash or canonical string
  - the internals of _decorate, _assign_name, or _process_bit
    (they are exercised through the public parse_yosys_dict surface)
"""
import pytest

from wordrev.models import NameSource
from wordrev.parser import parse_yosys_dict


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def make_json(ports=None, cells=None, netnames=None, creator=None):
    """Build a minimal Yosys JSON with a single module 'm'."""
    mod = {"ports": ports or {}, "cells": cells or {}, "netnames": netnames or {}}
    data = {"modules": {"m": mod}}
    if creator is not None:
        data["creator"] = creator
    return data


def get_module(data):
    return parse_yosys_dict(data).modules["m"]


def get_net(data, bit_id):
    return get_module(data).nets[bit_id]


# -----------------------------------------------------------------------------
# Module-level structure
# -----------------------------------------------------------------------------

def test_module_name_matches_key():
    m = get_module(make_json())
    assert m.name == "m"


def test_creator_preserved():
    data = make_json(creator="Yosys 0.70+190")
    nl = parse_yosys_dict(data)
    assert nl.creator == "Yosys 0.70+190"


def test_empty_module_is_valid():
    m = get_module(make_json())
    assert m.ports == {}
    assert m.gates == {}
    assert m.nets == {}


def test_multiple_modules_parsed():
    data = {"modules": {
        "a": {"ports": {}, "cells": {}, "netnames": {}},
        "b": {"ports": {}, "cells": {}, "netnames": {}},
    }}
    nl = parse_yosys_dict(data)
    assert set(nl.modules.keys()) == {"a", "b"}


# -----------------------------------------------------------------------------
# Port parsing
# -----------------------------------------------------------------------------

def test_port_direction_and_bits_captured():
    data = make_json(ports={
        "a": {"direction": "input", "bits": [2]},
        "y": {"direction": "output", "bits": [5]},
    })
    m = get_module(data)
    assert m.ports["a"] == {"direction": "input", "bits": [2]}
    assert m.ports["y"] == {"direction": "output", "bits": [5]}


def test_single_bit_port_bare_name():
    data = make_json(ports={"a": {"direction": "input", "bits": [2]}})
    net = get_net(data, 2)
    assert net.name == "a"
    assert net.name_source == NameSource.PORT
    assert net.is_port is True
    assert net.port_direction == "input"


def test_multi_bit_port_decorated_name():
    data = make_json(ports={
        "a": {"direction": "input", "bits": [10, 11, 12]},
    })
    m = get_module(data)
    assert m.nets[10].name == "a[0]"
    assert m.nets[11].name == "a[1]"
    assert m.nets[12].name == "a[2]"
    for bid in (10, 11, 12):
        assert m.nets[bid].name_source == NameSource.PORT
        assert m.nets[bid].is_port is True


def test_port_bit_that_is_constant_keeps_const_name():
    """A port whose bit is 0/1/x/z keeps the const_N name but still
    records is_port so inspection can surface the port context."""
    data = make_json(ports={"a": {"direction": "input", "bits": [0]}})
    net = get_net(data, 0)
    assert net.is_constant is True
    assert net.name == "const_0"
    assert net.name_source == NameSource.CONST
    assert net.is_port is True


# -----------------------------------------------------------------------------
# Netname parsing and naming precedence
# -----------------------------------------------------------------------------

def test_public_netname_assigned():
    data = make_json(netnames={"sig": {"hide_name": 0, "bits": [5]}})
    net = get_net(data, 5)
    assert net.name == "sig"
    assert net.name_source == NameSource.PUBLIC_NET


def test_hidden_netname_assigned_when_alone():
    data = make_json(netnames={"_x": {"hide_name": 1, "bits": [5]}})
    net = get_net(data, 5)
    assert net.name == "_x"
    assert net.name_source == NameSource.HIDDEN_NET


def test_public_beats_hidden_public_first():
    data = make_json(netnames={
        "public":  {"hide_name": 0, "bits": [5]},
        "_hidden": {"hide_name": 1, "bits": [5]},
    })
    assert get_net(data, 5).name == "public"


def test_public_beats_hidden_hidden_first():
    """Order-independence: this is the regression test for the bug
    in the first version, where a hidden name arriving first could
    block a later public name."""
    data = make_json(netnames={
        "_hidden": {"hide_name": 1, "bits": [5]},
        "public":  {"hide_name": 0, "bits": [5]},
    })
    assert get_net(data, 5).name == "public"


def test_port_beats_public_netname():
    data = make_json(
        ports={"a": {"direction": "input", "bits": [5]}},
        netnames={"sig": {"hide_name": 0, "bits": [5]}},
    )
    net = get_net(data, 5)
    assert net.name == "a"
    assert net.name_source == NameSource.PORT


def test_equal_rank_keeps_first_match():
    """Two public names on the same bit: first in dict order wins,
    per the documented 'equal rank keeps first match' rule."""
    data = make_json(netnames={
        "first":  {"hide_name": 0, "bits": [5]},
        "second": {"hide_name": 0, "bits": [5]},
    })
    # Python 3.7+ preserves dict insertion order.
    assert get_net(data, 5).name == "first"


def test_netname_decorated_for_vector():
    data = make_json(netnames={
        "sig": {"hide_name": 0, "bits": [7, 8]},
    })
    m = get_module(data)
    assert m.nets[7].name == "sig[0]"
    assert m.nets[8].name == "sig[1]"


# -----------------------------------------------------------------------------
# Constant detection
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected_value", [
    (0,   0),
    (1,   1),
    ("0", 0),
    ("1", 1),
    ("x", "x"),
    ("z", "z"),
    ("m", "x"),
])
def test_constants_detected(raw, expected_value):
    data = make_json(netnames={"sig": {"hide_name": 0, "bits": [raw]}})
    net = get_net(data, raw)
    assert net.is_constant is True
    assert net.constant_value == expected_value
    assert net.name == f"const_{expected_value}"
    assert net.name_source == NameSource.CONST


@pytest.mark.parametrize("raw", [False, True])
def test_bool_is_not_treated_as_constant(raw):
    """bool is an int subclass in Python; the parser must not
    classify True/False as constant bits 1/0."""
    data = make_json(cells={
        "g": {
            "type": "$_BUF_",
            "port_directions": {"A": "input", "Y": "output"},
            "connections": {"A": [raw], "Y": [5]},
        },
    })
    net = get_net(data, raw)
    assert net.is_constant is False


def test_constant_name_not_overwritten_by_netname():
    data = make_json(netnames={
        "zero": {"hide_name": 0, "bits": [0]},
        "sig":  {"hide_name": 0, "bits": [5]},
    })
    assert get_net(data, 0).name == "const_0"


def test_constant_name_not_overwritten_by_port():
    data = make_json(ports={"a": {"direction": "input", "bits": [1]}})
    assert get_net(data, 1).name == "const_1"


def test_non_constant_bit_with_no_netname_stays_unnamed():
    data = make_json(cells={
        "g": {
            "type": "$_AND_",
            "port_directions": {"A": "input", "B": "input", "Y": "output"},
            "connections": {"A": [2], "B": [3], "Y": [4]},
        },
    })
    net = get_net(data, 4)
    assert net.name is None
    assert net.name_source == NameSource.NONE


# -----------------------------------------------------------------------------
# Cell parsing
# -----------------------------------------------------------------------------

def test_cell_metadata_captured():
    data = make_json(cells={
        "g1": {
            "type": "$_DFF_P_",
            "parameters": {"CLK_POLARITY": "1'b1"},
            "attributes": {"src": "x.v:1"},
            "port_directions": {"D": "input", "CLK": "input", "Q": "output"},
            "connections": {"D": [2], "CLK": [3], "Q": [4]},
        },
    })
    gate = get_module(data).gates["g1"]
    assert gate.cell_type == "$_DFF_P_"
    assert gate.parameters == {"CLK_POLARITY": "1'b1"}
    assert gate.attributes == {"src": "x.v:1"}


def test_cell_inputs_outputs_split():
    data = make_json(cells={
        "g": {
            "type": "$_AND_",
            "port_directions": {"A": "input", "B": "input", "Y": "output"},
            "connections": {"A": [2], "B": [3], "Y": [4]},
        },
    })
    gate = get_module(data).gates["g"]
    assert gate.inputs == {"A": [2], "B": [3]}
    assert gate.outputs == {"Y": [4]}
    assert gate.inouts == {}


def test_cell_inout_captured_separately():
    data = make_json(cells={
        "pad": {
            "type": "$_TRI_",
            "port_directions": {"A": "input", "EN": "input", "Y": "inout"},
            "connections": {"A": [2], "EN": [3], "Y": [4]},
        },
    })
    gate = get_module(data).gates["pad"]
    assert gate.inouts == {"Y": [4]}
    assert "Y" not in gate.inputs
    assert "Y" not in gate.outputs


def test_cell_missing_direction_defaults_to_input():
    data = make_json(cells={
        "g": {
            "type": "$_BUF_",
            # no port_directions at all
            "connections": {"A": [2], "Y": [4]},
        },
    })
    gate = get_module(data).gates["g"]
    assert gate.inputs == {"A": [2], "Y": [4]}
    assert gate.outputs == {}


def test_cell_multi_bit_pin():
    data = make_json(cells={
        "g": {
            "type": "$_MUX_",
            "port_directions": {"A": "input", "S": "input", "Y": "output"},
            "connections": {"A": [10, 11], "S": [2], "Y": [20, 21]},
        },
    })
    gate = get_module(data).gates["g"]
    assert gate.inputs["A"] == [10, 11]
    assert gate.outputs["Y"] == [20, 21]


def test_cell_nets_created():
    """Every net referenced by a cell must have a Net entry, even if
    no port or netname mentions it."""
    data = make_json(cells={
        "g": {
            "type": "$_AND_",
            "port_directions": {"A": "input", "B": "input", "Y": "output"},
            "connections": {"A": [99], "B": [98], "Y": [97]},
        },
    })
    m = get_module(data)
    for bid in (97, 98, 99):
        assert bid in m.nets


# -----------------------------------------------------------------------------
# Structural invariants
# -----------------------------------------------------------------------------

def test_all_port_bits_have_nets():
    data = make_json(ports={
        "a": {"direction": "input", "bits": [2, 3, 4]},
        "y": {"direction": "output", "bits": [5]},
    })
    m = get_module(data)
    for info in m.ports.values():
        for bid in info["bits"]:
            assert bid in m.nets


def test_all_cell_pin_bits_have_nets():
    data = make_json(cells={
        "g": {
            "type": "$_AND_",
            "port_directions": {"A": "input", "B": "input", "Y": "output"},
            "connections": {"A": [11], "B": [12], "Y": [13]},
        },
    })
    m = get_module(data)
    for bid in (11, 12, 13):
        assert bid in m.nets


def test_name_source_and_name_stay_in_sync():
    """Every net with a name has a name_source > NONE, and vice versa."""
    data = make_json(
        ports={"a": {"direction": "input", "bits": [2]}},
        netnames={
            "sig":     {"hide_name": 0, "bits": [3]},
            "_hidden": {"hide_name": 1, "bits": [4]},
        },
        cells={
            "g": {
                "type": "$_AND_",
                "port_directions": {"A": "input", "B": "input", "Y": "output"},
                "connections": {"A": [2], "B": [3], "Y": [5]},
            },
        },
    )
    m = get_module(data)
    for net in m.nets.values():
        if net.name is None:
            assert net.name_source == NameSource.NONE
        else:
            assert net.name_source != NameSource.NONE