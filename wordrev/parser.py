# wordrev/parser.py
"""
Yosys JSON → Netlist parser.

Layering contract:
    This module produces a `Netlist` of Module/Gate/Net objects.
    It does NOT build graphs, compute shapes, or perform any analysis.

Name priority for a net (higher source rank wins; equal rank keeps first match):
    1. NameSource.PORT       (assigned during port processing)
    2. NameSource.PUBLIC_NET (hide_name == 0)
    3. NameSource.HIDDEN_NET (hide_name == 1)
    4. NameSource.CONST      (const_0, const_1, const_x, const_z)
"""
import json
import re
import logging
from typing import Any, Union

from .models import BitID, Gate, Module, NameSource, Net, Netlist

log = logging.getLogger(__name__)

CONSTANT_MAP: dict[BitID, Union[int, str]] = {
    0: 0,
    1: 1,
    "0": 0,
    "1": 1,
    "x": "x",
    "z": "z",
    "m": "x",
}


def _is_constant_bit(bit_id: BitID) -> bool:
    if isinstance(bit_id, bool):
        return False
    return bit_id in CONSTANT_MAP


def _process_bit(module: Module, bit_id: BitID) -> Net:
    net = module.get_net(bit_id)
    if _is_constant_bit(bit_id):
        net.is_constant = True
        net.constant_value = CONSTANT_MAP[bit_id]
        if net.name is None:
            net.name = f"const_{net.constant_value}"
            net.name_source = NameSource.CONST
    return net


def _decorate(base: str, idx: int, is_vector: bool) -> str:
    return f"{base}[{idx}]" if is_vector else base

# Matches exactly the format splitnets emits: base[idx]
# base: identifier starting with letter/underscore, then word chars/underscores
# idx:  non-negative integer
# Excludes: escaped names (\base[3]), names with dots, names with [N] mid-suffix
_SPLIT_VECTOR_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]$")


def _reconstruct_from_split_name(name: str) -> tuple[str, int] | None:
    """Reverse _decorate: parse 'base[idx]' into (base, idx), or None."""
    m = _SPLIT_VECTOR_RE.match(name)
    if m is None:
        return None
    return m.group(1), int(m.group(2))


def _backfill_vector_metadata(module: Module) -> int:
    """Reconstruct base_name/vector_index from split single-bit port and
    netname declarations.

    Runs as a post-pass after the main parse loops. Only sets fields that
    are still None, so it never overrides metadata from a multi-bit
    declaration. Returns the number of nets whose metadata was filled in,
    which is 0 for netlists that do not use splitnets.
    """
    count = 0
    for net in module.nets.values():
        if net.vector_index is not None or net.is_constant:
            continue
        if net.name is None:
            continue
        parsed = _reconstruct_from_split_name(net.name)
        if parsed is None:
            continue
        base, idx = parsed
        net.base_name = base
        net.vector_index = idx
        count += 1
    return count

def _assign_name(net: Net, candidate_name: str, source: NameSource) -> None:
    """Assign candidate_name if source outranks the net's existing name source."""
    if source > net.name_source:
        net.name = candidate_name
        net.name_source = source


def _parse_module(mod_name: str, mod_data: dict[str, Any]) -> Module:
    module = Module(name=mod_name)

    # --- Ports ---
    for port_name, port_info in mod_data.get("ports", {}).items():
        direction = port_info.get("direction", "input")
        bits = list(port_info.get("bits", []))
        module.ports[port_name] = {"direction": direction, "bits": bits}

        is_vector = len(bits) > 1
        for idx, bit_id in enumerate(bits):
            net = _process_bit(module, bit_id)
            net.is_port = True
            net.port_direction = direction
            if not net.is_constant:
                _assign_name(
                    net, _decorate(port_name, idx, is_vector), NameSource.PORT
                )
                # Populate vector metadata; higher-priority names win, so
                # only set it if a previous lower-priority pass did not.
                if is_vector and net.vector_index is None:
                    net.base_name = port_name
                    net.vector_index = idx

    # --- Netnames ---
    for net_name, net_info in mod_data.get("netnames", {}).items():
        is_hidden = bool(net_info.get("hide_name", 0))
        source = NameSource.HIDDEN_NET if is_hidden else NameSource.PUBLIC_NET
        bits = net_info.get("bits", [])
        is_vector = len(bits) > 1

        for idx, bit_id in enumerate(bits):
            net = _process_bit(module, bit_id)
            if net.is_constant:
                continue
            _assign_name(net, _decorate(net_name, idx, is_vector), source)
            if is_vector and net.vector_index is None:
                net.base_name = net_name
                net.vector_index = idx

    # --- Cells (unchanged) ---
    for cell_name, cell_data in mod_data.get("cells", {}).items():
        gate = Gate(
            name=cell_name,
            cell_type=cell_data.get("type", ""),
            attributes=cell_data.get("attributes", {}),
            parameters=cell_data.get("parameters", {}),
        )
        port_dirs = cell_data.get("port_directions", {})
        for pin, bits in cell_data.get("connections", {}).items():
            bits_list = list(bits)
            direction = port_dirs.get(pin, "input")
            if direction == "output":
                gate.outputs[pin] = bits_list
            elif direction == "inout":
                gate.inouts[pin] = bits_list
            else:
                gate.inputs[pin] = bits_list
            for bit_id in bits_list:
                _process_bit(module, bit_id)
        module.gates[cell_name] = gate

    filled = _backfill_vector_metadata(module)
    if filled > 0:
        log.info(
            "module %r: reconstructed vector metadata for %d nets from "
            "split single-bit declarations (splitnets in the synthesis flow)",
            module.name, filled,
        )

    return module

def parse_yosys_dict(data: dict[str, Any]) -> Netlist:
    """Parse an already-decoded Yosys JSON object into a Netlist."""
    netlist = Netlist(creator=data.get("creator", ""))
    for mod_name, mod_data in data.get("modules", {}).items():
        netlist.modules[mod_name] = _parse_module(mod_name, mod_data)
    return netlist


def parse_yosys_json(path: str) -> Netlist:
    """Parse a Yosys JSON file into a Netlist."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return parse_yosys_dict(data)