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

from collections.abc import Sequence
import json
import logging
from pathlib import Path
import re
from typing import Any

from .models import BitID, Direction, Gate, Module, NameSource, Net, Netlist

log = logging.getLogger(__name__)

CONSTANT_MAP: dict[BitID, int | str] = {
    0: 0,
    1: 1,
    "0": 0,
    "1": 1,
    "x": "x",  # Yosys undefined / don't-care bit
    "z": "z",  # Yosys high-impedance (tri-state) bit
    "m": "x",  # Yosys marker bit (internal undef/marker semantics; normalized to 'x')
}

# Matches exactly the format splitnets emits: base[idx]
# base: identifier starting with letter/underscore, then word chars/underscores
# idx:  non-negative integer
# Excludes: escaped names (\base[3]), names with dots, names with [N] mid-suffix
_SPLIT_VECTOR_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_yosys_json(path: str | Path) -> Netlist:
    """Parse a Yosys JSON file into a Netlist."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return parse_yosys_dict(data)


def parse_yosys_dict(data: dict[str, Any]) -> Netlist:
    """Parse an already-decoded Yosys JSON object into a Netlist."""
    netlist = Netlist(creator=data.get("creator", ""))
    for mod_name, mod_data in data.get("modules", {}).items():
        netlist.modules[mod_name] = _parse_module(mod_name, mod_data)
    return netlist


# ---------------------------------------------------------------------------
# Module & Section Parsers
# ---------------------------------------------------------------------------


def _parse_module(mod_name: str, mod_data: dict[str, Any]) -> Module:
    module = Module(name=mod_name)
    _parse_ports(module, mod_data)
    _parse_netnames(module, mod_data)
    _parse_cells(module, mod_data)
    _backfill_vector_metadata(module)
    return module


def _parse_ports(module: Module, mod_data: dict[str, Any]) -> None:
    for port_name, port_info in mod_data.get("ports", {}).items():
        direction: Direction = port_info.get("direction", "input")
        bits = list(port_info.get("bits", []))
        module.ports[port_name] = {"direction": direction, "bits": bits}
        _register_signal_bits(
            module,
            port_name,
            bits,
            NameSource.PORT,
            is_port=True,
            port_direction=direction,
        )


def _parse_netnames(module: Module, mod_data: dict[str, Any]) -> None:
    for net_name, net_info in mod_data.get("netnames", {}).items():
        is_hidden = bool(net_info.get("hide_name", 0))
        source = NameSource.HIDDEN_NET if is_hidden else NameSource.PUBLIC_NET
        bits = net_info.get("bits", [])
        _register_signal_bits(module, net_name, bits, source)


def _parse_cells(module: Module, mod_data: dict[str, Any]) -> None:
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


# ---------------------------------------------------------------------------
# Signal Registration & Post-Pass Helpers
# ---------------------------------------------------------------------------


def _register_signal_bits(
    module: Module,
    name: str,
    bits: Sequence[BitID],
    source: NameSource,
    *,
    is_port: bool = False,
    port_direction: Direction | None = None,
) -> None:
    is_vector = len(bits) > 1
    for idx, bit_id in enumerate(bits):
        net = _process_bit(module, bit_id)
        if is_port:
            net.is_port = True
            # Known limitation: direct feedthroughs (e.g., `assign out = in;`) share a
            # single bit_id, so a later port overwrites port_direction while _assign_name
            # retains the first port's name. Fixing this requires Net.port_direction ->
            # port_directions: frozenset[str] across graph.py, tests, and JSON output;
            # deferred since feedthroughs are rare post-synthesis.
            net.port_direction = port_direction
        if net.is_constant:
            continue
        _assign_name(net, _decorate(name, idx, is_vector), source)
        # Intentional layering: net.name comes from the highest-priority NameSource,
        # while base_name/vector_index are populated from any multi-bit declaration
        # that provides them so downstream consumers (_group_free_vars, _bit_sort_key)
        # preserve bus grouping even when a single-bit port aliases a multi-bit net.
        if is_vector and net.vector_index is None:
            net.base_name = name
            net.vector_index = idx


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
    if count > 0:
        log.info(
            "module %r: reconstructed vector metadata for %d nets from "
            "split single-bit declarations (splitnets in the synthesis flow)",
            module.name,
            count,
        )
    return count


def _reconstruct_from_split_name(name: str) -> tuple[str, int] | None:
    """Reverse _decorate: parse 'base[idx]' into (base, idx), or None."""
    m = _SPLIT_VECTOR_RE.match(name)
    if m is None:
        return None
    return m.group(1), int(m.group(2))


def _assign_name(net: Net, candidate_name: str, source: NameSource) -> None:
    """Assign candidate_name if source outranks the net's existing name source."""
    if source > net.name_source:
        net.name = candidate_name
        net.name_source = source


def _decorate(base: str, idx: int, is_vector: bool) -> str:
    return f"{base}[{idx}]" if is_vector else base


def _get_constant_value(bit_id: BitID) -> int | str | None:
    if isinstance(bit_id, bool):
        return None
    return CONSTANT_MAP.get(bit_id)


def _process_bit(module: Module, bit_id: BitID) -> Net:
    net = module.get_net(bit_id)
    const_val = _get_constant_value(bit_id)
    if const_val is not None:
        net.is_constant = True
        net.constant_value = const_val
        if net.name is None:
            net.name = f"const_{const_val}"
            net.name_source = NameSource.CONST
    return net