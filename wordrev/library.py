# wordrev/library.py
"""
Library of known word-level operations for function identification.

Each Component is a pure function from named input ports (each a bit
vector) to a single output word (LSB at index 0).

Components also declare `commutative_groups` — sets of ports that may be
freely permuted without changing the component's function. This is used
by identify._canonical_alignment_key to deduplicate equivalent
alignments without over-collapsing asymmetric ports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Component:
    """A known word-level operation.

    name:
        Human-readable name (e.g., "adder_4").
    output_width:
        Width of the output word in bits.
    input_ports:
        Tuple of (port_name, port_width). Ports are bit vectors; a
        port_width of 1 is a single-bit control signal.
    compute:
        Function from {port_name: integer value} to output integer.
        Bit i of an input port value is that port's bit i. Bit i of the
        return value is the output's bit i.
    commutative_groups:
        Groups of port names that may be permuted without changing the
        function. Ports in the same group must have the same width, and
        each port may appear in at most one group. A component with no
        commutative ports leaves this empty (the default).
    """
    name: str
    output_width: int
    input_ports: tuple[tuple[str, int], ...]
    compute: Callable[[dict[str, int]], int]
    commutative_groups: tuple[frozenset[str], ...] = ()
    select_ports: frozenset[str] = frozenset()

    def __post_init__(self):
        widths = dict(self.input_ports)
        seen: set[str] = set()
        for group in self.commutative_groups:
            if not group:
                raise ValueError(f"{self.name}: empty commutative group")
            unknown = set(group) - set(widths)
            if unknown:
                raise ValueError(
                    f"{self.name}: commutative group references unknown "
                    f"port(s) {unknown}"
                )
            group_widths = {widths[p] for p in group}
            if len(group_widths) > 1:
                raise ValueError(
                    f"{self.name}: ports in commutative group {set(group)} "
                    f"have different widths {group_widths}"
                )
            overlap = seen & set(group)
            if overlap:
                raise ValueError(
                    f"{self.name}: port(s) {overlap} in multiple "
                    f"commutative groups"
                )
            seen |= set(group)

    @property
    def total_input_bits(self) -> int:
        return sum(w for _, w in self.input_ports)

    @property
    def port_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.input_ports)

    def port_widths(self) -> dict[str, int]:
        return dict(self.input_ports)


# -----------------------------------------------------------------------------
# Factories
# -----------------------------------------------------------------------------

def _mask(width: int) -> int:
    return (1 << width) - 1


def make_adder(width: int) -> Component:
    m = _mask(width)
    return Component(
        name=f"adder_{width}",
        output_width=width,
        input_ports=(("a", width), ("b", width)),
        compute=lambda i: (i["a"] + i["b"]) & m,
        commutative_groups=(frozenset({"a", "b"}),),
    )


def make_adder_inc(width: int) -> Component:
    m = _mask(width)
    return Component(
        name=f"adder_inc_{width}",
        output_width=width,
        input_ports=(("a", width), ("b", width)),
        compute=lambda i: (i["a"] + i["b"] + 1) & m,
        commutative_groups=(frozenset({"a", "b"}),),
    )


def make_subtractor(width: int) -> Component:
    m = _mask(width)
    # At width 1, a - b mod 2 == a XOR b, which is commutative.
    # At width >= 2, a - b != b - a in general, so no commutative group.
    commutative = (frozenset({"a", "b"}),) if width == 1 else ()
    return Component(
        name=f"subtractor_{width}",
        output_width=width,
        input_ports=(("a", width), ("b", width)),
        compute=lambda i: (i["a"] - i["b"]) & m,
        commutative_groups=commutative,
    )


def make_mux(width: int) -> Component:
    # a and b are asymmetric: sel chooses between them.
    # Neither can be swapped with sel. No commutative group.
    return Component(
        name=f"mux_{width}",
        output_width=width,
        input_ports=(("a", width), ("b", width), ("sel", 1)),
        compute=lambda i: i["b"] if i["sel"] else i["a"],
        select_ports=frozenset({"sel"}),
    )


def make_bitwise_and(width: int) -> Component:
    return Component(
        name=f"bitwise_and_{width}",
        output_width=width,
        input_ports=(("a", width), ("b", width)),
        compute=lambda i: i["a"] & i["b"],
        commutative_groups=(frozenset({"a", "b"}),),
    )


def make_bitwise_or(width: int) -> Component:
    return Component(
        name=f"bitwise_or_{width}",
        output_width=width,
        input_ports=(("a", width), ("b", width)),
        compute=lambda i: i["a"] | i["b"],
        commutative_groups=(frozenset({"a", "b"}),),
    )


def make_bitwise_xor(width: int) -> Component:
    return Component(
        name=f"bitwise_xor_{width}",
        output_width=width,
        input_ports=(("a", width), ("b", width)),
        compute=lambda i: i["a"] ^ i["b"],
        commutative_groups=(frozenset({"a", "b"}),),
    )


def make_bitwise_not(width: int) -> Component:
    m = _mask(width)
    return Component(
        name=f"bitwise_not_{width}",
        output_width=width,
        input_ports=(("a", width),),
        compute=lambda i: (~i["a"]) & m,
    )


def candidates_for_width(width: int) -> list[Component]:
    """All library components with the given output width."""
    return [
        make_bitwise_and(width),
        make_bitwise_or(width),
        make_bitwise_xor(width),
        make_bitwise_not(width),
        make_mux(width),
        make_adder(width),
        make_adder_inc(width),
        make_subtractor(width),
    ]