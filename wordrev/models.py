from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal, TypedDict

BitID = int | str
Direction = Literal["input", "output", "inout"]


class PortInfo(TypedDict):
    direction: Direction
    bits: list[BitID]


class NameSource(IntEnum):
    NONE = 0
    CONST = 1
    HIDDEN_NET = 2
    PUBLIC_NET = 3
    PORT = 4


@dataclass
class Net:
    id: BitID
    name: str | None = None
    name_source: NameSource = NameSource.NONE
    base_name: str | None = None
    vector_index: int | None = None
    is_constant: bool = False
    constant_value: int | str | None = None
    is_port: bool = False
    port_direction: Direction | None = None


@dataclass
class Gate:
    name: str
    cell_type: str
    inputs: dict[str, list[BitID]] = field(default_factory=dict)
    outputs: dict[str, list[BitID]] = field(default_factory=dict)
    inouts: dict[str, list[BitID]] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Gate({self.name}, {self.cell_type})"


@dataclass
class Module:
    name: str
    ports: dict[str, PortInfo] = field(default_factory=dict)
    gates: dict[str, Gate] = field(default_factory=dict)
    nets: dict[BitID, Net] = field(default_factory=dict)

    def get_net(self, net_id: BitID) -> Net:
        if net_id not in self.nets:
            self.nets[net_id] = Net(id=net_id)
        return self.nets[net_id]


@dataclass
class Netlist:
    modules: dict[str, Module] = field(default_factory=dict)
    creator: str = ""