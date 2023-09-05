from __future__ import annotations
from dataclasses import InitVar

from src.pyiak_instr.rwfile import RWConfig
from src.pyiak_instr.store.bin import (
    Field,
    FieldPattern,
    Container,
    ContainerPattern,
    Struct,
    StructPattern,
    ContinuousStructPattern,
)


class TIField(Field):
    ...


class TIStruct(Struct[TIField]):
    ...


class TIContainer(Container[TIField, TIStruct, "ContainerPattern"]):
    ...


class TIFieldPattern(FieldPattern[TIField]):
    _options = {"basic": TIField}


class TIStructPattern(
    StructPattern[TIStruct, TIFieldPattern]
):
    _sub_p_type = TIFieldPattern
    _options = {"basic": TIStruct}


class TIContinuousStructPattern(
    ContinuousStructPattern[TIStruct, TIFieldPattern]
):
    _sub_p_type = TIFieldPattern
    _options = {"basic": TIStruct}


class TIContainerPattern(ContainerPattern[TIContainer, TIStructPattern]):
    _sub_p_type = TIStructPattern
    _options = {"basic": TIContainer}
