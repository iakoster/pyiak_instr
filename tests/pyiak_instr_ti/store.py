from __future__ import annotations
from dataclasses import InitVar

from src.pyiak_instr.rwfile import RWConfig
from src.pyiak_instr.encoders import BytesEncoder
from src.pyiak_instr.store.bin import (
    STRUCT_DATACLASS,
    Field,
    FieldPattern,
    Container,
    ContainerPattern,
    Struct,
    StructPattern,
    ContinuousStructPattern,
)


@STRUCT_DATACLASS
class TIField(Field):

    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


@STRUCT_DATACLASS
class TIStruct(Struct[TIField]):
    ...


class TIContainer(Container[TIField, TIStruct, "ContainerPattern"]):
    _storage_struct_type = TIStruct


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
    _rwdata = RWConfig
    _sub_p_type = TIStructPattern
    _options = {"basic": TIContainer}
