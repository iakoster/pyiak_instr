from copy import deepcopy
from typing import Any

import pandas as pd

from pyinstr_iakoster.core import Code
from pyinstr_iakoster.communication import (
    FieldSetter,
    RegisterMap,
    AsymmetricResponseField,
    MessageType,
    MessageSetter,
    MessageFormat,
    MessageFormatMap,
    PackageFormat,
)


SETTERS = [
    dict(
        address=FieldSetter.address(fmt=">I"),
        data_length=FieldSetter.data_length(
            fmt=">I", units=FieldSetter.WORDS
        ),
        operation=FieldSetter.operation(fmt=">I", desc_dict={"w": 0, "r": 1}),
        data=FieldSetter.data(expected=-1, fmt=">I")
    ),
    dict(
        preamble=FieldSetter.static(fmt=">H", default=0xaa55),
        operation=FieldSetter.operation(
            fmt=">B", desc_dict={"wp": 1, "rp": 2, "wn": 3, "rn": 4}
        ),
        response=FieldSetter.response(
            fmt=">B", codes={0: Code.OK}, default=0, default_code=Code.ERROR
        ),
        address=FieldSetter.address(fmt=">H"),
        data_length=FieldSetter.data_length(fmt=">H"),
        data=FieldSetter.data(expected=-1, fmt=">f"),
        crc=FieldSetter.crc(fmt=">H", wo_fields={"preamble"})
    ),
    dict(
        operation=FieldSetter.operation(fmt=">B", desc_dict={"r": 1, "w": 2}),
        response=FieldSetter.response(
            fmt=">B",
            codes={0: Code.OK, 4: Code.WAIT},
            default=0,
            default_code=Code.UNDEFINED
        ),
        address=FieldSetter.address(fmt=">H"),
        data_length=FieldSetter.data_length(fmt=">H"),
        data=FieldSetter.data(expected=-1, fmt=">f"),
        crc=FieldSetter.crc(fmt=">H")
    ),
    dict(
        operation=FieldSetter.operation(fmt="B", desc_dict={"r": 1, "w": 2}),
        response1=FieldSetter.response(
            fmt="B",
            codes={0: Code.OK, 4: Code.WAIT},
            default=0,
            default_code=Code.ERROR
        ),
        address=FieldSetter.address(fmt="B"),
        data_length=FieldSetter.data_length(fmt="B"),
        data=FieldSetter.data(expected=-1, fmt="B"),
        response2=FieldSetter.response(
            fmt="B",
            codes={0: Code.OK, 4: Code.WAIT},
            default=0,
            default_code=Code.ERROR
        )
    ),
    dict(
        id=FieldSetter.single(fmt=">I"),
        address=FieldSetter.address(fmt=">I"),
        data=FieldSetter.data(expected=-1, fmt=">I"),
    ),
]

MF_MSG_ARGS = [
    dict(
        message_setter=MessageSetter(
            message_type="strong",
            mf_name="n0",
            splittable=True,
            slice_length=256,
        ),
        arf=AsymmetricResponseField(
            operand="!=",
            start=12,
            stop=16,
            value=b"\x00\x00\x00\x01"
        )
    ),
    dict(
        message_setter=MessageSetter(
            message_type="strong",
            mf_name="n1",
            splittable=False,
            slice_length=1024,
        ),
    ),
    dict(
        message_setter=MessageSetter(
            message_type="strong",
            mf_name="n2",
            splittable=False,
            slice_length=1024,
        ),
    ),
    dict(
        message_setter=MessageSetter(
            message_type="strong",
            mf_name="n3",
            splittable=False,
            slice_length=1024,
        ),
    ),
    dict(
        message_setter=MessageSetter(
            message_type="field",
            mf_name="n4",
            splittable=False,
            slice_length=1024,
        )
    ),
]


def get_setters(num: int) -> dict[str, FieldSetter]:
    return SETTERS[num]


def get_message(num: int) -> MessageType:
    msg_args = deepcopy(MF_MSG_ARGS[num])
    setter = msg_args["message_setter"]
    return setter.message.configure(**SETTERS[num])


def get_mf(num: int, get_ref=True):
    def unpack_setter(setter: FieldSetter) -> dict[str, Any]:
        return dict(field_type=setter.field_type, kwargs=setter.kwargs)

    mf = MessageFormat(**MF_MSG_ARGS[num], **SETTERS[num])
    if get_ref:
        return mf, dict(
            message_setter=mf.message_setter,
            setters={n: unpack_setter(s) for n, s in mf.setters.items()}
        )
    return mf


REGISTER_MAP_TABLE = pd.DataFrame(columns=RegisterMap.EXPECTED_COLUMNS)
for i, reg_args in enumerate([
    ("n0", 1, "rw", 1, None),
    ("n0", 0x200, "rw", 1, ">H"),
    ("n0", 0x10, "ro", 20, None),
    ("n0", 0x100, "wo", 5, None),
    ("n0", 0x1000, "rw", 1024, None),
    ("n1", 0x500, "ro", 4, ">f"),
    ("n1", 0xf000, "rw", 6, ">I"),
    ("n2", 0x10, "rw", 4, None),
    ("n2", 0x11, "rw", 4, ">I"),
    ("n3", 0x24, "rw", 4, None),
    ("n4", 0x1123, "rw", 256, None),
]):
    REGISTER_MAP_TABLE.loc[i] = [
        f"t{i}", f"t_{i}", *reg_args, f"Short {i}. Long."
    ]

MF_DICT: dict[str, MessageFormat] = {
    f"n{n}": get_mf(n, get_ref=False) for n in range(len(SETTERS))
}

MF_CFG_DICT = dict(
    n0=dict(
        message_setter="\\dct(message_type,strong,mf_name,n0,"
                       "splittable,True,slice_length,256)",
        arf="\\dct(operand,!=,"
            "value,\\bts(0,0,0,1),"
            "start,12,"
            "stop,16)",
        address="\\dct(field_type,address,fmt,>I)",
        data_length="\\dct(field_type,data_length,fmt,>I,units,17,additive,0)",
        operation="\\dct(field_type,operation,fmt,>I,"
                  "desc_dict,\\dct(w,0,r,1))",
        data="\\dct(field_type,data,expected,-1,fmt,>I)",
    ),
    n1=dict(
        message_setter="\\dct(message_type,strong,mf_name,n1,"
                       "splittable,False,slice_length,1024)",
        arf="\\dct()",
        preamble="\\dct(field_type,static,fmt,>H,default,43605)",
        operation="\\dct(field_type,operation,fmt,>B,"
                  "desc_dict,\\dct(wp,1,rp,2,wn,3,rn,4))",
        response="\\dct(field_type,response,"
                 "fmt,>B,"
                 "codes,\\dct(0,1280),"
                 "default,0,"
                 "default_code,1282)",
        address="\\dct(field_type,address,fmt,>H)",
        data_length="\\dct(field_type,data_length,fmt,>H,units,16,additive,0)",
        data="\\dct(field_type,data,expected,-1,fmt,>f)",
        crc="\\dct(field_type,crc,fmt,>H,algorithm_name,crc16-CCITT/XMODEM,"
            "wo_fields,\\set(crc,preamble))",  # todo: check set order
    ),
    n2=dict(
        message_setter="\\dct(message_type,strong,mf_name,n2,"
                       "splittable,False,slice_length,1024)",
        arf="\\dct()",
        operation="\\dct(field_type,operation,fmt,>B,"
                  "desc_dict,\\dct(r,1,w,2))",
        response="\\dct(field_type,response,"
                 "fmt,>B,"
                 "codes,\\dct(0,1280,4,1281),"
                 "default,0,"
                 "default_code,255)",
        address="\\dct(field_type,address,fmt,>H)",
        data_length="\\dct(field_type,data_length,fmt,>H,units,16,additive,0)",
        data="\\dct(field_type,data,expected,-1,fmt,>f)",
        crc="\\dct(field_type,crc,fmt,>H,algorithm_name,crc16-CCITT/XMODEM,"
            "wo_fields,None)",
    ),
    n3=dict(
        message_setter="\\dct(message_type,strong,mf_name,n3,"
                       "splittable,False,slice_length,1024)",
        arf="\\dct()",
        operation="\\dct(field_type,operation,fmt,B,"
                  "desc_dict,\\dct(r,1,w,2))",
        response1="\\dct(field_type,response,"
                  "fmt,B,"
                  "codes,\\dct(0,1280,4,1281),"
                  "default,0,"
                  "default_code,1282)",
        address="\\dct(field_type,address,fmt,B)",
        data_length="\\dct(field_type,data_length,fmt,B,units,16,additive,0)",
        data="\\dct(field_type,data,expected,-1,fmt,B)",
        response2="\\dct(field_type,response,"
                  "fmt,B,"
                  "codes,\\dct(0,1280,4,1281),"
                  "default,0,"
                  "default_code,1282)",
    ),
    n4=dict(
        message_setter="\\dct(message_type,field,mf_name,n4,splittable,False,"
               "slice_length,1024)",
        arf="\\dct()",
        id="\\dct(field_type,single,fmt,>I,default,\\lst(),"
           "may_be_empty,False)",
        address="\\dct(field_type,address,fmt,>I)",
        data="\\dct(field_type,data,expected,-1,fmt,>I)",
    )
)

PF = PackageFormat(
    registers=RegisterMap(REGISTER_MAP_TABLE),
    formats=MessageFormatMap(*MF_DICT.values()),
)
