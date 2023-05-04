from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Generator,
    ClassVar,
    Any,
    overload,
)

import numpy as np
import numpy.typing as npt

from ...exceptions import (
    MessageContentError,
    NotConfiguredMessageError,
)
from src.pyiak_instr.core import Code


__all__ = [
    "FieldMessage",
    "MessageContentError",
    "NotConfiguredMessageError",
]

class FieldMessage(object):

    # todo: parts count for bytes but split by data with int count of word
    def split(self) -> Generator[FieldMessage, None, None]:
        """
        Split data field on slices.

        Yields
        ------
        FieldMessage
            message part.
        """
        if not self._splittable:
            yield self
            return

        if self.has.AddressField \
                and self.has.OperationField \
                and self.has.DataLengthField:  # todo: refactor
            address = self.get.AddressField
            operation = self.get.OperationField
            data_length = self.get.DataLengthField
            parts_count = int(np.ceil(
                data_length[0] / self._slice_length
            ))

            for i_part in range(parts_count):
                if i_part == parts_count - 1:
                    data_len = data_length[0] - i_part * self._slice_length
                else:
                    data_len = self._slice_length

                msg = deepcopy(self)

                if operation.base == 'r':
                    msg.get.DataLengthField.set(data_len)
                elif operation.base == 'w':
                    start = i_part * self._slice_length
                    end = start + data_len
                    if data_length.units == Code.WORDS:
                        msg.data.set(self.data[start:end])
                    elif data_length.units == Code.BYTES:
                        msg.data.set(self.data.content[start:end])
                    else:
                        raise TypeError('Unsupported data units')
                    msg.get.DataLengthField.update()
                msg.get.AddressField.set(
                    address[0] + i_part * self._slice_length)
                msg.set_src_dst(self._src, self._dst)
                yield msg

    def _validate_content(self) -> None:
        """Validate content."""
        for field in self:
            if not (field.words_count or field.may_be_empty):
                raise MessageContentError(
                    self.__class__.__name__, field.name, "field is empty"
                )

        if self.has.OperationField and self.has.DataLengthField:
            operation = self.get.OperationField
            data_length = self.get.DataLengthField

            if operation.base == "w" and (
                    not self.data.may_be_empty or self.data.words_count
            ) and data_length[0] != data_length.calculate(self.data):
                raise MessageContentError(
                    self.__class__.__name__,
                    data_length.name,
                    "invalid length"
                )

        if self.has.CrcField:
            crc = self.get.CrcField
            res_crc, ref_crc = crc.calculate(self), crc[0]
            if ref_crc != res_crc:
                raise MessageContentError(
                    self.__class__.__name__,
                    crc.name,
                    "invalid crc value, '%x' != '%x'" % (ref_crc, res_crc)
                )

    @property
    def response_codes(self) -> dict[str, Code]:
        """
        Returns
        -------
        dict of {str, Code}
            dictionary of all response codes in the message, where
            key is the field name and value is the code.
            If there is no ResponseField in the message,
            the dictionary will be empty.
        """
        if ResponseField not in self._field_types:
            return {}

        codes = {}
        for field in self._fields.values():
            if isinstance(field, ResponseField):
                codes[field.name] = field.current_code
        return codes

    def __add__(self, other: FieldMessage | bytes) -> FieldMessage:
        """
        Add new data to the data field.

        If the passed argument is the bytes type, the passed argument
        is added to the data field (right concatenation).

        If the passed argument is an instance of the same class or is
        a child, then a content and class validation.

        In both cases it is checked before writing that the new data
        will contain an integer number of words in the format of the current.
        instance

        Parameters
        ----------
        other: bytes or FieldMessage
            additional data or message.

        Returns
        -------
        FieldMessage
            self instance.
        """

        if isinstance(other, FieldMessage):
            if other.mf_name != self._mf_name:
                raise TypeError(
                    "messages have different formats: %s != %s" % (
                        other.mf_name, self._mf_name
                    )
                )
            other = other.data.content
        elif isinstance(other, bytes):
            pass
        else:
            raise TypeError(
                "%s cannot be added to the message" % type(other)
            )

        self.data.append(other)
        if self.has.DataLengthField:
            self.get.DataLengthField.update()
        return self
