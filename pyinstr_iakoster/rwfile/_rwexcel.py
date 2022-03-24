import re
from pathlib import Path
from typing import overload, Any

from ._rwf_utils import *

import openpyxl as opxl
from openpyxl.cell.cell import Cell


__all__ = ['RWExcel']


class RWExcel(object):

    FILENAME_PATTERN = re.compile('\w+.xlsx$')

    def __init__(self, filepath: Path | str, autosave: bool = False):
        """
        :param filepath: path to the excel file
        :param autosave: boolean value, autosave after any changes
        """
        filepath = if_str2path(filepath)
        match_filename(self.FILENAME_PATTERN, filepath)
        create_dir_if_not_exists(filepath)

        self._filepath = filepath
        self._autosave = autosave
        self._xcl = opxl.open(self._filepath)

    def active_sheet(self, title: str) -> None:
        """
        change active sheet by name

        :param title: sheet name
        """
        self._xcl.active = self._xcl[title]

    def save(self) -> None:
        """
        save excel parser to the excel file
        """
        self._xcl.save(self._filepath)

    @classmethod
    def new_empty(
            cls,
            filepath: Path | str,
            autosave: bool = False,
            first_sheet: str = 'Sheet',
    ):
        """
        create new excel file to the filepath with the first sheet.

        Rise error if excel file to the path is already exists.

        :param filepath: path to the excel file
        :param autosave: boolean value, autosave after any changes
        :param first_sheet: name of the first sheet
        :return: new instance of RWExcel
        :rtype: RWExcel
        """
        filepath = if_str2path(filepath)
        match_filename(cls.FILENAME_PATTERN, filepath)
        create_dir_if_not_exists(filepath)
        if filepath.exists():
            raise FileExistsError('Excel file is already exists')

        wb = opxl.Workbook()
        wb.active.title = first_sheet
        wb.save(filepath)
        wb.close()

        return cls(filepath, autosave=autosave)

    @overload
    def cell(self, cell_name: str, value: Any = None) -> Cell:
        """
        :param cell_name: cell name string in excel format
        :param value: value to be set in the cell
        :return: excel cell
        """
        ...

    @overload
    def cell(self, row: int, col: int, value: Any = None) -> Cell:
        """
        :param row: row index
        :param col: col index
        :param value: value to be set in the cell
        :return: excel cell
        """
        ...

    def cell(self, *coords, **kwargs) -> Cell:
        """
        coords can be represented as a string or
        as a tuple of string and column indices

        The row and col indexes start from 0

        It is not guaranteed to work correctly
        when trying to get multiple cells

        :param coords: cell coordinates
        :param kwargs: kwargs
        :return: excel cell
        """

        match coords:
            case (str() as cell_name,):
                cell = self._xcl.active[cell_name]
            case (int() as row, int() as col):
                cell = self._xcl.active.cell(row + 1, col + 1)
            case _:
                raise ValueError(f'Incorrect input: {coords}')
        if 'value' in kwargs:
            cell.value = kwargs['value']
            if self._autosave:
                self.save()
        return cell

    @property
    def filepath(self):
        """
        :return: path to the excel file
        """
        return self._filepath

    @property
    def excel(self):
        """
        :return: excel parser
        """
        return self._xcl

    def __getitem__(self, *coords: str | tuple[int, int]) -> Cell:
        """
        coords can be represented as a string or
        as a tuple of string and column indices

        The row and col indexes start from 0

        :param coords: cell coordinates
        :return: excel cell
        """
        match coords:
            case ((int(), int()),):
                coords = coords[0]
        res = self.cell(*coords)
        if isinstance(res, Cell):
            return res
        else:
            assert False, f'incorrect .cell return: {type(res)}'
