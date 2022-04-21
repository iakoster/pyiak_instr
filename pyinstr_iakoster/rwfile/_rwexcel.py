import re
from pathlib import Path
from typing import overload, Any

from ._utils import *

import openpyxl as opxl
from openpyxl.cell.cell import Cell


__all__ = ['RWExcel']


class RWExcel(object):
    """
    Class for reading and writing to the excel file as *.xlsx.

    In the class used a `openpyxl` library.

    Parameters
    ----------
    filepath: Path or path-like str
        path to the *.xlsx excel file.
    autosave: bool, default=False
        if True save to file after any changes.

    Raises
    ------
    FilepathPatternError:
        if the filepath does not lead to the *.xlsx file.
    """

    FILENAME_PATTERN = re.compile('\S+.xlsx$')

    def __init__(self, filepath: Path | str, autosave: bool = False):
        filepath = if_str2path(filepath)
        match_filename(self.FILENAME_PATTERN, filepath)
        create_dir_if_not_exists(filepath)

        self._filepath = filepath
        self._autosave = autosave
        self._xcl = opxl.open(self._filepath)

    def active_sheet(self, title: str) -> None:
        """
        Change active sheet by name.

        Parameters
        ----------
        title: str
            sheet name.
        """
        self._xcl.active = self._xcl[title]

    def save(self) -> None:
        """
        Save excel parser to the excel file.
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
        Create new excel file to the filepath with the first sheet.

        Parameters
        ----------
        filepath: Path or path-like str
            path to the *.xlsx excel file.
        autosave: bool, default=False
            if True save to file after any changes.
        first_sheet: str, default='Sheet'
            name of the first sheet.

        Returns
        -------
        RWExcel
            new instance of the RWExcel.

        Raises
        ------
        FilepathPatternError:
            if the filepath does not lead to the *.xlsx file.
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
        Parameters
        ----------
        cell_name: str
            cell name in excel format (e.g. 'A1' or 'A1:B2').
        value: Any, default=None
            value to be set in the cell.

        Returns
        -------
        Cell
            excel cell
        """
        ...

    @overload
    def cell(self, row: int, col: int, value: Any = None) -> Cell:
        """
        Parameters
        ----------
        row: int
            row index.
        col: int
            column index.
        value: Any
            Value to be set in the cell.

        Returns
        -------
        Cell
            excel cell.
        """
        ...

    def cell(self, *coords, **kwargs) -> Cell:
        """
        Coords can be represented as a string or
        as a tuple of strings and column indices.

        The row and column indexes start from 0.

        It is not guaranteed to work correctly
        when trying to get multiple cells.

        Parameters
        ----------
        *coords
            cell coordinates.
        **kwargs
            At this point,only the 'value' key
            can be included. The others will be
            ignored.

        Returns
        -------
        Cell
            excel cell.
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
        Returns
        -------
        Path
            path to the excel file
        """
        return self._filepath

    @property
    def excel(self):
        """
        Returns
        -------
        openpyxl.Workbook
            excel parser
        """
        return self._xcl

    def __getitem__(self, *coords: str | tuple[int, int]) -> Cell:
        """
        Call .cell method with *coords.

        Parameters
        ----------
        *coords
            cell coordinates.

        Returns
        -------
        Cell
            excel cell.

        Raises
        ------
        AssertionError
            if .cell method return not a Cell instance

        See Also
        --------
        .cell: get excel cell.
        """
        match coords:
            case ((int(), int()),):
                coords = coords[0]
        res = self.cell(*coords)
        if isinstance(res, Cell):
            return res
        else:
            assert False, f'incorrect .cell return: {type(res)}'
