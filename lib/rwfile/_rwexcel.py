from pathlib import Path
from typing import overload, Any, Iterable

import openpyxl as opxl
import openpyxl.utils as opxl_u
from openpyxl.cell.cell import Cell


class RWExcel(object):

    def __init__(self, filepath: Path, autosave: bool = False):
        self._filepath = filepath
        self._autosave = autosave
        self._xcl = opxl.open(self._filepath)

    def change_active(self, sheet_name: str) -> None:
        self._xcl.active = self._xcl[sheet_name]

    def save(self) -> None:
        self._xcl.save(self._filepath)

    @classmethod
    def create_empty(
            cls,
            filepath: Path,
            sheet_name: str = 'Sheet'
    ) -> None:
        if filepath.exists():
            raise FileExistsError('Excel file is already excists')
        wb = opxl.Workbook()
        wb.active.title = sheet_name
        wb.save(filepath)

    @overload
    def cell(self, cell_name: str, value: Any = None) -> Cell:
        ...

    @overload
    def cell(self, row: int, col: int, value: Any = None) -> Cell:
        ...

    @overload
    def cell(self, row_num: int, value: Iterable = None) -> tuple[Cell]:
        ...

    @overload
    def cell(self, row: slice, col: int, value: Iterable = None) -> tuple[Cell]:
        ...

    def cell(self, *coords, **kwargs) -> Cell | tuple[Cell]:
        coords = coords[0]
        match coords:
            case str() as cell_name:
                cell = self._xcl.active[cell_name]
            case (int() as row, int() as col):
                cell = self._xcl.active.cell(row + 1, col + 1)
            case int() as row:
                cell = self._xcl.active[row + 1]
            case (slice(start=None, stop=None, step=None), int() as col):
                cell = self._xcl.active[
                    opxl_u.cell.get_column_letter(col + 1)]
            case _:
                raise ValueError(f'Incorrect input: {coords}')
        if 'value' in kwargs:
            cell.value = kwargs['value']
            if self._autosave:
                self.save()
        return cell

    @property
    def filepath(self):
        return self._filepath

    @property
    def excel(self):
        return self._xcl

    def __getitem__(self, *coords: str | int | tuple[int, int]) -> Any:
        res = self.cell(*coords)
        if isinstance(res, Iterable):
            return tuple(cell.value for cell in res)
        elif isinstance(res, Cell):
            return res.value
        else:
            assert False, f'incorrect .cell return: {type(res)}'
