from typing import TypeVar, Generic


__all__ = ["UseApi", "API_TYPE"]


API_TYPE = TypeVar("API_TYPE")


class UseApi(Generic[API_TYPE]):

    _api: API_TYPE

    @property
    def api(self) -> API_TYPE:
        """An instance of an object that is used as an API."""
        return self._api
