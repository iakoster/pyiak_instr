from typing import Any, Type


__all__ = [
    'DataSpace', 'DataSpaceTemplate'
]


class DataSpace(object):
    """
    class for data storage

        Protected attributes:
        ---------------------
        _rules : dict[Any, tuple[str]]
            The dictionary rules that if the reference
            to the argument of the class matches the key,
            then it returns multiple attributes
    """

    _rules: dict[Any, tuple[str]] = {}

    @classmethod
    def _annotations(cls) -> dict[str, Type]:
        return dict(cls.__annotations__)

    @staticmethod
    def _del_protected(attrs: dict[str, Any] | set[str]):
        """
        remove strings starts with '_'

        :param attrs: dict or set
            (usually __annotations__ or __dict__)
        :return: clear iterable object
        """
        match attrs:
            case dict():
                cleared = {}
                for k, v in attrs.items():
                    if not k.startswith('_'):
                        cleared[k] = v

            case set():
                cleared = set()
                for key in attrs:
                    if not key.startswith('_'):
                        cleared.add(key)

            case _:
                raise TypeError(f'Unsupportable type: {type(attrs)}')

        return cleared

    @classmethod
    def _dict(cls) -> dict[str, Any]:
        return dict(cls.__dict__)

    @classmethod
    def _get_rule_attrs(cls, name):
        """
        :param name: rule key
        :return: tuple with several attributes
        """
        attrs = []
        for attr in cls._rules[name]:
            attrs.append(cls.attr(attr))
        return tuple(attrs)

    @classmethod
    def _name(cls) -> str:
        return cls.__name__

    @classmethod
    def attr(cls, name) -> Any | tuple[Any]:
        """
        :param name: attribute or rule name
        :return: value of an attribute or
            tuple of attributes
        """
        if name not in cls.attrs() and name in cls._rules:
            return cls._get_rule_attrs(name)
        return object.__getattribute__(cls, name)

    @classmethod
    def attrs(cls) -> set[str]:
        """
        :return: set with attributes names
        """
        set_annot = cls._del_protected(set(cls._annotations()))
        set_dict = cls._del_protected(set(cls._dict()))
        return set_annot.union(set_dict)

    @classmethod
    def rules(cls) -> dict[Any, tuple[str]]:
        """
        :return: dict of rules
        """
        return cls._rules


class DataSpaceTemplate(DataSpace):
    """
    class of data spaсe template

        Protected attributes:
        ---------------------
        _redirects : dict[Any, str]
            Dictionary of rules, where if the reference
            to the argument of the class matches the key,
            then the attribute specified in the value
    """

    _redirects: dict[Any, str] = {}

    def __init__(self, **attrs):
        def_attrs = self.attrs()
        def_annot = self._del_protected(self._annotations())
        def_dict = self._del_protected(self._dict())
        set_attrs = set(attrs)

        diff_attrs = def_attrs.difference(set_attrs.union(def_dict))
        if diff_attrs:
            raise AttributeError(
                f'Attributes {diff_attrs} is undefined')

        for name, value in attrs.items():
            if name in def_annot and not isinstance(value, def_annot[name]):
                raise TypeError(
                    f'The annotation of \'{name}\' is different from '
                    f'the real type (exp/rec): '
                    f'{def_annot[name]} != {type(value)}'
                )
            object.__setattr__(self, name, value)

    def _check_attr_exists(self, name: str) -> bool:
        """
        check name in the rules

        :param name: attribute name
        :return: exists or not
        """
        return name not in self.attrs() and (
                name in self._rules or name in self._redirects)

    def _get_rule_attrs(self, name):
        """
        :param name: attribute name in rules
        :return: tuple of attributes
        """
        attrs = []
        for attr in self._rules[name]:
            attrs.append(self.attr(attr))
        return tuple(attrs)

    def _redirect(self, name):
        """
        :param name: key name in redirect rules
        :return: attribute value
        """
        return self.attr(self._redirects[name])

    def attr(self, name) -> Any | tuple[Any]:
        """
        :param name: attribute name
        :return: attribute value or tuple of attributes
        """
        if self._check_attr_exists(name):
            if name in self._rules:
                return self._get_rule_attrs(name)
            elif name in self._redirects:
                return self._redirect(name)
        return object.__getattribute__(self, name)

    @property
    def redirects(self) -> dict[Any, str]:
        """
        :return: dict of redirect rules
        """
        return self._redirects

    def __getattr__(self, name):
        if self._check_attr_exists(name):
            if name in self._rules:
                return self._get_rule_attrs(name)
            elif name in self._redirects:
                return self._redirect(name)
        object.__getattribute__(self, name)

    def __getitem__(self, name):
        if self._check_attr_exists(name):
            if name in self._rules:
                return self._get_rule_attrs(name)
            elif name in self._redirects:
                return self._redirect(name)
        return self.__getattribute__(name)
