from typing import Any, Type


__all__ = [
    'DataSpace', 'DataSpaceTemplate'
]


class DataSpace(object):
    """
    class for data storing

        Protected attributes:
        ---------------------
        _mul_rules : dict[Any, tuple[str]]
            The dictionary rules that if the reference
            to the argument of the class matches the key,
            then it returns multiple attributes
    """

    _mul_rules: dict[Any, tuple[str]] = {}

    @staticmethod
    def _remove_protected(attrs: dict[str, Any] | set[str]):
        """
        remove strings starts with '_'

        :param attrs: dict or set
            (usually __annotations__ or __dict__)
        :return: clear iterable object
        """
        if isinstance(attrs, dict):
            cleared = {}
            for k, v in attrs.items():
                if not k.startswith('_'):
                    cleared[k] = v

        elif isinstance(attrs, set):
            cleared = set()
            for key in attrs:
                if not key.startswith('_'):
                    cleared.add(key)

        else:
            raise TypeError(f'invalid type: {type(attrs)}')

        return cleared

    @classmethod
    def _public_annotations(cls):
        return cls._remove_protected(cls.__annotations__)

    @classmethod
    def _public_dict(cls):
        return cls._remove_protected(dict(cls.__dict__))

    @classmethod
    def _apply_mul_rule(cls, name):
        """
        :param name: rule key
        :return: tuple with several attributes
        """
        attrs = []
        for attr in cls._mul_rules[name]:
            attrs.append(cls.var(attr))
        return tuple(attrs)

    @classmethod
    def var(cls, name) -> Any | tuple[Any]:
        """
        :param name: attribute or rule name
        :return: value of an attribute or
            tuple of attributes
        """
        if name not in cls.vars() and name in cls._mul_rules:
            return cls._apply_mul_rule(name)
        return object.__getattribute__(cls, name)

    @classmethod
    def vars(cls) -> set[str]:
        """
        :return: set with attributes names
        """
        return set(cls._public_annotations())\
            .union(set(cls._public_dict()))

    @classmethod
    def mul_rules(cls) -> dict[Any, tuple[str]]:
        """
        :return: dict of rules
        """
        return cls._mul_rules


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

    _redirect_rules: dict[Any, str] = {}

    def __init__(self, **variables):
        all_vars = self.vars()
        annot = self.__class__._public_annotations()
        dict_ = self.__class__._public_dict()
        var_names = set(variables)

        var_diff = all_vars.difference(
            var_names.union(dict_))
        if len(var_diff) != 0:
            raise AttributeError(
                f'Attributes {var_diff} is undefined')

        for name, value in variables.items():
            if name in annot and \
                    not isinstance(value, annot[name]):
                raise TypeError(
                    f'The annotation of \'{name}\' is different from '
                    f'the real type (exp/rec): '
                    f'{annot[name]}/{type(value)}'
                )
            object.__setattr__(self, name, value)

    def _var_exists(self, name: str) -> bool:
        """
        check name in the rules

        :param name: attribute name
        :return: exists or not
        """
        return name not in self.vars() and (
                name in self._mul_rules or
                name in self._redirect_rules)

    def _apply_mul_rule(self, name):
        """
        :param name: attribute name in rules
        :return: tuple of attributes
        """
        attrs = []
        for attr in self._mul_rules[name]:
            attrs.append(self.var(attr))
        return tuple(attrs)

    def _redirect(self, name):
        """
        :param name: key name in redirect rules
        :return: attribute value
        """
        return self.var(self._redirect_rules[name])

    def var(self, name) -> Any | tuple[Any]:
        """
        :param name: attribute name
        :return: attribute value or tuple of attributes
        """
        if self._var_exists(name):
            if name in self._mul_rules:
                return self._apply_mul_rule(name)
            elif name in self._redirect_rules:
                return self._redirect(name)
            assert False, 'impossible: %s' % name
        return object.__getattribute__(self, name)

    def redirect_rules(self) -> dict[Any, str]:
        """
        :return: dict of redirect rules
        """
        return self._redirect_rules

    def __getattr__(self, name):
        return self.var(name)

    def __getitem__(self, name):
        return self.var(name)
