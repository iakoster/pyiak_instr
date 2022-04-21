from typing import Any, Type


__all__ = [
    'DataSpace', 'DataSpaceTemplate'
]


class DataSpace(object):
    """
    Class for data storage and structured access to stored values.

    Does not require initialization.
    """

    _mul_rules: dict[Any, tuple[str]] = {}
    """The dictionary rules that if the reference 
    to the argument of the class matches the key,
    then it returns multiple attributes"""

    @staticmethod
    def _remove_protected(attrs: dict[str, Any] | set[str]):
        """
        Remove strings starts with '_'.

        Parameters
        ----------
        attrs: dict of {str: Any} or set of str
            usually __annotations__ or __dict__.

        Returns
        -------
        dict of {str: Any} or set of str
            attrs without protected attributes.
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
    def _public_annotations(cls) -> dict[str, type]:
        """
        Returns
        -------
        dict of {str: type}
            annotations for public attributes.
        """
        return cls._remove_protected(cls.__annotations__)

    @classmethod
    def _public_dict(cls) -> dict[str, Any]:
        """
        Returns
        -------
        dict of {str: Any}
            class dict with public attributes.
        """
        return cls._remove_protected(dict(cls.__dict__))

    @classmethod
    def _apply_mul_rule(cls, name):
        """
        Get attributes by ._mul_rules.

        Parameters
        ----------
        name: Any
            rule key.

        Returns
        -------
        tuple of Any
            tuple with several attributes.

        See Also
        --------
        _mul_rules: access to the values of
            several attributes with a single key
        """
        attrs = []
        for attr in cls._mul_rules[name]:
            attrs.append(cls.var(attr))
        return tuple(attrs)

    @classmethod
    def var(cls, name) -> Any | tuple[Any]:
        """
        Get attribute by name.

        Parameters
        ----------
        name: Any
            attribute name or rule name.

        Returns
        -------
        Any or tuple of Any
            attribute or tuple of attributes
        """

        if name not in cls.vars() and name in cls._mul_rules:
            return cls._apply_mul_rule(name)
        return object.__getattribute__(cls, name)

    @classmethod
    def vars(cls) -> set[str]:
        """
        Get public variables.

        Returns
        -------
        set of str
            attributes names
        """
        return set(cls._public_annotations())\
            .union(set(cls._public_dict()))

    @classmethod
    def mul_rules(cls) -> dict[Any, tuple[str]]:
        """
        Returns
        -------
        dict of {Any: tuple of str}
            multiple rules.
        """
        return cls._mul_rules


class DataSpaceTemplate(DataSpace):
    """
    Class for data storage and structured access to stored values.

    Parameters
    ----------
    **variables
        vars to be set in format {name: value}.

    See Also
    --------
    DataSpace: parent class.
    """

    _redirect_rules: dict[Any, str] = {}
    """Dictionary of rules, where if the reference 
    to the argument of the class matches the key,
    then the attribute specified in the value"""

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
        Check name exists in the rules.

        Parameters
        ----------
        name: str
            attribute name or rule key.

        Returns
        -------
        bool
            name exists in rules.
        """
        return name not in self.vars() and (
                name in self._mul_rules or
                name in self._redirect_rules)

    def _apply_mul_rule(self, name):
        """
        Get attributes by ._mul_rules.

        Parameters
        ----------
        name: Any
            rule key.

        Returns
        -------
        tuple of Any
            tuple with several attributes.

        See Also
        --------
        _mul_rules: access to the values of
            several attributes with a single key.
        """
        attrs = []
        for attr in self._mul_rules[name]:
            attrs.append(self.var(attr))
        return tuple(attrs)

    def _redirect(self, name):
        """
        Get attribute by ._redirects.

        Parameters
        ----------
        name: Any
            key in redirect rules.

        Returns
        -------
        Any
            attribute.
        """
        return self.var(self._redirect_rules[name])

    def var(self, name) -> Any | tuple[Any]:
        """
        Get attribute by name.

        Parameters
        ----------
        name: Any
            attribute name or rule name.

        Returns
        -------
        Any or tuple of Any
            attribute or tuple of attributes.
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
        Returns
        -------
        dict of {Any: str}
            redirect rules.
        """
        return self._redirect_rules

    def __getattr__(self, name):
        return self.var(name)

    def __getitem__(self, name):
        return self.var(name)
