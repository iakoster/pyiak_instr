from typing import Any, Callable

import numpy as np

from ..exceptions import CompletedWorkError, InterruptedWorkError

__all__ = [
    'NoWork', 'BlankWork', 'Work',
    'CompletedWorkError', 'InterruptedWorkError'
]


class NoWork(object):
    """
    Work class with empty call
    """

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


class BlankWork(object):
    """
    Base class for loging work without work function.

    Attributes
    ----------
    multiple: bool
        if True, the work can be called multiply times.

    Parameters
    ----------
    *args
        arguments for a work function
    multiple: bool, default=False
        if True, the work can be called multiply times.
    **kwargs
        keyword arguments for a work function

    Methods
    -------
    add_step(step=None)
        add new step with or without (if step=None) title
    add_substep(substep, result, next_step=False)
        add new substep witn name and result
    report()
        get a report on the function's work
    """

    def __init__(
            self, *args,
            multiple: bool = False,
            **kwargs):
        self._work: NoWork | Callable = NoWork()
        if len(args) > 0 and isinstance(args[0], Callable):
            work = args[0]
            args = args[1:]
        else:
            work = self._work

        self._work_name = self._get_func_path(work)
        self._work_args = args
        self._add_args = ()
        self._work_kw = kwargs
        self._add_kw = {}

        self.multiple = multiple
        self._iscalled = False
        self._steps = {}
        self._substeps = {}
        self._steps_count = 0

        self._interrupt_reason: Exception | None = None

    def add_step(self, step: str = None):
        """
        Add new step.

        Parameters
        ----------
        step: str, default=None
            name of the step.

        Returns
        -------
        None

        Raises
        ------
        InterruptedWorkError
            if interrupt reason is specified.
        """
        if self._interrupt_reason is not None:
            raise InterruptedWorkError(self._interrupt_reason)
        self._steps_count += 1
        self._steps[self._steps_count] = step
        if self._steps_count not in self._substeps:
            self._substeps[self._steps_count] = []

    def add_substep(self, substep: str, result,
                    next_step: bool = False):
        """
        Add new substep in current step.

        If next_step=True or the number of steps
        is zero then a new step is added without a title.

        Parameters
        ----------
        substep: str
            name of the substep.
        result:
            substep result or another information about substep.
        next_step:
            if True add new step without title.

        Returns
        -------
        None

        Raises
        ------
        InterruptedWorkError
            if interrupt reason is specified.
        """
        if self._interrupt_reason is not None:
            raise InterruptedWorkError(self._interrupt_reason)
        if next_step or self._steps_count == 0:
            self.add_step()
        result = self._fmt_val(result)
        self._substeps[self._steps_count].append(
            (substep, result))

    def interrupt(self, reason):
        """
        Set interrupt reason.

        This blocks .__call__, .add_step, .add_substep methods.

        Parameters
        ----------
        reason
            the reason of the interruption.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            if the reason is not instance of Exception.
        """
        if not issubclass(reason.__class__, BaseException):
            raise TypeError('Invalid reason type: %s' % type(reason))
        self._interrupt_reason = reason

    def report(self) -> str:
        """
        Get a work report.

        Report contains information about
        the work function (including arguments) and
        the work steps to be added by the user.

        Returns
        -------
        str
            report.
        """
        parts = [f'Work report:\n{self._fmt_self()}']
        if self._steps_count and len(self._substeps):
            parts.append(f'Steps:\n{self._fmt_steps()}')
        if self.interrupt_reason is not None:
            parts.append(f'Interrupt: %r' % self._interrupt_reason)
        return '\n'.join(parts)

    def _fmt_val(self, value: Any, repr_str: bool = False) -> str:
        """
        Format the input value to a user-readable form.

        Numpy arrays wil be format as 'np.array(shape=..., dtype=...)'.

        Parameters
        ----------
        value: Any
            value to format.
        repr_str: bool, default=False
            indicates the use a repr for the str instance.

        Returns
        -------
        str
            formatted value.

        Examples
        --------
        An example of how the repr_str argument works
        in the ._fmt_val method
        >>> work = BlankWork()
        ... work._fmt_val('example', repr_str=False)
        'example'
        >>> work._fmt_val('example', repr_str=True)
        "'example'"

        An example of how the method formats a numpy array
        >>> work._fmt_val(np.zeros((12, 15), dtype=np.uint16))
        'np.array(shape=(12, 15), dtype=np.uint16)'
        """
        match value:
            case np.ndarray(shape=shape, dtype=dtype):
                return f'np.array(shape={shape}, dtype={dtype})'
            case str() if not repr_str:
                return value
            case bytes() | bytearray():
                return 'bytes({})'.format(value.hex(' '))
            case _ if id(self) == id(value):
                return 'self'
            case _:
                return repr(value)

    def _fmt_args(self, args: tuple | dict[str, Any]) -> str:
        """
        Format the (keyword) arguments to a user-readable form.

        Parameters
        ----------
        args: tuple of Any or dict of {str: Any}
            (keywords) arguments of the work.

        Returns
        -------
        str
            formatted arguments.
        """

        if isinstance(args, dict):
            return '{%s}' % ', '.join(
                f'{k}={self._fmt_val(v, repr_str=True)}'
                for k, v in args.items())
        else:
            return '(%s%s)' % (
                ', '.join(self._fmt_val(arg, repr_str=True)
                          for arg in args),
                ',' if len(args) == 1 else '')

    def _fmt_self(self):
        """
        Format the self definition in a user-readable form.

        Can be used for .__repr__ method.

        Returns
        -------
        str
            formatted self definition
        """
        args = self._fmt_args(self._work_args)
        add_args = self._fmt_args(self._add_args)
        kwargs = self._fmt_args(self._work_kw)
        add_kwargs = self._fmt_args(self._add_kw)
        return (
            f'{self.__class__.__name__}(work={self._work_name}, '
            f'args={args}, additional_args={add_args}, '
            f'kwargs={kwargs}, additional_kwargs={add_kwargs}, '
            f'iscalled={self._iscalled})'
        )

    def _fmt_steps(self):
        """
        Format the steps for the report in a user-readable form

        Returns
        -------
        str
            formatted steps
        """
        none_in = None in self._steps.values()
        all_none = none_in
        if none_in:
            for title in self._steps.values():
                all_none = title is None
                if not all_none:
                    break

        lines = []
        for step, title in self._steps.items():
            if not all_none:
                lines.append('{}. {}'.format(
                    step, 'Without title' if title is None else title))
            for substep, result in self._substeps[step]:
                lines.append('{}{}{} {}'.format(
                    '\t' if not all_none else '',
                    f'{step}.' if all_none else '',
                    substep, result
                ))

        return '\n'.join(lines)

    @staticmethod
    def _get_func_path(func) -> str:
        """
        Get full name of the func.

        Return value contains module (if exists),
        class name (if exisis and not equal 'module') and
        self name.

        Parameters
        ----------
        func: Any
            basicaly callable object.

        Returns
        -------
        str
            full path which contains module, class and name.
        """
        parts = []
        if hasattr(func, '__module__'):
            parts.append(func.__module__)
        if hasattr(func, '__self__'):
            func_cls = func.__self__.__class__.__name__
            if func_cls != 'module':
                parts.append(func_cls)
        parts.append(
            (func if hasattr(func, '__name__') else
             func.__class__).__name__)
        return '.'.join(parts)

    @property
    def work(self) -> Callable:
        """
        Returns
        -------
        Callable
            work function.
        """
        return self._work

    @property
    def work_name(self) -> str:
        """
        Returns
        -------
        str
            work name as path through modules.
        """
        return self._work_name

    @property
    def args(self) -> tuple[Any, ...]:
        """
        Returns
        -------
        tuple of Any
            arguments for the work that was specified in __init__.
        """
        return self._work_args

    @property
    def additional_args(self) -> tuple[Any, ...]:
        """
        Returns
        -------
        tuple of Any
            additional arguments for the work that
            was specified when work was called.
        """
        return self._add_args

    @property
    def kwargs(self) -> dict[str, Any]:
        """
        Returns
        -------
        dict of {str: Any}
            keywords arguments for the work that
            was specified in __init__.
        """
        return self._work_kw

    @property
    def additional_kwargs(self) -> dict[str, Any]:
        """
        Returns
        -------
        dict of {str: Any}
            additional keywords arguments for the work that
            was specified when work was called.
        """
        return self._add_kw

    @property
    def iscalled(self):
        """
        Returns
        -------
        bool
            flag that the work was called.
        """
        return self._iscalled

    @property
    def interrupt_reason(self) -> Exception | None:
        """
        The reason for the interruption.

        Reason is specified manually by the user.
        None means no interruption.

        Returns
        -------
        Exception or None
            Interruption reason.
        """
        return self._interrupt_reason

    def __call__(self, *additional_args, **additional_kwargs) -> Any:
        """
        Call the work and return the result of the call.

        Sets the iscalled flag after a call.

        Parameters
        ----------
        *additional_args
            additional arguments to be added after init_args.
        **additional_kwargs
            additional keywords arguments to be added after init_kwargs.

        Returns
        -------
        Any
            result of the work.

        Raises
        ------
        InterruptedWorkError
            if interrupt reason is specified.
        CompletedWorkError
            if the work has been called before and
            cannot be called again.
        """
        if self._interrupt_reason is not None:
            raise InterruptedWorkError(self._interrupt_reason)
        if not self.multiple and self._iscalled:
            raise CompletedWorkError(self._work_name)
        self._add_args = additional_args
        self._add_kw = additional_kwargs
        result = self._work(
            *self._work_args, *self._add_args,
            **self._work_kw, **self._add_kw)
        self._iscalled = True
        return result


class Work(BlankWork):
    """
    Parameters
    ----------
    work: Callable
        work function
    *args
        arguments for a work function
    multiple: bool, default=False
        if True, the work can be called multiply times.
    **kwargs
        keyword arguments for a work function

    See Also
    --------
    BlankWork: basic class of the work.
    """

    def __init__(
            self, work: Callable, *args,
            multiple: bool = False, **kwargs):
        super(Work, self).__init__(
            work, *args, multiple=multiple, **kwargs)
        self._work = work
