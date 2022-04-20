import unittest
from pathlib import Path

import numpy as np

from pyinstr_iakoster.log import (
    NoWork, BlankWork, Work,
    CompletedWorkError, InterruptedWorkError
)


class TestBlankWork(unittest.TestCase):

    def test_work_name(self):
        self.assertEqual('pyinstr_iakoster.log._work.NoWork',
                         BlankWork().work_name)
        self.assertEqual('builtins.print',
                         BlankWork(print).work_name)
        self.assertEqual('pathlib.WindowsPath.mkdir',
                         BlankWork(Path().mkdir).work_name)

    def test_report_empty(self):
        work = BlankWork()
        self.assertEqual(
            'Work report:\n'
            'BlankWork(work=pyinstr_iakoster.log._work.NoWork, '
            'args=(), additional_args=(), kwargs={}, '
            'additional_kwargs={}, iscalled=False)',
            work.report()
        )

    def test_report_basic(self):
        work = BlankWork(
            print, 'qwerty',
            np.zeros((512, 1), dtype=np.float64),
            end='\t')
        work(333, spec=work)
        self.assertEqual(
            'Work report:\n'
            'BlankWork(work=builtins.print, '
            'args=(\'qwerty\', np.array(shape=(512, 1), dtype=float64)), '
            'additional_args=(333,), kwargs={end=\'\\t\'}, '
            'additional_kwargs={spec=self}, iscalled=True)',
            work.report()
        )

    def test_report_steps(self):
        work = BlankWork()
        work()
        work.add_step('step_1')
        work.add_substep('substep_1.1', 'result_1.1')
        work.add_substep('substep_1.2', None)
        work.add_substep('substep_1.3', np.zeros((1, 123)))
        work.add_step()
        work.add_substep('substep_2.1', 123)
        work.add_substep('substep_2.2', b'\x00\x01\x02')
        work.interrupt(KeyboardInterrupt('test'))
        self.assertEqual(
            'Work report:\n'
            'BlankWork(work=pyinstr_iakoster.log._work.NoWork, '
            'args=(), additional_args=(), kwargs={}, '
            'additional_kwargs={}, iscalled=True)\n'
            'Steps:\n1. step_1\n\tsubstep_1.1 result_1.1\n'
            '\tsubstep_1.2 None\n\tsubstep_1.3 '
            'np.array(shape=(1, 123), dtype=float64)\n'
            '2. Without title\n\tsubstep_2.1 123\n'
            '\tsubstep_2.2 bytes(00 01 02)\n'
            'Interrupt: KeyboardInterrupt(\'test\')',
            work.report()
        )

    def test_report_steps_wo_title(self):
        work = BlankWork()
        work.add_substep('substep_1', 'result_1.1')
        work.add_substep('substep_1', 123, next_step=True)
        work.add_substep('substep_2', None)
        self.assertEqual(
            'Work report:\n'
            'BlankWork(work=pyinstr_iakoster.log._work.NoWork, '
            'args=(), additional_args=(), kwargs={}, '
            'additional_kwargs={}, iscalled=False)\n'
            'Steps:\n1.substep_1 result_1.1\n'
            '2.substep_1 123\n2.substep_2 None',
            work.report()
        )

    def test_report_steps_interrupt(self):
        work = BlankWork(NoWork())
        work.add_substep('substep_1', 'result_1.1')
        work.interrupt(KeyboardInterrupt('test'))
        with self.assertRaises(InterruptedWorkError) as exc:
            work.add_substep('', None)
        self.assertIsInstance(exc.exception.args[0], KeyboardInterrupt)
        self.assertEqual(
            'Work was interrupted by KeyboardInterrupt(\'test\')',
            exc.exception.message)
        with self.assertRaises(InterruptedWorkError):
            work.add_substep('', None)
        with self.assertRaises(InterruptedWorkError):
            work.add_step()
        with self.assertRaises(InterruptedWorkError):
            work()
        self.assertEqual(
            'Work report:\n'
            'BlankWork(work=pyinstr_iakoster.log._work.NoWork, '
            'args=(), additional_args=(), kwargs={}, '
            'additional_kwargs={}, iscalled=False)\n'
            'Steps:\n1.substep_1 result_1.1\n'
            'Interrupt: KeyboardInterrupt(\'test\')',
            work.report()
        )

    def test_report_work_completed(self):
        work = BlankWork(lambda x: str(x), 123)
        work()
        with self.assertRaises(CompletedWorkError) as exc:
            work()
        self.assertEqual(
            'Work tests.test_log.test_work.<lambda> is already done',
            exc.exception.message)
        self.assertEqual('tests.test_log.test_work.<lambda>', exc.exception.work_name)
        self.assertEqual(
            'Work report:\n'
            'BlankWork(work=tests.test_log.test_work.<lambda>, '
            'args=(123,), additional_args=(), kwargs={}, '
            'additional_kwargs={}, iscalled=True)',
            work.report()
        )

    def test_interrupt(self):
        work = BlankWork()
        with self.assertRaises(TypeError) as exc:
            work.interrupt(print)
        self.assertEqual(
            'Invalid reason type: <class \'builtin_function_or_method\'>',
            exc.exception.args[0])


class TestWork(unittest.TestCase):

    def test_call(self):

        def test_func():
            self.TEST_VAL = 6
            return self.TEST_VAL

        self.TEST_VAL = 5
        work = Work(test_func)
        self.assertEqual(5, self.TEST_VAL)
        self.assertEqual(6, work())
        self.assertEqual(6, self.TEST_VAL)
        self.assertEqual(
            'Work report:\n'
            'Work(work=tests.test_log.test_work.test_func, '
            'args=(), additional_args=(), kwargs={}, '
            'additional_kwargs={}, iscalled=True)',
            work.report()
        )

