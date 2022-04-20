import unittest
from pathlib import Path

from pyinstr_iakoster.log import get_logging_dict_config


class TestLogUtilsMethods(unittest.TestCase):

    def test_get_logging_dict_config(self):

        def compare_iter_dicts(exp_dict, act_dict, lvl_names: dict[str, str] = None):
            if lvl_names is None:
                lvl_names = {}

            for lvl_name, lvl in exp_dict.items():
                lvl_names[f'l{len(lvl_names)}'] = lvl_name
                with self.subTest(**lvl_names):
                    self.assertIn(lvl_name, act_dict)

                if lvl_name in act_dict:
                    act_content = act_dict[lvl_name]
                    if isinstance(act_content, dict):
                        compare_iter_dicts(lvl, act_content)
                    else:
                        with self.subTest(**lvl_names):
                            self.assertEqual(lvl, act_content)

        expected = {
            'version': 1,
            'disable_existing_loggers': False,
            'loggers': {
                '': {
                    'level': 'NOTSET',
                    'handlers': ['debug_console_handler', 'info_rotating_file_handler',
                                 'error_file_handler', 'critical_mail_handler'],
                }
            },
            'handlers': {
                'debug_console_handler': {
                    'level': 'DEBUG',
                    'formatter': 'info',
                    'class': 'logging.StreamHandler',
                    'stream': 'ext://sys.stdout',
                },
                'info_rotating_file_handler': {
                    'level': 'INFO',
                    'formatter': 'info',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': Path(r'.\data\log\info.log'),
                    'mode': 'a',
                    'maxBytes': 1048576,
                    'backupCount': 10
                },
                'error_file_handler': {
                    'level': 'WARNING',
                    'formatter': 'error',
                    'class': 'logging.FileHandler',
                    'filename': Path(r'.\data\log\error.log'),
                    'mode': 'a',
                },
                'critical_mail_handler': {
                    'level': 'CRITICAL',
                    'formatter': 'error',
                    'class': 'logging.handlers.SMTPHandler',
                    'mailhost': 'localhost',
                    'fromaddr': 'monitoring@domain.com',
                    'toaddrs': ['dev@domain.com', 'qa@domain.com'],
                    'subject': 'Critical error with application name'
                }
            },
            'formatters': {
                'info': {
                    'format': '%(asctime)s-%(levelname)s %(name)s-%(module)s|%(lineno)s: %(message)s'
                },
                'error': {
                    'format': '%(asctime)s-%(levelname)s %(name)s(%(process)d)-%(module)s|%(lineno)s: %(message)s'
                },
            }
        }

        with self.assertRaises(ValueError) as exc:
            get_logging_dict_config(critical_mail_handler=True)
        self.assertEqual(
            'mailhost and addresses must be specified',
            exc.exception.args[0])
        actual = get_logging_dict_config(
            log_directory=Path(r'.\data\log'),
            critical_mail_handler=True,
            mailhost='localhost',
            mail_from_addr='monitoring@domain.com',
            mail_to_addr=['dev@domain.com', 'qa@domain.com']
        )
        compare_iter_dicts(expected, actual)
        compare_iter_dicts(actual, expected)
