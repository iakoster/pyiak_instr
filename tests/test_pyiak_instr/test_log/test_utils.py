import unittest
from pathlib import Path

from src.pyiak_instr.log import get_logging_dict_config


class TestCommonFunctions(unittest.TestCase):
    def test_get_logging_dict_config(self):
        ref = {
            "version": 1,
            "disable_existing_loggers": False,
            "loggers": {
                "": {
                    "level": 'NOTSET',
                    "handlers": [
                        "debug_console_handler",
                        "info_rotating_file_handler",
                        "error_file_handler",
                        "critical_mail_handler",
                    ],
                }
            },
            "handlers": {
                "debug_console_handler": {
                    "level": "DEBUG",
                    "formatter": "info",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
                "info_rotating_file_handler": {
                    "level": "INFO",
                    "formatter": "info",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": Path(r".\data\log\info.log"),
                    "mode": "a",
                    "maxBytes": 1048576,
                    "backupCount": 10
                },
                "error_file_handler": {
                    "level": "WARNING",
                    "formatter": "error",
                    "class": "logging.FileHandler",
                    "filename": Path(r".\data\log\error.log"),
                    "mode": "a",
                },
                "critical_mail_handler": {
                    "level": "CRITICAL",
                    "formatter": "error",
                    "class": "logging.handlers.SMTPHandler",
                    "mailhost": "localhost",
                    "fromaddr": "monitoring@domain.com",
                    "toaddrs": ["dev@domain.com", "qa@domain.com"],
                    "subject": "Critical error with application name"
                }
            },
            "formatters": {
                "info": {
                    "format": (
                        "%(asctime)s-%(levelname)s %(name)s-%(module)s|"
                        "%(lineno)s: %(message)s"
                    )
                },
                "error": {
                    "format": (
                        "%(asctime)s-%(levelname)s %(name)s(%(process)d)-"
                        "%(module)s|%(lineno)s: %(message)s"
                    )
                },
            }
        }

        res = get_logging_dict_config(
            log_directory=Path(r'.\data\log'),
            critical_mail_handler=True,
            mailhost='localhost',
            mail_from_addr='monitoring@domain.com',
            mail_to_addr=['dev@domain.com', 'qa@domain.com']
        )
        self.assertDictEqual(ref, res)

    def test_get_logging_config_dict_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            get_logging_dict_config(critical_mail_handler=True)
        self.assertEqual(
            "mailhost and addresses must be specified",
            exc.exception.args[0]
        )
