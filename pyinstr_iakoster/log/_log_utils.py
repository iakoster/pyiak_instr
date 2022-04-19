from pathlib import Path


__all__ = ['get_logging_dict_config']


def get_logging_dict_config(
        version: int = 1,
        disable_existing_loggers: bool = False,
        debug_console_handler: bool = True,
        log_directory: Path = Path(),
        info_rotating_file_handler: bool = True,
        info_rotating_filename: str = 'info.log',
        error_file_handler: bool = True,
        error_filename: str = 'error.log',
        critical_mail_handler: bool = False,
        mailhost: str = None, mail_from_addr: str = None,
        mail_to_addr: str | list[str] = None,
        app_name: str = 'application name'
) -> dict:
    """
    create and return a dict with standard settings that
    can be used almost always

    Parameters
    ----------
    version: int
        value representing the schema version
    disable_existing_loggers: bool
        If specified as False, loggers which exist when this call
        is made are left enabled.
    debug_console_handler: bool
        if set to True, enables the handler that writes everything
        whose level starts with DEBUG to the console
    log_directory: Path
        directory in which the log files will be saved
    info_rotating_file_handler: bool
        if set to True, enables the handler that writes everything
        whose level starts with INFO to the info file
    info_rotating_filename: str
        file name for info_rotating_file_handler
    error_file_handler: bool
        if set to True, enables the handler that writes everything
        whose level starts with WARNING to the error file
    error_filename: str
        file name for error_file_handler
    critical_mail_handler: bool
        if set to True, enables the handler that sends messages
        with critical errors. Must be specified mailhost,
        mail_from_addr and mail_to_addr
    mailhost: str
        mailhost for the critical_mail_handler
    mail_from_addr: str
        outgoing address for the critical_mail_handler
    mail_to_addr: str or list of str
        address or addresses of the recipients of the
        critical_mail_handler messages
    app_name: str
        application name for a subject in the mail

    Returns
    -------
    dict
        dictionary with logger cobfiguration for the method
        logging.config.dictConfig

    Raises
    ------
    ValueError
        if critical_mail_handler is True and at least
        one of mailhost, mail_from_addr, mail_to_addr is None
    """
    if critical_mail_handler and None in (
            mailhost, mail_from_addr, mail_to_addr):
        raise ValueError('There is must be written mailhost and addresses')
    handlers_names = []
    handlers = {}

    if debug_console_handler:
        handlers_names.append('debug_console_handler')
        handlers.update(debug_console_handler={
            'level': 'DEBUG',
            'formatter': 'info',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        })

    if info_rotating_file_handler:
        handlers_names.append('info_rotating_file_handler')
        handlers.update(info_rotating_file_handler={
            'level': 'INFO',
            'formatter': 'info',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': log_directory / info_rotating_filename,
            'mode': 'a',
            'maxBytes': 1048576,
            'backupCount': 10
        })

    if error_file_handler:
        handlers_names.append('error_file_handler')
        handlers.update(error_file_handler={
            'level': 'WARNING',
            'formatter': 'error',
            'class': 'logging.FileHandler',
            'filename': log_directory / error_filename,
            'mode': 'a',
        })

    if critical_mail_handler:
        handlers_names.append('critical_mail_handler')
        handlers.update(critical_mail_handler={
            'level': 'CRITICAL',
            'formatter': 'error',
            'class': 'logging.handlers.SMTPHandler',
            'mailhost': mailhost,
            'fromaddr': mail_from_addr,
            'toaddrs': mail_to_addr,
            'subject': f'Critical error with {app_name}'
        })

    return dict(
        version=version,
        disable_existing_loggers=disable_existing_loggers,
        loggers={
            '': dict(
                level='NOTSET',
                handlers=handlers_names
            )
        },
        handlers=handlers,
        formatters={
            'info': {'format': '%(asctime)s-%(levelname)s %(name)s-%(module)s|%(lineno)s: %(message)s'},
            'error': {'format': '%(asctime)s-%(levelname)s %(name)s(%(process)d)-%(module)s|%(lineno)s: %(message)s'}
        }
    )

# settings = {
#     'version': 1,
#     'disable_existing_loggers': False,
#     'loggers': {
#         '': {  # root logger
#             'level': 'NOTSET',
#             'handlers': ['debug_console_handler', 'info_rotating_file_handler', 'error_file_handler'],
#         }
#     },
#     'handlers': {
#         'debug_console_handler': {
#             'level': 'DEBUG',
#             'formatter': 'info',
#             'class': 'logging.StreamHandler',
#             'stream': 'ext://sys.stdout',
#         },
#         'info_rotating_file_handler': {
#             'level': 'INFO',
#             'formatter': 'info',
#             'class': 'logging.handlers.RotatingFileHandler',
#             'filename': info_path,
#             'mode': 'a',
#             'maxBytes': 1048576,
#             'backupCount': 10
#         },
#         'error_file_handler': {
#             'level': 'WARNING',
#             'formatter': 'error',
#             'class': 'logging.FileHandler',
#             'filename': crit_path,
#             'mode': 'a',
#         },
#         'critical_mail_handler': {
#             'level': 'CRITICAL',
#             'formatter': 'error',
#             'class': 'logging.handlers.SMTPHandler',
#             'mailhost': 'localhost',
#             'fromaddr': 'monitoring@domain.com',
#             'toaddrs': ['dev@domain.com', 'qa@domain.com'],
#             'subject': 'Critical error with application name'
#         }
#     },
#     'formatters': {
#         'info': {
#             'format': '%(asctime)s-%(levelname)s %(name)s-%(module)s|%(lineno)s: %(message)s'
#         },
#         'error': {
#             'format': '%(asctime)s-%(levelname)s %(name)s(%(process)d)-%(module)s|%(lineno)s: %(message)s'
#         },
#     }
# }
