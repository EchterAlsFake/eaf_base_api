import os
import sys
import logging

from typing import Union, Dict
from curl_cffi.requests import post


# Default formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
loggers: Dict[str, logging.Logger] = {}

def is_android() -> bool:
    """Detects if the script is running on an Android device."""
    return "ANDROID_ROOT" in os.environ and "ANDROID_DATA" in os.environ

def get_log_file_path(filename: str = "app.log") -> str:
    """Returns a valid log file path that works on Android and other OS."""
    if is_android():
        return os.path.join(os.environ["HOME"], filename)  # Internal app storage
    return filename  # Default for Linux, Windows, Mac


def send_log_message(ip: str, port: Union[int, str], message: str) -> None:
    """Sends a log message to a remote server via HTTP or HTTPS."""
    try:
        url = f"https://{ip}:{port}/feedback"
        post(url, json={"message": message}, timeout=5, impersonate="chrome110")
    except Exception as e:
        print(f"Failed to send log to {ip}:{port} - {e}", file=sys.stderr)


class HTTPLogHandler(logging.Handler):
    """Custom log handler that sends logs to a remote HTTP server."""
    def __init__(self, ip: str, port: Union[int, str]) -> None:
        super().__init__()
        self.ip = ip
        self.port = port

    def emit(self, record: logging.LogRecord) -> None:
        log_entry = self.format(record)
        send_log_message(self.ip, self.port, log_entry)


def setup_logger(name: str, log_file: str | None = None, level: int = logging.CRITICAL,
                 http_ip: str | None = None, http_port: str | int | None = None) -> logging.Logger:
    """Creates or updates a logger for a specific module."""
    format_ = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if name in loggers:
        logger = loggers[name]
        logger.setLevel(level)

        file_handler_exists = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        http_handler_exists = any(isinstance(h, HTTPLogHandler) for h in logger.handlers)

        if log_file and not file_handler_exists:
            log_file = get_log_file_path(log_file)
            fh = logging.FileHandler(log_file, mode='a')
            fh.setFormatter(logging.Formatter(format_))
            logger.addHandler(fh)

        if http_ip and http_port and not http_handler_exists:
            http_handler = HTTPLogHandler(http_ip, http_port)
            http_handler.setFormatter(logging.Formatter(format_))
            logger.addHandler(http_handler)

        return logger

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(format_))
        logger.addHandler(ch)

    if log_file:
        log_file = get_log_file_path(log_file)
        if not hasattr(sys, '_first_run'):
            setattr(sys, '_first_run', True)
            file_mode = 'w'
        else:
            file_mode = 'a'
        fh = logging.FileHandler(log_file, mode=file_mode)
        fh.setFormatter(logging.Formatter(format_))
        logger.addHandler(fh)

    if http_ip and http_port:
        http_handler = HTTPLogHandler(http_ip, http_port)
        http_handler.setFormatter(logging.Formatter(format_))
        logger.addHandler(http_handler)

    loggers[name] = logger
    return logger
