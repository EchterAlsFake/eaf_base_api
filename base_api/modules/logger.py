import os
import sys
import logging
from typing import Union
from curl_cffi.requests import post

def is_android() -> bool:
    return "ANDROID_ROOT" in os.environ and "ANDROID_DATA" in os.environ

def get_log_file_path(filename: str = "app.log") -> str:
    if is_android():
        return os.path.join(os.environ.get("HOME", ""), filename)
    return filename

def send_log_message(ip: str, port: Union[int, str], message: str) -> None:
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
        try:
            log_entry = self.format(record)
            send_log_message(self.ip, self.port, log_entry)
        except Exception:
            self.handleError(record)


def configure_app_logging(
    logger_name: str | None = None,
    log_file: str | None = None,
    level: int = logging.INFO,
    http_ip: str | None = None,
    http_port: str | int | None = None,
    overwrite_file: bool = False
) -> logging.Logger:
    """
    Opinionated helper to set up logging.
    Call this ONCE at the start of your main application.
    """
    # If logger_name is None, it configures the Root Logger (captures everything)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    format_ = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_)

    # Avoid duplicating handlers if this is called twice
    existing_handler_types = [type(h) for h in logger.handlers]

    if logging.StreamHandler not in existing_handler_types:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_file and logging.FileHandler not in existing_handler_types:
        log_file = get_log_file_path(log_file)
        file_mode = 'w' if overwrite_file else 'a'
        fh = logging.FileHandler(log_file, mode=file_mode)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if http_ip and http_port and HTTPLogHandler not in existing_handler_types:
        http_handler = HTTPLogHandler(http_ip, http_port)
        http_handler.setFormatter(formatter)
        logger.addHandler(http_handler)

    return logger