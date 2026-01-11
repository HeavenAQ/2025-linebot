import logging
import inspect
from functools import wraps
from typing import Callable


def log_with_frame_info(log_method: Callable):
    """Decorator that appends caller file:line to log messages.

    This version supports instance methods by accepting and forwarding `self`.
    """
    @wraps(log_method)
    def wrapper(self, message: str, *args, **kwargs):
        frame = inspect.currentframe()
        try:
            if frame and frame.f_back:
                caller = frame.f_back
                enhanced_msg = (
                    f"{message} - {caller.f_code.co_filename}:{caller.f_lineno}"
                )
            else:
                enhanced_msg = message
            return log_method(self, enhanced_msg, *args, **kwargs)
        finally:
            del frame

    return wrapper


class Logger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Show debug too
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M",
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    @log_with_frame_info
    def debug(self, message: str):
        return self.logger.debug(message)

    @log_with_frame_info
    def info(self, message: str):
        return self.logger.info(message)

    @log_with_frame_info
    def warning(self, message: str):
        return self.logger.warning(message)

    @log_with_frame_info
    def error(self, message: str):
        return self.logger.error(message)

    @log_with_frame_info
    def critical(self, message: str):
        return self.logger.critical(message)
