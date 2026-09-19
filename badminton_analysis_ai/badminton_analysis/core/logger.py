import logging
import inspect
from functools import wraps
from typing import Any, TypeVar, Protocol, cast

from api.logging_config import json_stream_handler


class _LogMethod(Protocol):
    def __call__(self, __self: Any, __message: str) -> None: ...


F = TypeVar("F", bound=_LogMethod)


def log_with_frame_info(log_method: F) -> F:
    @wraps(log_method)
    def wrapper(self: Any, message: str) -> None:
        frame = inspect.currentframe()
        try:
            if frame and frame.f_back:
                caller = frame.f_back
                enhanced_message = (
                    f"{message} - {caller.f_code.co_filename}:{caller.f_lineno}"
                )
                log_method(self, enhanced_message)
            else:
                log_method(self, message)
        finally:
            del frame

    return cast(F, wrapper)


class Logger:
    def __init__(self, name: str, level: int = logging.INFO) -> None:
        self.logger: logging.Logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        if not self.logger.handlers:
            # Same JSON lines as the service's root logger, so these messages
            # keep their severity and request correlation in Cloud Logging.
            self.logger.addHandler(json_stream_handler())

    @log_with_frame_info
    def debug(self, message: str) -> None:
        self.logger.debug(message)

    @log_with_frame_info
    def info(self, message: str) -> None:
        self.logger.info(message)

    @log_with_frame_info
    def warning(self, message: str) -> None:
        self.logger.warning(message)

    @log_with_frame_info
    def error(self, message: str) -> None:
        self.logger.error(message)

    @log_with_frame_info
    def critical(self, message: str) -> None:
        self.logger.critical(message)
