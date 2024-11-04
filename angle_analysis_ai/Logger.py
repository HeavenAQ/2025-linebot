import logging
import inspect


class Logger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "{asctime} - {levelname} - {filename}:{lineno} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M",
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def debug(self, message: str):
        if frame := inspect.currentframe():
            frame_back = frame.f_back
            assert frame_back is not None
            self.logger.debug(
                f"{message} - {frame_back.f_code.co_filename}:{frame_back.f_lineno}"
            )

    def info(self, message: str):
        if frame := inspect.currentframe():
            frame_back = frame.f_back
            assert frame_back is not None
            self.logger.info(
                f"{message} - {frame_back.f_code.co_filename}:{frame_back.f_lineno}"
            )

    def warning(self, message: str):
        if frame := inspect.currentframe():
            frame_back = frame.f_back
            assert frame_back is not None
            self.logger.warning(
                f"{message} - {frame_back.f_code.co_filename}:{frame_back.f_lineno}"
            )

    def error(self, message: str):
        if frame := inspect.currentframe():
            frame_back = frame.f_back
            assert frame_back is not None
            self.logger.error(
                f"{message} - {frame_back.f_code.co_filename}:{frame_back.f_lineno}"
            )

    def critical(self, message: str):
        if frame := inspect.currentframe():
            frame_back = frame.f_back
            assert frame_back is not None
            self.logger.critical(
                f"{message} - {frame_back.f_code.co_filename}:{frame_back.f_lineno}"
            )
