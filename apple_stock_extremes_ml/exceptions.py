import logging


class AppleStockExtremesML(Exception):
    def __init__(self, message: str, traceback: str | None = None) -> None:
        self.code: str = "AppleStockExtremesML"
        self.message: str = message
        self.traceback: str | None = traceback

        super().__init__(self.message)

    def log_error(self) -> None:
        if not self.traceback:
            logging.error(self.__str__())
        else:
            logging.error(f"{self.__str__()}.\n\n Raised by: {self.traceback}")

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class DataInconsistencyException(AppleStockExtremesML):
    def __init__(self, message: str, traceback: str | None = None) -> None:
        super().__init__(message, traceback)
        self.code: str = "DataInconsistencyException"
