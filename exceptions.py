class ModelNotFoundException(Exception):
    """Raised when a requested model is not available or not supported."""

    def __init__(self, message: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class GlobalTimeoutError(Exception):
    """Raised when the global request timeout is exceeded."""

    def __init__(self, message: str | None = None):
        super().__init__(message)


class CooldownTimeoutError(Exception):
    """Raised when the active cooldown period would exceed the remaining global timeout."""

    def __init__(self, message: str | None = None):
        super().__init__(message)


class UnauthorizedError(Exception):
    """Raised when the upstream API rejects the key with a 401 Unauthorized response."""

    def __init__(self, message: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class RequestFailedError(Exception):
    """Raised when a request fails with a non-retryable upstream error."""

    def __init__(self, message: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code

class EmptyResponseError(Exception):
    """Raised when the upstream API returns an empty response."""

    def __init__(self, message: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code