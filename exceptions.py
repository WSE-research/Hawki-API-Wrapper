class ModelNotFoundException(Exception):
    """Raised when a requested model is not available or not supported."""

    def __init__(self, message: str | None = None):
        super().__init__(message)