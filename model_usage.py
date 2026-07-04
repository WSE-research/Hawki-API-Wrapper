from collections import deque
from datetime import datetime, timedelta
from logger_config import logger


class ModelUsage:
    # Holds timestamps for one model, and provides method to get usage per hour for the last 24 hours
    # Care for testing: add() simply appends the timestamp to the right of the deque as no sorting is needed for this use case. Thus, the oldest timestamp is always on the left and the newest on the right.

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._timestamps = deque()

    def add(self, dt: datetime | None = None):
        if dt is None:
            dt = datetime.now()
        self._timestamps.append(dt)
        self._cleanup()
    
    def getTimestamps(self):
        self._cleanup()
        return list(self._timestamps)

    def getModelName(self):
        return self.model_name

    def is_empty(self) -> bool:
        """Return True if there are no timestamps within the last 24 hours."""
        self._cleanup()
        return len(self._timestamps) == 0

    def _cleanup(self):
        cutoff = datetime.now() - timedelta(hours=24)
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def getUsagePerHour(self): # Return cumulative counts for usage of the past 24 hours in a list of length 24, where index 0 is the count for the last hour, index 1 for the last 2 hours, and so on
        self._cleanup()
        now = datetime.now()

        result = []
        total = len(self._timestamps)
        idx = total - 1  # start from newest

        # For hour = 1 to 24
        for hours in range(1, 25):
            cutoff = now - timedelta(hours=hours)

            # Move index left while timestamps are >= cutoff
            while idx >= 0 and self._timestamps[idx] >= cutoff:
                idx -= 1

            # total elements minus elements before cutoff
            result.append(total - (idx + 1))
            logger.debug(f"Model {self.model_name} - Usage in last {hours} hour(s): {result[hours -1]}")

        return result
