"""
Unit tests for the ModelUsage class in wrapper.py.
"""

import unittest
from datetime import datetime, timedelta
from wrapper import ModelUsage


class TestModelUsageInit(unittest.TestCase):
    """Tests for __init__"""

    def test_model_name_stored(self):
        usage = ModelUsage("gpt-4o")
        self.assertEqual(usage.model_name, "gpt-4o")

    def test_timestamps_empty_on_init(self):
        usage = ModelUsage("gpt-4o")
        self.assertEqual(usage.getTimestamps(), [])


class TestModelUsageGetModelName(unittest.TestCase):
    """Tests for getModelName()"""

    def test_different_model_names(self):
        for name in ["gpt-4o", "gpt-4o-mini", "o1-mini", "gemini-2.0-flash"]:
            self.assertEqual(ModelUsage(name).getModelName(), name)


class TestModelUsageAdd(unittest.TestCase):
    """Tests for add()"""

    def test_add_default_timestamp_is_recent(self):
        usage = ModelUsage("m")
        before = datetime.now()
        usage.add()
        after = datetime.now()
        ts = usage.getTimestamps()
        self.assertEqual(len(ts), 1)
        self.assertGreaterEqual(ts[0], before)
        self.assertLessEqual(ts[0], after)

    def test_add_explicit_datetime(self):
        usage = ModelUsage("m")
        dt = datetime.now() - timedelta(hours=1)
        usage.add(dt)
        self.assertIn(dt, usage.getTimestamps())

    def test_add_multiple_timestamps_accumulate(self):
        usage = ModelUsage("m")
        now = datetime.now()
        for i in reversed(range(5)):
            usage.add(now - timedelta(minutes=i))
        self.assertEqual(len(usage.getTimestamps()), 5)

    def test_add_mixes_valid_and_expired(self):
        usage = ModelUsage("m")
        now = datetime.now()
        usage.add(now - timedelta(hours=25))  # expired
        usage.add(now - timedelta(hours=1))   # valid
        usage.add(now - timedelta(minutes=30))  # valid
        ts = usage.getTimestamps()
        self.assertEqual(len(ts), 2)

class TestModelUsageCleanup(unittest.TestCase):
    """Tests for _cleanup()"""

    def test_removes_entries_older_than_24h(self):
        usage = ModelUsage("m")
        now = datetime.now()
        usage._timestamps.append(now - timedelta(hours=24, seconds=1))
        usage._timestamps.append(now - timedelta(hours=23, minutes=59))
        usage._cleanup()
        self.assertEqual(len(usage._timestamps), 1)

    def test_keeps_entries_within_24h(self):
        usage = ModelUsage("m")
        now = datetime.now()
        for minutes in [1, 60, 120, 1439]:  # 1m to 23h59m ago
            usage._timestamps.append(now - timedelta(minutes=minutes))
        usage._cleanup()
        self.assertEqual(len(usage._timestamps), 4)

    def test_empty_deque_safe(self):
        usage = ModelUsage("m")
        usage._cleanup()  # should not raise
        self.assertEqual(usage.getTimestamps(), [])


class TestModelUsageGetUsagePerHour(unittest.TestCase):
    """Tests for getUsagePerHour()"""

    def test_returns_24_values(self):
        usage = ModelUsage("m")
        result = usage.getUsagePerHour()
        self.assertEqual(len(result), 24)

    def test_all_zeros_when_empty(self):
        usage = ModelUsage("m")
        self.assertEqual(usage.getUsagePerHour(), [0] * 24)

    def test_recent_call_counted_in_first_slot(self):
        """An event 30m ago should appear in result[0] (last 1 hour)."""
        usage = ModelUsage("m")
        usage.add(datetime.now() - timedelta(minutes=30))
        result = usage.getUsagePerHour()
        self.assertEqual(result[0], 1)

    def test_event_1h30m_ago_slot_placement(self):
        """An event 90m ago must NOT be in result[0] (last 1h) but MUST be in result[1] (last 2h)."""
        usage = ModelUsage("m")
        usage.add(datetime.now() - timedelta(hours=1, minutes=30))
        result = usage.getUsagePerHour()
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 1)

    def test_total_at_slot_23_equals_all_events(self):
        """result[23] (last 24h) must equal total valid event count."""
        usage = ModelUsage("m")
        now = datetime.now()
        timestamps = [
            now - timedelta(hours=23),
            now - timedelta(hours=12),
            now - timedelta(hours=5),
            now - timedelta(minutes=10)
        ]
        for ts in timestamps:
            usage.add(ts)
        result = usage.getUsagePerHour()
        self.assertEqual(result[23], len(timestamps))

    def test_expired_events_not_counted(self):
        """Events older than 24h must not appear in any slot."""
        usage = ModelUsage("m")
        usage._timestamps.append(datetime.now() - timedelta(hours=25))
        result = usage.getUsagePerHour()
        self.assertEqual(result[23], 0)

    def test_multiple_events_same_hour(self):
        """Multiple events in the same hour are all counted."""
        usage = ModelUsage("m")
        now = datetime.now()
        for _ in range(7):
            usage.add(now - timedelta(minutes=20))
        result = usage.getUsagePerHour()
        self.assertEqual(result[0], 7)

    def test_slot_counts_reflect_correct_boundaries(self):
        """
        Place exactly 1 event in each of hours 1-3 and verify slot values.
        """
        usage = ModelUsage("m")
        now = datetime.now()
        usage.add(now - timedelta(minutes=150))   # within last 3h
        usage.add(now - timedelta(minutes=90))    # within last 2h
        usage.add(now - timedelta(minutes=30))    # within last 1h

        result = usage.getUsagePerHour()
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 2)
        self.assertEqual(result[2], 3)
        for i in range(3, 24):
            self.assertEqual(result[i], 3)

class SummaryTestResult(unittest.TextTestResult):
    """Custom TestResult that prints a per-test status table after the run."""

    STATUS_PASS  = "PASS"
    STATUS_FAIL  = "FAIL"
    STATUS_ERROR = "ERROR"
    STATUS_SKIP  = "SKIP"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._summary: list[tuple[str, str, str | None]] = []

    def _short_name(self, test: unittest.TestCase) -> str:
        return f"{type(test).__name__}.{test._testMethodName}"

    def addSuccess(self, test):
        super().addSuccess(test)
        self._summary.append((self.STATUS_PASS, self._short_name(test), None))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._summary.append((self.STATUS_FAIL, self._short_name(test), self._exc_info_to_string(err, test)))

    def addError(self, test, err):
        super().addError(test, err)
        self._summary.append((self.STATUS_ERROR, self._short_name(test), self._exc_info_to_string(err, test)))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._summary.append((self.STATUS_SKIP, self._short_name(test), reason))

    def print_summary(self):
        colors = {
            self.STATUS_PASS:  "\033[32m",
            self.STATUS_FAIL:  "\033[31m",
            self.STATUS_ERROR: "\033[31m",
            self.STATUS_SKIP:  "\033[33m",
        }
        reset = "\033[0m"
        width = max((len(name) for _, name, _ in self._summary), default=40)

        print("\n" + "=" * (width + 14))
        print(f"  {'TEST':<{width}}  STATUS")
        print("=" * (width + 14))
        for status, name, _ in self._summary:
            color = colors.get(status, "")
            print(f"  {name:<{width}}  {color}{status}{reset}")
        print("-" * (width + 14))

        counts = {s: sum(1 for st, *_ in self._summary if st == s) for s in colors}
        totals = "  ".join(f"{color}{s}: {counts[s]}{reset}" for s, color in colors.items() if counts[s])
        print(f"  Total: {len(self._summary)}   {totals}")
        print("=" * (width + 14) + "\n")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__("test_model_usage"))
    runner = unittest.TextTestRunner(resultclass=SummaryTestResult, verbosity=2, buffer=False)
    result = runner.run(suite)
    result.print_summary()
