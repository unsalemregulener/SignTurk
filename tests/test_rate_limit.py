import time
import unittest

from rate_limit import SlidingWindowLimiter


class SlidingWindowLimiterTests(unittest.TestCase):
    def test_blocks_after_the_limit(self):
        limiter = SlidingWindowLimiter(limit=2, window=60)
        self.assertFalse(limiter.blocked("a"))
        limiter.hit("a")
        limiter.hit("a")
        self.assertTrue(limiter.blocked("a"))
        self.assertFalse(limiter.blocked("b"))

    def test_reset_and_window(self):
        limiter = SlidingWindowLimiter(limit=1, window=0.05)
        limiter.hit("a")
        self.assertTrue(limiter.blocked("a"))
        limiter.reset("a")
        self.assertFalse(limiter.blocked("a"))
        limiter.hit("a")
        time.sleep(0.08)
        self.assertFalse(limiter.blocked("a"))


if __name__ == "__main__":
    unittest.main()
