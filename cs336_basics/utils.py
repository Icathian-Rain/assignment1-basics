
import time
import functools
from contextlib import contextmanager


def timer(func):
    """装饰器：统计函数执行时间"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"[Timer] {func.__name__}: {elapsed:.4f}s", flush=True)

    return wrapper


@contextmanager
def timeit(name="Block"):
    """上下文管理器：统计代码块执行时间"""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[Timer] {name}: {elapsed:.4f}s", flush=True)



class Timer:
    def __init__(self, name="Timer", auto_print=True, enabled=True):
        self.name = name
        self.auto_print = auto_print
        self.enabled = enabled
        self.reset()

    def start(self):
        if not self.enabled:
            return self
        if self._start is not None:
            raise RuntimeError("Timer is already running")
        self._start = time.perf_counter()
        return self

    def stop(self):
        if not self.enabled:
            return 0.0
        if self._start is None:
            raise RuntimeError("Timer has not started")

        elapsed = time.perf_counter() - self._start
        self.total += elapsed
        self._start = None

        if self.auto_print:
            self.print()

        return elapsed

    def reset(self):
        self._start = None
        self.total = 0.0
        return self

    @property
    def elapsed(self):
        if not self.enabled:
            return 0.0
        if self._start is None:
            return self.total
        return self.total + time.perf_counter() - self._start

    def print(self, message: str = ""):
        if not self.enabled:
            return
        suffix = f" | {message}" if message else ""
        print(f"[Timer] {self.name}: {self.elapsed:.4f}s{suffix}", flush=True)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


import os
import time
import threading
import psutil

class MemoryMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        process = psutil.Process(os.getpid())
        total = 0

        for p in [process] + process.children(recursive=True):
            try:
                total += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        self.peak_rss = max(self.peak_rss, total)

    def _monitor(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak_rss = 0
        self._stop.clear()
        self._sample()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        print(f"[Memory] Peak RSS: {self.peak_rss / 1024**2:.2f} MiB")
        return False


