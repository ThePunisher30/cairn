import time


class Timer:
    def __enter__(self):
        self.seconds = 0.0
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()
        self.seconds = self.end_time - self.start_time
        return False  # propagate any exception that occurred in the with block
