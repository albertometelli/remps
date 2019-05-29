import time
from contextlib import contextmanager

from baselines.common import colorize


@contextmanager
def timed(msg):
    print(colorize(msg, color="magenta"))
    tstart = time.time()
    yield
    print(
        colorize(msg + "done in %.3f seconds" % (time.time() - tstart), color="magenta")
    )
