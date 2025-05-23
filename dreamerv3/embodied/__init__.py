try:
    import rich.traceback
    rich.traceback.install()
except ImportError:
    pass

from dreamerv3.embodied import replay
from dreamerv3.embodied import run
from dreamerv3.embodied.core import *
