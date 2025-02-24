from .core.mode import AutoDiffMode
from .core.mode import Mode as Mode
from .core.tensor import Tensor as Tensor

set_mode = AutoDiffMode.set_mode
get_mode = AutoDiffMode.get_mode

__all__ = ["set_mode", "get_mode", "Mode", "Tensor"]