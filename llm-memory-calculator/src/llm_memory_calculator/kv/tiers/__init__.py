"""One module per tier. Each takes the plan context, decides, and records why."""

from .t0 import plan_t0
from .t1 import plan_t1, t1_mode
from .t2 import plan_t2
from .t3 import plan_t3

__all__ = ["plan_t0", "plan_t1", "plan_t2", "plan_t3", "t1_mode"]
