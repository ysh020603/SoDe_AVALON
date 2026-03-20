import copy
import random
from typing import Any, Dict, List, Optional, Tuple


class SelfPlayPool:
    """
    简单策略池：维护 state_dict 列表，用于对抗采样。
    """

    def __init__(self, *, max_size: int = 5) -> None:
        self.max_size = int(max_size)
        self._items: List[Tuple[str, Dict[str, Any]]] = []

    def push(self, name: str, state_dict: Dict[str, Any]) -> None:
        sd = copy.deepcopy(state_dict)
        self._items.append((str(name), sd))
        if len(self._items) > self.max_size:
            # 随机丢弃旧策略，保持多样性
            drop_idx = random.randrange(0, len(self._items))
            self._items.pop(drop_idx)

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def sample_state_dict(self) -> Dict[str, Any]:
        if not self._items:
            raise RuntimeError("SelfPlayPool is empty.")
        _name, sd = random.choice(self._items)
        return copy.deepcopy(sd)

    def size(self) -> int:
        return len(self._items)

