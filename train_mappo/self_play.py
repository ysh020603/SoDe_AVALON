import copy
import random
from typing import Any, Dict, List, Optional, Tuple


class SelfPlayPool:
    """
    简单策略池：维护 state_dict 列表，用于对抗采样。
    """

    def __init__(self, *, max_size: int = 5) -> None:
        self.max_size = int(max_size)
        # entry: {name, state_dict, rating, num_games, wins}
        self._items: List[Dict[str, Any]] = []

    def push(self, name: str, state_dict: Dict[str, Any], rating: Optional[float] = None) -> None:
        sd = copy.deepcopy(state_dict)
        name = str(name)
        if rating is None:
            rating = 1000.0
        rating = float(rating)

        # replace if name exists
        for e in self._items:
            if e["name"] == name:
                e["state_dict"] = sd
                e["rating"] = rating
                return

        self._items.append(
            {
                "name": name,
                "state_dict": sd,
                "rating": rating,
                "num_games": 0,
                "wins": 0,
            }
        )

        # keep top by rating (more stable than random drop for Elo top-k)
        self._items.sort(key=lambda x: float(x["rating"]), reverse=True)
        if len(self._items) > self.max_size:
            self._items = self._items[: self.max_size]

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def sample_entry(
        self,
        *,
        strategy: str = "uniform",
        top_k: int = 3,
    ) -> Tuple[str, Dict[str, Any], float]:
        """
        返回：(name, state_dict, rating)
        """
        if not self._items:
            raise RuntimeError("SelfPlayPool is empty.")

        if strategy == "elo_topk":
            k = max(1, int(top_k))
            sorted_items = sorted(self._items, key=lambda x: float(x["rating"]), reverse=True)
            candidates = sorted_items[: min(k, len(sorted_items))]
            e = random.choice(candidates)
        else:
            e = random.choice(self._items)

        return e["name"], copy.deepcopy(e["state_dict"]), float(e["rating"])

    def sample_state_dict(self) -> Dict[str, Any]:
        if not self._items:
            raise RuntimeError("SelfPlayPool is empty.")
        _name, sd, _rating = self.sample_entry(strategy="uniform")
        return sd

    def size(self) -> int:
        return len(self._items)

    def update_rating(self, name: str, new_rating: float) -> None:
        target = str(name)
        new_rating = float(new_rating)
        for e in self._items:
            if e["name"] == target:
                e["rating"] = new_rating
                # keep list ordered for later top-k sampling
                self._items.sort(key=lambda x: float(x["rating"]), reverse=True)
                return

