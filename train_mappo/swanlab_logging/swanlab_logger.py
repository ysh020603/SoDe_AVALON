from dataclasses import asdict
from typing import Any, Dict, Optional


class SwanLabLogger:
    """
    轻量封装 SwanLab：滚动记录 rollout/update 指标 + 超参。
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._inited = False
        self._swanlab = None

    def init(self) -> None:
        try:
            import swanlab  # type: ignore

            self._swanlab = swanlab
        except Exception:
            self._swanlab = None
            return

        if self._swanlab is None:
            return

        project = getattr(self.cfg, "swanlab_project", "avalon_mappo")
        run_name = getattr(self.cfg, "swanlab_run_name", "") or "mappo_train"
        init_kwargs = {
            "project": project,
            "experiment_name": run_name,
            "config": asdict(self.cfg) if hasattr(self.cfg, "__dict__") else {},
        }
        self._swanlab.init(**init_kwargs)
        self._inited = True

    def log_iter_start(self, iter_idx: int, train_side: str) -> None:
        if self._swanlab is None or not self._inited:
            return
        self._swanlab.log({"iter_idx": iter_idx, "train_side": 1 if train_side == "good" else 0}, step=iter_idx)

    def log_episode(self, *, iter_idx: int, ep_idx: int, train_side: str, stats: Dict[str, Any]) -> None:
        if self._swanlab is None or not self._inited:
            return
        # 这里 stats 是单 episode 的最后一条结构
        if not stats:
            return
        payload = {
            "episode_index": int(ep_idx),
            "final_result_is_good": 1.0 if stats["final_result"] == "good_win" else 0.0,
            "reward_for_train_side": float(stats["reward_for_train_side"]),
            "episode_transitions": float(stats["transitions"]),
        }
        step = iter_idx * 1000 + ep_idx
        self._swanlab.log(payload, step=step)

    def log_iter_end(self, *, iter_idx: int, train_side: str, update_metrics: Dict[str, float]) -> None:
        if self._swanlab is None or not self._inited:
            return
        payload: Dict[str, float] = {"train_side": 1.0 if train_side == "good" else 0.0}
        for k, v in (update_metrics or {}).items():
            try:
                payload[k] = float(v)
            except Exception:
                continue
        self._swanlab.log(payload, step=iter_idx)

    def finish(self) -> None:
        # swanlab 没有统一 close 接口时不做操作
        return

