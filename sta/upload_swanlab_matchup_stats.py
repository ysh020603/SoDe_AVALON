#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _sanitize_metric_fragment(s: str, max_len: int = 120) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown"
    # SwanLab metric name 通常允许较多字符，但这里做保守处理：
    s = s.replace("/", "_")
    s = re.sub(r"[^0-9a-zA-Z_.-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "unknown"

def _sanitize_metric_name(s: str, max_len: int = 120) -> str:
    """
    SwanLab columns 创建时对 metric name 更严格；这里对“完整 metric key”
    再做一次清洗（重点是去掉 '/' 层级分隔符，并做长度控制）。
    """
    s = (s or "").strip()
    if not s:
        return "unknown"
    s = s.replace("/", "_")
    s = re.sub(r"[^0-9a-zA-Z_.-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    # 从尾部截断，尽量保留如 good_win_ratio / good_win / total 这类后缀信息
    if len(s) > max_len:
        s = s[-max_len:]
    return s or "unknown"


def _run_stat_script(stat_script: Path, input_paths: List[str], json_out: Path) -> Tuple[bool, str]:
    cmd = [sys.executable, str(stat_script), "--json", str(json_out)]
    if input_paths:
        # 注意：stat_matchup_results 的参数名是 input_paths（位置参数可能不带 --）
        cmd.extend(input_paths)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0 and json_out.exists()
    if ok:
        return True, ""
    err = proc.stderr.strip() or proc.stdout.strip() or f"returncode={proc.returncode}"
    return False, err


def _flatten_metrics(stats_obj: Dict[str, Any]) -> Tuple[Dict[str, float], int]:
    """
    从 stat_matchup_results.py 输出的 json 结构中提取：
    - good_win_ratio
    - good_win / total
    - total
    返回：
    - metrics: swanlab.log 可用的 metrics dict
    - num_metrics: 指标数量
    """
    exp_dirs = stats_obj.get("experiment_dirs") or {}
    metrics: Dict[str, float] = {}
    num_metrics = 0

    for exp_dir, exp_data in exp_dirs.items():
        exp_alias = _sanitize_metric_fragment(Path(exp_dir).name)
        results = exp_data.get("results") or {}
        if not isinstance(results, dict):
            continue

        for matchup_key, final_counts in results.items():
            if not isinstance(final_counts, dict):
                continue
            total = 0
            good_cnt = 0
            for k, v in final_counts.items():
                if not isinstance(v, (int, float)):
                    continue
                total += int(v)
                if k == "good_win":
                    good_cnt = int(v)
            ratio = (good_cnt / total) if total > 0 else 0.0

            matchup_alias = _sanitize_metric_fragment(str(matchup_key))
            base = f"matchup/{exp_alias}/{matchup_alias}"
            metrics[_sanitize_metric_name(f"{base}/good_win_ratio")] = float(ratio)
            metrics[_sanitize_metric_name(f"{base}/good_win")] = float(good_cnt)
            metrics[_sanitize_metric_name(f"{base}/total")] = float(total)
            num_metrics += 3

    return metrics, num_metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="每隔一段时间重新统计 Avalon *_condition.json，并把 good_win_ratio 等指标上传到 SwanLab。"
    )
    parser.add_argument(
        "--input_paths",
        nargs="*",
        default=[],
        help="传给 stat_matchup_results.py 的 input_paths（可为空，表示使用 stat_matchup_results.py 默认路径）。",
    )
    parser.add_argument(
        "--interval_minutes",
        type=float,
        default=10.0,
        help="多久重新统计一次并上传（分钟）。默认 10 分钟。",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="Avalon",
        help="SwanLab project 名称。",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="matchup_good_win_ratio",
        help="SwanLab experiment 名称。",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Avalon matchup stats (periodic upload)",
        help="SwanLab experiment 描述。",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="cloud",
        help="SwanLab init 的 mode（cloud/local/offline/disabled）。默认 cloud。",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="",
        help="SwanLab workspace/组织名（可为空）。",
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default="/tmp/avalon_matchup_stats.json",
        help="中间产物 json 路径（每次覆盖）。",
    )
    args = parser.parse_args()

    # --------- 准备路径 ----------
    cur_dir = Path(__file__).resolve().parent
    stat_script = cur_dir / "stat_matchup_results.py"
    if not stat_script.exists():
        print(f"stat 脚本不存在: {stat_script}")
        return 1

    json_out = Path(args.json_out)

    # --------- 初始化 SwanLab ----------
    try:
        import swanlab  # type: ignore
    except Exception as e:
        print("未安装 swanlab 包，无法上传到 SwanLab。")
        print(f"import swanlab error: {e}")
        return 2

    config = {
        "stat_script": str(stat_script),
        "input_paths": args.input_paths,
        "interval_minutes": args.interval_minutes,
    }
    init_kwargs: Dict[str, Any] = {
        "project": args.project,
        "experiment_name": args.experiment_name,
        "description": args.description,
        "config": config,
        "mode": args.mode,
    }
    if args.workspace:
        init_kwargs["workspace"] = args.workspace
    swanlab.init(**init_kwargs)

    interval_sec = max(1.0, float(args.interval_minutes) * 60.0)
    step = 0
    while True:
        t0 = time.time()
        step += 1
        print(f"[loop step={step}] 重新统计并上传... (sleep interval={interval_sec:.1f}s)")

        ok, err = _run_stat_script(
            stat_script=stat_script, input_paths=args.input_paths, json_out=json_out
        )
        if not ok:
            print(f"[loop step={step}] 统计失败，stderr/stdout: {err}")
            # 统计失败也继续等下次（避免脚本挂死）
            time.sleep(max(1.0, interval_sec - (time.time() - t0)))
            continue

        try:
            stats_obj = json.loads(json_out.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[loop step={step}] 读取 json_out 失败: {e}")
            time.sleep(max(1.0, interval_sec - (time.time() - t0)))
            continue

        metrics, num_metrics = _flatten_metrics(stats_obj)
        if not metrics:
            print(f"[loop step={step}] 未提取到任何指标（num_metrics={num_metrics}），不上传。")
        else:
            # 一次 log 一批，便于在时间轴上对齐
            swanlab.log(metrics, step=step)
            print(f"[loop step={step}] 已上传指标 num_metrics={num_metrics}")

        elapsed = time.time() - t0
        sleep_for = max(1.0, interval_sec - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    raise SystemExit(main())

