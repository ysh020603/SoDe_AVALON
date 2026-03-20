#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


POSITIVE_ROLES = {"Merlin", "Percival", "Loyal Servant"}
NEGATIVE_ROLES = {"Morgana", "Assassin", "Minion", "Oberon", "Mordred"}

# 你可以在这里手动填写**多个**默认统计路径（列表）：
# - 每一项可以是任意一个 *_condition.json 文件路径
# - 也可以直接是实验目录（会递归查找所有 *_condition.json）
DEFAULT_INPUT_PATHS = [
    # 示例：
    "/data2/AVALON/SoDe_Avalon_3/logs/Avalon/Qwen3-32B_VS_Qwen3-235B-A22B_7Players",
    "/data2/AVALON/SoDe_Avalon_3/logs/Avalon/Qwen3-32B_VS_Qwen3-235B-A22B_7Players_smallmodel_A",
    "/data2/AVALON/SoDe_Avalon_3/logs/Avalon/New_Qwen3-32B_VS_Qwen3-235B-A22B_7Players_smallmodel_A",
    "/data2/AVALON/SoDe_Avalon_3/logs/Avalon/New_Qwen3-235B-A22B_VS_Qwen3-32B_7Players_smallmodel_A"
    # "/data2/AVALON/SoDe_Avalon_3/logs/Avalon/Qwen3-235B-A22B_VS_glm-4_7_7Players",
    # "/data2/AVALON/SoDe_Avalon_3/logs/Avalon/Qwen3-235B-A22B_VS_glm-4_7_7Players_smallmodel_A",
]



@dataclass(frozen=True)
class FileStat:
    path: str
    final_result: str
    good_models: Tuple[str, ...]
    evil_models: Tuple[str, ...]


def _iter_condition_json_files(root_dir: Path) -> Iterable[Path]:
    for cur_root, _dirs, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith("_condition.json"):
                yield Path(cur_root) / fn


def _extract_models_by_side(doc: dict) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    players_config = doc.get("players_config", {}) or {}
    good: List[str] = []
    evil: List[str] = []

    for _pid, info in players_config.items():
        if not isinstance(info, dict):
            continue
        role = info.get("role")
        model = info.get("model")
        if not role or not model:
            continue
        if role in POSITIVE_ROLES:
            good.append(str(model))
        elif role in NEGATIVE_ROLES:
            evil.append(str(model))

    return tuple(sorted(set(good))), tuple(sorted(set(evil)))


def _load_file_stat(p: Path) -> Optional[FileStat]:
    try:
        with p.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return None

    meta = doc.get("meta", {}) or {}
    final_result = meta.get("final_result")
    if not final_result:
        return None

    good_models, evil_models = _extract_models_by_side(doc)
    return FileStat(
        path=str(p),
        final_result=str(final_result),
        good_models=good_models,
        evil_models=evil_models,
    )


def _infer_experiment_dir(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path
    if input_path.is_file():
        # 典型路径: .../<exp_dir>/<run_id>/<xxx_condition.json>
        return input_path.parent.parent
    return input_path


def _matchup_key(good_models: Tuple[str, ...], evil_models: Tuple[str, ...]) -> str:
    if len(good_models) == 1 and len(evil_models) == 1:
        return f"{good_models[0]}(Good)_VS_{evil_models[0]}(Evil)"
    return f"MIXED__GOOD={'+'.join(good_models) if good_models else 'UNKNOWN'}__EVIL={'+'.join(evil_models) if evil_models else 'UNKNOWN'}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="统计 Avalon *_condition.json：按(好人模型 vs 坏人模型)汇总 final_result 计数"
    )
    parser.add_argument(
        "input_paths",
        type=str,
        nargs="*",
        default=[],
        help="可输入多个路径：每个可以是 *_condition.json 文件或实验目录（会递归查找所有 *_condition.json）",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        type=str,
        default="",
        help="可选：把统计结果写入该 json 文件路径",
    )
    args = parser.parse_args()

    if args.input_paths:
        raw_paths = [p.strip() for p in args.input_paths if p.strip()]
        print("使用命令行提供的路径进行统计：")
        for p in raw_paths:
            print(f"  - {p}")
    else:
        raw_paths = [p.strip() for p in DEFAULT_INPUT_PATHS if p.strip()]
        print("未提供 input_paths，使用 DEFAULT_INPUT_PATHS 中的路径：")
        for p in raw_paths:
            print(f"  - {p}")

    if not raw_paths:
        print("没有可用的输入路径，请在 DEFAULT_INPUT_PATHS 中填写，或在命令行传入。")
        return 1

    # 归一化为实验目录，并去重
    exp_dirs = []
    seen = set()
    for p in raw_paths:
        exp_dir = _infer_experiment_dir(Path(p))
        if not exp_dir.exists():
            print(f"警告：路径不存在，已跳过: {exp_dir}")
            continue
        key = str(exp_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        exp_dirs.append(exp_dir)

    if not exp_dirs:
        print("所有提供的路径都不可用，结束。")
        return 1

    # 对每个实验目录分别统计（不混在一起）
    all_json_result = {}
    has_any_file = False

    for exp_dir in exp_dirs:
        files = list(_iter_condition_json_files(exp_dir))
        if not files:
            print(f"\n实验目录: {exp_dir}")
            print("  在目录中未找到 *_condition.json")
            all_json_result[str(exp_dir)] = {
                "scanned_files": 0,
                "parsed_files": 0,
                "failed_files": 0,
                "results": {},
            }
            continue

        has_any_file = True
        print(f"\n实验目录: {exp_dir}  | 找到 *_condition.json: {len(files)}")

        per_matchup: Dict[str, Counter] = defaultdict(Counter)
        mixed_bucket: Counter = Counter()
        bad_files: List[str] = []

        for p in files:
            st = _load_file_stat(p)
            if st is None:
                bad_files.append(str(p))
                continue
            key = _matchup_key(st.good_models, st.evil_models)
            per_matchup[key][st.final_result] += 1
            if key.startswith("MIXED__"):
                mixed_bucket[st.final_result] += 1

        total_ok = sum(sum(c.values()) for c in per_matchup.values())
        print(f"  扫描文件数: {len(files)}  | 成功解析: {total_ok}  | 失败/跳过: {len(bad_files)}\n")

        for matchup in sorted(per_matchup.keys()):
            c = per_matchup[matchup]
            total = sum(c.values())
            good_cnt = c.get("good_win", 0)
            ratio = (good_cnt / total) if total > 0 else 0.0
            print(f"  [{matchup}]  total={total}  |  good_win={good_cnt}  |  good_win_ratio={ratio:.3f}")
            for result, cnt in c.most_common():
                print(f"    - {result}: {cnt}")
            print("")

        if bad_files:
            print("  解析失败/缺字段文件（最多展示前20个）:")
            for p in bad_files[:20]:
                print(f"    - {p}")
            if len(bad_files) > 20:
                print(f"    ... 还有 {len(bad_files) - 20} 个")
            print("")

        all_json_result[str(exp_dir)] = {
            "scanned_files": len(files),
            "parsed_files": total_ok,
            "failed_files": len(bad_files),
            "results": {k: dict(v) for k, v in per_matchup.items()},
        }

    if not has_any_file:
        print("所有目录中都没有 *_condition.json，结束。")
        return 2

    if args.output_json:
        out = {
            "experiment_dirs": all_json_result,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入统计结果: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

