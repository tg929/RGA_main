#!/usr/bin/env python3
"""
为目录下的每个 txt 结果文件，在每一行的“对接分数”字段之后插入两列：SA 与 QED。

约定行格式（优先）：
SMILES<TAB>ID<TAB>DockScore<TAB>其它字段...

实现要点：
- 优先按制表符分列；若无制表符则退化为按空白分隔；
- DockScore 默认取第3列（index=2），若不满足则在前5列中寻找首个可转为浮点的字段；
- 使用 RDKit 的 QED 以及 utils.sascorer 的 SA 进行计算；
- 计算失败时以 NA 填充；
- 仅在 DockScore 之后插入两列（顺序：SA、QED），保留其余字段与分隔符不变。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit.Chem import rdmolops
# import datamol as dm
import numpy as np

import utils.sascorer as sascorer
from rdkit.Chem import QED

def _looks_like_float(token: str) -> bool:
    try:
        float(token)
        return True
    except Exception:
        return False


def _detect_separator(line: str) -> str:
    return "\t" if "\t" in line else None  # type: ignore[return-value]


def _split_fields(line: str) -> Tuple[str, List[str]]:
    line = line.rstrip("\n")
    sep = _detect_separator(line)
    if sep is not None:
        return sep, line.split(sep)
    parts = re.split(r"\s+", line.strip()) if line.strip() else [""]
    return " ", parts


def _join_fields(sep: str, parts: List[str]) -> str:
    return sep.join(parts) + "\n"


def _find_dock_score_index(parts: List[str]) -> Optional[int]:
    if len(parts) >= 3 and _looks_like_float(parts[2]):
        return 2
    search_upto = min(5, len(parts))
    for idx in range(1, search_upto):
        if _looks_like_float(parts[idx]):
            return idx
    return None


def _calc_qed_sa(smiles: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        from rdkit import Chem
        from rdkit.Chem import QED
        import utils.sascorer as sascorer  
    except Exception:
        return None, None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        try:
            qed_val: Optional[float] = float(QED.qed(mol))
        except Exception:
            qed_val = None

        try:
            sa_val: Optional[float] = float(sascorer.calculateScore(mol))
        except Exception:
            sa_val = None

        return qed_val, sa_val
    except Exception:
        return None, None


def process_file(in_path: str, out_path: str, encoding: str = "utf-8") -> None:
    with open(in_path, "r", encoding=encoding, errors="ignore") as fin, open(
        out_path, "w", encoding=encoding
    ) as fout:
        for line in fin:
            if not line.strip():
                fout.write(line)
                continue

            sep, parts = _split_fields(line)
            if len(parts) < 1:
                fout.write(line)
                continue

            smiles = parts[0]
            score_idx = _find_dock_score_index(parts)
            if score_idx is None:
                fout.write(line)
                continue

            qed_val, sa_val = _calc_qed_sa(smiles)
            sa_str = f"{sa_val:.3f}" if isinstance(sa_val, float) else "NA"
            qed_str = f"{qed_val:.3f}" if isinstance(qed_val, float) else "NA"

            new_parts = parts[: score_idx + 1] + [sa_str, qed_str] + parts[score_idx + 1 :]
            fout.write(_join_fields(sep, new_parts))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="在 Dock 分数后插入 SA 与 QED 两列")
    parser.add_argument("--input_dir", required=True, help="输入目录，包含若干 .txt 文件")
    parser.add_argument("--output_dir", required=True, help="输出目录，将写出同名 .txt 文件")
    parser.add_argument("--pattern", default=".txt", help="只处理文件名包含该子串的文件（默认：.txt)")
    parser.add_argument("--encoding", default="utf-8", help="文件编码(默认:utf-8)")

    args = parser.parse_args(argv)

    in_dir = os.path.abspath(args.input_dir)
    out_dir = os.path.abspath(args.output_dir)
    pattern = str(args.pattern)

    if not os.path.isdir(in_dir):
        print(f"[错误] 输入目录不存在：{in_dir}", file=sys.stderr)
        return 2

    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(in_dir) if pattern in f and os.path.isfile(os.path.join(in_dir, f))]
    if not files:
        print(f"[警告] 输入目录无匹配文件（pattern='{pattern}'）：{in_dir}")

    for fname in sorted(files):
        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)
        try:
            process_file(in_path, out_path, encoding=args.encoding)
            print(f"[完成] {fname}")
        except Exception as exc:
            print(f"[失败] {fname}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


