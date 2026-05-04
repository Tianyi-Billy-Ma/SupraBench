"""
Task 3 · Step 01b — 数据准备（Host SMILES-only 溶剂预测）

与 Task 2 同一任务（溶剂预测），但 host 输入只给 SMILES，不给名字/family。
测试 LLM 能否纯粹从分子结构推理出正确溶剂。

数据来源:
  - task2_eval.parquet              Task 2 的主评估集
  - PubChem API + CDEnrichedData    Host SMILES 补全

产出（写入 data/task7/）:
  - host_smiles_dict.json     host 名字 → SMILES 映射（40+ hosts）
  - eval.parquet              过滤后评估集（仅含有 host SMILES 的行）
  - stats.json                数据集统计

用法:
    python scripts/01b_data_prep_task3.py [--root <repo_root>]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Host SMILES 获取: CDEnrichedData + PubChem API
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.lower()
                  .replace("α", "alpha").replace("β", "beta").replace("γ", "gamma"))


def _load_cd_smiles(cd_path: Path) -> dict[str, str]:
    """从 CDEnrichedData 提取 CD 族 host SMILES。"""
    cd = pd.read_csv(cd_path)
    return (
        cd.drop_duplicates(subset=["Host"])[["Host", "IsomericSMILES_Host"]]
        .dropna()
        .set_index("Host")["IsomericSMILES_Host"]
        .to_dict()
    )


def _pubchem_lookup(name: str) -> str | None:
    """通过 PubChem PUG REST 查 SMILES。"""
    variants = [name]
    # 尝试去掉修饰前缀
    base = re.sub(
        r"(?i)^(mono|di|tri|tetra|penta|hexa|hepta|octa|dodeca|hexakis|heptakis)"
        r"\(.*?\)-?", "", name
    ).strip()
    if base != name:
        variants.append(base)

    for v in variants:
        try:
            url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                f"{requests.utils.quote(v)}/property/"
                "IsomericSMILES,CanonicalSMILES/JSON"
            )
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                props = r.json()["PropertyTable"]["Properties"][0]
                smi = (
                    props.get("IsomericSMILES")
                    or props.get("CanonicalSMILES")
                    or props.get("SMILES")
                )
                if smi:
                    return smi
        except Exception:
            pass
        time.sleep(0.3)
    return None


def build_host_smiles_dict(
    hosts: list[str], cd_path: Path
) -> dict[str, str]:
    """合并 CDEnrichedData + PubChem 获取 host SMILES。"""
    result: dict[str, str] = {}

    # Source 1: CDEnrichedData
    cd_dict = _load_cd_smiles(cd_path)
    cd_norm = {_norm(k): (k, v) for k, v in cd_dict.items()}
    for h in hosts:
        nk = _norm(h)
        if nk in cd_norm:
            result[h] = cd_norm[nk][1]
    print(f"[host-smiles] CDEnrichedData 匹配: {len(result)}")

    # Source 2: PubChem
    remaining = [h for h in hosts if h not in result]
    print(f"[host-smiles] 查询 PubChem: {len(remaining)} hosts")
    for i, h in enumerate(remaining):
        smi = _pubchem_lookup(h)
        if smi:
            result[h] = smi
            print(f"  [{i+1}/{len(remaining)}] [OK]  {h}")
        else:
            if i < 50:
                print(f"  [{i+1}/{len(remaining)}] [--]  {h}")
        if i % 5 == 0:
            time.sleep(0.5)

    print(f"[host-smiles] 总计找到: {len(result)} / {len(hosts)}")
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main(root: Path) -> int:
    task2_eval_path = root / "data" / "processed" / "task2_eval.parquet"
    cd_path = root / "CDEnrichedData.csv"
    out_dir = root / "data" / "task7"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(task2_eval_path)
    unique_hosts = df["host"].unique().tolist()
    print(f"[load] task2_eval: {len(df)} 行, {len(unique_hosts)} unique hosts")

    # 构建 host SMILES 字典
    dict_path = out_dir / "host_smiles_dict.json"
    if dict_path.exists():
        print(f"[cache] 已有 {dict_path}, 直接加载")
        host_dict = json.loads(dict_path.read_text())
    else:
        host_dict = build_host_smiles_dict(unique_hosts, cd_path)
        dict_path.write_text(json.dumps(host_dict, indent=2, ensure_ascii=False))
        print(f"[save] host_smiles_dict.json: {len(host_dict)} hosts")

    # 过滤: 只保留有 host SMILES 的行
    df["host_smiles_new"] = df["host"].map(host_dict)
    eval_df = df[df["host_smiles_new"].notna()].copy()
    eval_df["host_smiles"] = eval_df["host_smiles_new"]
    eval_df = eval_df.drop(columns=["host_smiles_new"])

    print(f"\n[filter] 有 host SMILES: {len(eval_df)} / {len(df)} 行 ({len(eval_df)/len(df):.1%})")

    # 保存
    eval_df.to_parquet(out_dir / "eval.parquet", index=False)
    print(f"[save] eval.parquet: {len(eval_df)} 行")

    # 统计
    stats = {
        "total_rows": int(len(eval_df)),
        "unique_hosts": int(eval_df["host"].nunique()),
        "unique_guests": int(eval_df["guest"].nunique()),
        "host_smiles_coverage": 1.0,
        "guest_smiles_coverage": float(eval_df["guest_smiles"].notna().mean()),
        "solvent_distribution": eval_df["solvent_label"].value_counts().to_dict(),
        "host_family_distribution": eval_df["host_family"].value_counts().to_dict(),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print("\n=== solvent_label 分布 ===")
    print(eval_df["solvent_label"].value_counts().to_string())
    print("\n=== host_family 分布 ===")
    print(eval_df["host_family"].value_counts().to_string())
    print(f"\n=== guest_smiles 覆盖率: {stats['guest_smiles_coverage']:.1%} ===")

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()
    sys.exit(main(Path(args.root)))
