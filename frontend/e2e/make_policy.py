"""
画面の通しのテスト用に, 骨組みのプロジェクトの実験の中に方策の一式を置く

ML-Quest の方策の骨組みの学習のように, 実験 (outputs/<名前>/<日付>/<時刻>)
の中の policy に方策の一式を書き出し, モデルの定義を一式の model.yaml に
写して metadata.model.spec に書く. 一式は ML-Inference のテストの方策の一式
(ML-Found で書き出した CartPole の方策) を元にする
"""

import json
import shutil
import sys
from pathlib import Path

# ML-Inference のテストの方策の一式
SOURCE = (
    Path(__file__).resolve().parents[2] / "ML-Inference" / "tests"
    / "policies" / "cartpole"
)

# 骨組みが一式に写すモデルの定義のファイルの名前
MODEL_SPEC = "model.yaml"


def main(project: Path) -> None:
    """
    プロジェクトの実験の中に, 骨組みの形の方策の一式を置く関数

    Parameters
    ----------
    project: Path
        骨組みのプロジェクトのディレクトリ

    Returns
    ----------
    None
    """
    # 重みと policy.json を写し, 定義を骨組みと同じ名前に置き直す
    run = project / "outputs" / "cartpole" / "2026-01-04" / "09-00-00"
    policy = run / "policy"
    shutil.copytree(SOURCE, policy)
    old = json.loads((policy / "policy.json").read_text(encoding="utf-8"))
    (policy / old["metadata"]["model"]["spec"]).rename(policy / MODEL_SPEC)
    old["metadata"] = {
        **old["metadata"], "epoch": 2, "model": {"spec": MODEL_SPEC},
        "approver": "現場の責任者", "guards_set": True,
        "guards_text": "行動は 0 か 1",
    }
    (policy / "policy.json").write_text(
        json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
