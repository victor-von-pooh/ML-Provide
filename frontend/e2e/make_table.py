"""
画面の通しのテスト用に, 表を作る拡散のモデルの実験と, その中の表の出力の
雑音だけの一式 (版 7) を作る

ML-Quest の表の生成の骨組みの学習のスクリプトがするように, 骨組みの
プロジェクトの実験 (outputs/<名前>/<日付>/<時刻>) に, 重み (final_model.pt),
run.json, 列の符号化 (table_columns.json) と, 列の符号化を写した推論の一式
(inference) を置く. 一式は ML-Inference の make_bundle で作る. モデルは
ML-ModelControl の見本の雑音の予測器を符号化した行の幅にしたもので, 合成の
行を学習の DDPM の損失で少しだけ学習する
"""

import json
import sys
from pathlib import Path

import torch
import yaml
from ml_inference import make_bundle
from ml_modelcontrol import build_model, load_example

# 学習の時刻の数 (DDIM の既定の train_steps と同じ)
TRAIN_STEPS = 1000

# 学習の回数
EPOCHS = 200

# 列の符号化 (ML-Quest の骨組みの table_columns.json の columns と同じ形).
# 符号化した行の幅は数の列の 2 と種類の列の 3 の和
COLUMNS = [
    {"name": "price", "type": "number", "mean": 300.0, "std": 50.0,
     "integer": False},
    {"name": "count", "type": "number", "mean": 12.0, "std": 4.0,
     "integer": True},
    {"name": "shop", "type": "category", "categories": ["east", "north",
                                                         "west"]},
]
WIDTH = 5


def encoded_batch(count: int) -> torch.Tensor:
    """
    標準化した数の列 2 つと, ワンホットの種類の列の合成の行を作る関数

    Parameters
    ----------
    count: int
        行の数

    Returns
    ----------
    batch: torch.Tensor
        (行の数, 5) の符号化した行
    """
    # 数の列は標準正規, 種類は 3 つから一様に
    numbers = torch.randn(count, 2)
    shops = torch.nn.functional.one_hot(
        torch.randint(0, 3, (count,)), 3
    ).float()
    batch = torch.cat([numbers, shops], dim=1)

    return batch


def main(project: Path) -> None:
    """
    骨組みのプロジェクトの中に, 実験と表の出力の雑音だけの一式を作る関数

    Parameters
    ----------
    project: Path
        骨組みのプロジェクトのディレクトリ

    Returns
    ----------
    None
    """
    # 見本の雑音の予測器を符号化した行の幅に
    spec = load_example("diffusion_denoiser")
    spec["inputs"]["x"]["shape"] = [8, WIDTH]
    spec["nodes"]["x_fc"]["params"]["in_features"] = WIDTH
    spec["nodes"]["head"]["params"]["out_features"] = WIDTH

    # 線形の雑音の予定で, 加えた雑音を当てる学習
    torch.manual_seed(0)
    model = build_model(spec, seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    betas = torch.linspace(1e-4, 0.02, TRAIN_STEPS)
    alphas = torch.cumprod(1 - betas, dim=0)
    loss = torch.tensor(0.0)
    for _ in range(EPOCHS):
        x0 = encoded_batch(64)
        t = torch.randint(0, TRAIN_STEPS, (64,))
        noise = torch.randn_like(x0)
        a = alphas[t].unsqueeze(1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(xt, t), noise)
        loss.backward()
        optimizer.step()

    # ML-Found の実験のように重み, 記録, 定義と列の符号化
    run = project / "outputs" / "rows" / "2026-01-05" / "09-00-00"
    run.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run / "final_model.pt")
    state = {"format_version": 1, "status": "completed",
             "monitor": "train/loss", "best_value": float(loss.item()),
             "best_epoch": EPOCHS, "final_weights": "best"}
    (run / "run.json").write_text(json.dumps(state), encoding="utf-8")
    (run / "model.yaml").write_text(
        yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8"
    )
    (run / "table_columns.json").write_text(
        json.dumps({"columns": COLUMNS}), encoding="utf-8"
    )

    # 骨組みと同じ ddim で, 列の符号化を写した一式
    make_bundle(
        run / "inference", run / "model.yaml", None, run, role="generate",
        form="table",
        inputs={"x": {"source": "state"}, "t": {"source": "time"}},
        output_options={"columns": str(run / "table_columns.json")},
        procedure={"name": "ddim", "steps": 20, "seed": 0,
                   "options": {"eta": 0.0}},
        name="rows"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
