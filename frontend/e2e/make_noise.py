"""
画面の通しのテスト用に, 拡散のモデルの実験 (雑音だけの一式の元) を作る

ML-ModelControl の見本の雑音の予測器 (x と時刻 t から加えた雑音) を,
16 の値の正弦波を学習の DDPM の損失で少しだけ学習し, ML-Found の実験の
ように final_model.pt と run.json を書く. 定義は YAML で並べる
"""

import json
import math
import sys
from pathlib import Path

import torch
import yaml
from ml_modelcontrol import build_model, load_example

# 学習の時刻の数 (DDIM の既定の train_steps と同じ)
TRAIN_STEPS = 1000

# 学習の回数
EPOCHS = 200


def wave_batch(count: int) -> torch.Tensor:
    """
    位相のずれた 16 の値の正弦波を作る関数

    Parameters
    ----------
    count: int
        件数

    Returns
    ----------
    batch: torch.Tensor
        (件数, 16) の値
    """
    # 位相を件ごとに変える
    grid = torch.linspace(0, 2 * math.pi, 16)
    phase = torch.rand(count, 1) * 2 * math.pi
    batch = torch.sin(grid + phase)

    return batch


def main(root: Path) -> None:
    """
    定義と実験のディレクトリを作る関数

    Parameters
    ----------
    root: Path
        作るディレクトリ

    Returns
    ----------
    None
    """
    # 定義のファイル
    root.mkdir(parents=True, exist_ok=True)
    spec = load_example("diffusion_denoiser")
    (root / "denoiser.yaml").write_text(
        yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8"
    )

    # 線形の雑音の予定で, 加えた雑音を当てる学習
    torch.manual_seed(0)
    model = build_model(spec, seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    betas = torch.linspace(1e-4, 0.02, TRAIN_STEPS)
    alphas = torch.cumprod(1 - betas, dim=0)
    loss = torch.tensor(0.0)
    for _ in range(EPOCHS):
        x0 = wave_batch(64)
        t = torch.randint(0, TRAIN_STEPS, (64,))
        noise = torch.randn_like(x0)
        a = alphas[t].unsqueeze(1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(xt, t), noise)
        loss.backward()
        optimizer.step()

    # ML-Found の実験のように重みと記録
    run = root / "runs" / "diffusion"
    run.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run / "final_model.pt")
    state = {"format_version": 1, "status": "completed",
             "monitor": "train/loss", "best_value": float(loss.item()),
             "best_epoch": EPOCHS, "final_weights": "best"}
    (run / "run.json").write_text(json.dumps(state), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
