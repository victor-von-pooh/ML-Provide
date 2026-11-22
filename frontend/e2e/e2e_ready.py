"""
画面の通しのテスト用の, 既製のモデルの作り物の出どころを足すモジュール

起動の設定 (環境変数 ML_PROVIDE_UNITS) で読ませ, 部品を登録し, ML-Inference
の既製のモデルの出どころ (READY_LIBRARIES) に e2e を足す. 出どころの重みは
部品が自分で作るので, ネットワークを使わない
"""

from typing import Any

import torch
from ml_inference import READY_LIBRARIES
from torch import nn

# 作り物の出どころの種類の名前の表
CLASSES = ["赤いもの", "緑のもの", "青いもの"]


class ReadyColors(nn.Module):
    def __init__(self, name: str = "colors", weights: str = "V1"):
        # 親クラスのコンストラクタを呼び出す
        super(ReadyColors, self).__init__()

        # 出どころの重み (チャンネルの平均をそのまま種類の logit に)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3, 3)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(3) * 4.0)
            self.fc.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        チャンネルの平均から種類の logit を求める関数

        Parameters
        ----------
        x: torch.Tensor
            (件数, 3, 高さ, 幅) の画像

        Returns
        ----------
        out: torch.Tensor
            (件数, 3) の logit
        """
        # 平均して全結合
        out = self.fc(self.flatten(self.pool(x)))

        return out


def colors_weights(model: str, weights: str) -> dict[str, Any]:
    """
    作り物の出どころの重みの説明を返す関数

    Parameters
    ----------
    model: str
        モデルの名前
    weights: str
        重みの名前

    Returns
    ----------
    found: dict[str, Any]
        READY_LIBRARIES の関数の結果の形
    """
    # 名前は 1 つだけ
    if model != "colors" or weights != "V1":
        raise ValueError(f"{model} の重みに {weights} がありません (重み: V1)")
    found = {
        "unit": "e2e.ReadyColors",
        "params": {"name": model, "weights": weights},
        "classes": list(CLASSES), "input": "image",
        "preprocess": {
            "resize": 16, "crop": 16, "interpolation": "bilinear",
            "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
        },
        "url": "https://example.com/e2e-colors-0123abcd.pth",
        "file": "e2e-colors-0123abcd.pth",
    }

    return found


# 起動の設定で読んだときに登録する部品と出どころ
UNITS = [{"cls": ReadyColors, "name": "e2e.ReadyColors"}]
READY_LIBRARIES["e2e"] = colors_weights
