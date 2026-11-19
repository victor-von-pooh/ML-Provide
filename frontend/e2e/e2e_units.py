"""
画面の通しのテスト用の, 自作の部品を宣言するモジュール

起動の設定 (環境変数 ML_PROVIDE_UNITS) で読ませ, この部品を使う一式を推論
する
"""

import torch
from torch import nn


class Twice(nn.Module):
    def __init__(self, features: int = 3):
        # 親クラスのコンストラクタを呼び出す
        super(Twice, self).__init__()

        # 入力と同じ幅の全結合の層
        self.fc = nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        全結合の層を通して 2 倍にする関数

        Parameters
        ----------
        x: torch.Tensor
            入力

        Returns
        ----------
        out: torch.Tensor
            出力
        """
        # 全結合と 2 倍
        out = self.fc(x) * 2.0

        return out


# 起動の設定で読んだときに登録する部品
UNITS = [{"cls": Twice, "name": "e2e.Twice"}]
