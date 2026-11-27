import argparse
import json
from pathlib import Path

import uvicorn

from ml_provide.app import build_openapi, create_app, get_token
from ml_provide.procedures import module_names


def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を読み込む関数

    Parameters
    ----------
    None

    Returns
    ----------
    args: argparse.Namespace
        コマンドライン引数
    """
    parser = argparse.ArgumentParser(prog="ml-provide")
    parser.add_argument("--host", default="127.0.0.1", help="待ち受けるホスト")
    parser.add_argument("--port", type=int, default=8769, help="ポート番号")
    parser.add_argument(
        "--procedures", default=None,
        help="自作の繰り返しの手順を登録するモジュール (, で区切る. 省くと"
        "環境変数 ML_PROVIDE_PROCEDURES). 明示したものだけを読む"
    )
    parser.add_argument(
        "--units", default=None,
        help="自作の部品を宣言するモジュール (, で区切る. 省くと環境変数 "
        "ML_PROVIDE_UNITS). 明示したものだけを読む"
    )
    parser.add_argument(
        "--openapi", type=Path, default=None,
        help="起動せずに OpenAPI の仕様をこのファイルに書き出す"
    )
    args = parser.parse_args()

    return args


def main() -> None:
    """
    アプリを起動する関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    args = parse_args()

    # 画面の型を生成するための仕様の書き出し
    if args.openapi is not None:
        # 仕様の書き出しでは自作の部品と手順のモジュールを読み込まない
        text = json.dumps(build_openapi(), ensure_ascii=False, indent=2)
        args.openapi.write_text(text + "\n", encoding="utf-8")
        return

    # 外から接続できるホストでは, 他の端末から使える可能性を警告
    if args.host not in ("127.0.0.1", "localhost"):
        print(
            f"警告: {args.host} で待ち受けます. トークンを知る人は, "
            "手元のファイルを読めます"
        )

    # 起動ごとのトークン付きの URL の表示
    # 全てのアドレスで待ち受ける場合も, 手元のブラウザで開ける URL を表示
    # 外から接続するホストの名前も受け付ける (全てのアドレスなら全ての名前)
    hosts = ["127.0.0.1", "localhost"]
    if args.host in ("0.0.0.0", "::"):
        hosts = ["*"]
    elif args.host not in hosts:
        hosts.append(args.host)
    modules = None
    if args.procedures is not None:
        modules = module_names(args.procedures)
    units = None
    if args.units is not None:
        units = module_names(args.units)
    # 読めない自作の部品と手順のモジュールでの, 理由の表示と停止
    try:
        app = create_app(
            token=get_token(None), allowed_hosts=hosts, procedures=modules,
            units=units
        )
    except ValueError as error:
        print(f"起動できません: {error}", flush=True)
        raise SystemExit(1) from error
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(
        "ML-Provide を開く: "
        f"http://{shown}:{args.port}/api/auth?token={app.state.token}",
        flush=True
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
