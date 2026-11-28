import hashlib
import os
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles

from ml_provide.api import router
from ml_provide.datasets import router as dataset_router
from ml_provide.demands import router as demand_router
from ml_provide.experiments import router as experiment_router
from ml_provide.policies import router as policy_router
from ml_provide.policy_results import router as policy_result_router
from ml_provide.prices import router as price_router
from ml_provide.problems import router as problem_router
from ml_provide.procedures import configured_modules, load_modules
from ml_provide.reviews import router as review_router
from ml_provide.searches import router as search_router
from ml_provide.settings import get_data_dir, get_problems_dir
from ml_provide.store import Store
from ml_provide.treated import router as treated_router
from ml_provide.units import configured_units, load_unit_modules

# ビルド済みの画面を置く場所 (パッケージの中)
STATIC_DIR = (Path(__file__).parent / "static").resolve()

# ローカルからだけ使う前提で受け付けるホスト名
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]

# 状態を変えるメソッド
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# 起動ごとのトークンを指定する環境変数と, 保存する Cookie の名前
TOKEN_ENV = "ML_PROVIDE_TOKEN"
TOKEN_COOKIE = "ml_provide_token"

# 単独の起動で API を置く頭と, トークンを受け取って Cookie に保存するパス
API_PREFIX = "/api"
AUTH_PATH = "/api/auth"

# 仕様の名前と版
TITLE = "ML-Provide"
VERSION = "0.1.0"

# API だけのアプリに加える router (どれも頭なし)
ROUTERS = [
    router, problem_router, experiment_router, dataset_router,
    policy_router, policy_result_router, review_router, treated_router,
    price_router, demand_router, search_router
]


def route_id(route: APIRoute) -> str:
    """
    OpenAPI の操作の ID を関数名から作る関数

    Parameters
    ----------
    route: APIRoute
        API のルート

    Returns
    ----------
    operation_id: str
        関数名 (画面の型の名前に使われる)
    """
    operation_id = route.name

    return operation_id


async def check_origin(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    他のサイトから状態を変える API を呼べないようにする関数

    Parameters
    ----------
    request: Request
        リクエスト
    call_next: Callable[[Request], Awaitable[Response]]
        次の処理

    Returns
    ----------
    response: Response
        応答, 別の Origin からの書き込みは 403
    """
    # ブラウザが付ける Origin が, 接続先 (Host) と同じホストとポートかの確認
    origin = request.headers.get("origin")
    if request.method in UNSAFE_METHODS and origin is not None:
        if urlsplit(origin).netloc != request.headers.get("host"):
            response: Response = JSONResponse(
                {"detail": "別のサイトからの操作は受け付けません"},
                status_code=403
            )
            return response

    response = await call_next(request)

    return response


async def check_token(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    API の呼び出しに起動ごとのトークンを求める関数

    Parameters
    ----------
    request: Request
        リクエスト (request.app は外側のアプリ. token を持つ)
    call_next: Callable[[Request], Awaitable[Response]]
        次の処理

    Returns
    ----------
    response: Response
        応答, Cookie かヘッダーのトークンが合わない API の呼び出しは 401
    """
    # 手元のファイルを読めるので, 同じ端末の他の利用者からも使えないように
    # 確認
    token: str = request.app.state.token
    path = request.url.path
    # トークンを受け取る GET /api/auth だけを除く (他のメソッドは確かめる)
    opening = path == AUTH_PATH and request.method == "GET"
    if path.startswith(API_PREFIX) and not opening:
        given = [
            request.cookies.get(cookie_name(token), ""),
            request.headers.get("x-provide-token", "")
        ]
        if not any(same_token(value, token) for value in given):
            response: Response = JSONResponse(
                {
                    "detail": "起動したときに表示された URL "
                    f"({AUTH_PATH}?token=...) を開いてください"
                },
                status_code=401
            )
            return response

    response = await call_next(request)

    return response


def cookie_name(token: str) -> str:
    """
    トークンを保存する Cookie の名前を作る関数

    Cookie はポートを区別しないので, 同じ端末で動かす別のインスタンス
    (開発用のサーバーなど) と重ならないように, トークンから名前を作る

    Parameters
    ----------
    token: str
        起動ごとのトークン

    Returns
    ----------
    name: str
        Cookie の名前
    """
    digest = hashlib.sha256(token.encode()).hexdigest()[:12]
    name = f"{TOKEN_COOKIE}_{digest}"

    return name


def same_token(given: str, token: str) -> bool:
    """
    トークンが一致するかを, 時間で推測されないように比べる関数

    Parameters
    ----------
    given: str
        送られたトークン
    token: str
        起動ごとのトークン

    Returns
    ----------
    same: bool
        一致するかどうか (ASCII 以外の文字も比較可能)
    """
    same = secrets.compare_digest(given.encode(), token.encode())

    return same


def get_token(token: Optional[str]) -> str:
    """
    起動ごとのトークンを決める関数

    Parameters
    ----------
    token: Optional[str]
        指定されたトークン, None か空の場合は環境変数かランダムに作成

    Returns
    ----------
    value: str
        トークン
    """
    value = token or os.environ.get(TOKEN_ENV) or secrets.token_urlsafe(24)

    return value


def static_file(static_dir: Path, path: str) -> Optional[Path]:
    """
    要求されたパスに当たる画面の置き場のファイルを探す関数

    Parameters
    ----------
    static_dir: Path
        ビルド済みの画面のディレクトリ
    path: str
        要求されたパス

    Returns
    ----------
    target: Optional[Path]
        画面の置き場の中のファイル, 無い場合は None
        (NUL の字や長すぎる名前のように調べられない場合も None)
    """
    if not path:
        return None

    try:
        target = (static_dir / path).resolve()
        if not (target.is_relative_to(static_dir) and target.is_file()):
            return None
    except (ValueError, OSError):
        return None

    return target


def serve_index(request: Request, path: str) -> FileResponse:
    """
    画面のファイルか, 画面の中の移動先として index.html を返す関数

    Parameters
    ----------
    request: Request
        リクエスト (request.app は画面のディレクトリを持つ)
    path: str
        要求されたパス

    Returns
    ----------
    response: FileResponse
        ファイルか index.html
    """
    # 存在しない API は画面ではなく 404 として返却
    if path == "api" or path.startswith("api/"):
        raise HTTPException(404, f"API の経路 /{path} が見つかりません")

    static_dir: Path = request.app.state.static_dir
    target = static_file(static_dir, path)
    if target is not None:
        response = FileResponse(target)
        return response

    response = FileResponse(static_dir / "index.html")

    return response


def mount_frontend(app: FastAPI, static_dir: Path) -> None:
    """
    ビルド済みの画面を配信する設定を加える関数

    Parameters
    ----------
    app: FastAPI
        アプリ
    static_dir: Path
        ビルド済みの画面のディレクトリ

    Returns
    ----------
    None
    """
    if not (static_dir / "index.html").is_file():
        return

    # JavaScript や CSS などのファイルと, 他のパスの index.html の配信
    # (HEAD も GET と同じく受け付け)
    app.state.static_dir = static_dir
    app.mount(
        "/assets", StaticFiles(directory=static_dir / "assets"), name="assets"
    )
    app.add_api_route(
        "/{path:path}", serve_index, methods=["GET", "HEAD"],
        include_in_schema=False
    )


def authenticate(request: Request, token: str = "") -> Response:
    """
    起動時の URL のトークンを Cookie に保存して画面に移る関数

    Parameters
    ----------
    request: Request
        リクエスト (request.app は外側のアプリ. token を持つ)
    token: str = ""
        URL で渡されたトークン

    Returns
    ----------
    response: Response
        画面への移動, トークンが違う場合は 401
    """
    if not same_token(token, request.app.state.token):
        response: Response = JSONResponse(
            {"detail": "トークンが違います"}, status_code=401
        )
        return response

    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        cookie_name(token), token, httponly=True, samesite="strict"
    )

    return response


def create_api(
    data_dir: Optional[Path] = None, *, problems_dir: Optional[Path] = None,
    procedures: Optional[list[str]] = None,
    units: Optional[list[str]] = None
) -> FastAPI:
    """
    確かめと画面の配信を持たない, API だけのアプリを組み立てる関数

    殻 (ML-Studio) は /api/infer のような頭の下に mount し, トークン,
    Origin, ホスト名の確かめは殻が持つ

    Parameters
    ----------
    data_dir: Optional[Path] = None
        アプリのデータの置き場, None の場合は既定の場所
    problems_dir: Optional[Path] = None
        ML-Pathfinder の問題の設定の置き場, None の場合は既定の場所
        (環境変数 ML_PATHFINDER_HOME か, ホームの下)
    procedures: Optional[list[str]] = None
        自作の繰り返しの手順を登録するモジュール, None の場合は環境変数
        ML_PROVIDE_PROCEDURES (起動の設定で明示したものだけを読む)
    units: Optional[list[str]] = None
        自作の部品を宣言するモジュール, None の場合は環境変数
        ML_PROVIDE_UNITS (起動の設定で明示したものだけを読む)

    Returns
    ----------
    app: FastAPI
        頭なしの経路を持つアプリ (仕様の画面は閉じる)

    Raises
    ----------
    ValueError
        読めないモジュールがある場合
    """
    # 手元のファイルを読めるので, 仕様の画面は出さない
    # (仕様の書き出しは build_openapi)
    app = FastAPI(
        title=TITLE, version=VERSION, generate_unique_id_function=route_id,
        docs_url=None, redoc_url=None, openapi_url=None
    )

    # 一式と推論の結果の置き場と, 問題の設定の置き場をアプリに保存
    # (ルートは request.app.state から)
    home = data_dir or get_data_dir()
    app.state.store = Store(home)
    app.state.problems_dir = problems_dir or get_problems_dir()

    # 起動の設定で明示した自作の部品と手順のモジュールだけの読み込み
    # (手順のモジュールが部品を使えるように部品が先. 登録簿はプロセスで
    # 1 つで, 読み込み済みのモジュールは前の結果なので, 何度呼んでも
    # 重ねて登録しない)
    app.state.unit_modules = load_unit_modules(configured_units(units))
    app.state.procedure_modules = load_modules(
        configured_modules(procedures)
    )

    # API の登録
    for part in ROUTERS:
        app.include_router(part)

    return app


def build_openapi() -> dict[str, Any]:
    """
    単独の起動での経路 (/api の下) の OpenAPI の仕様を作る関数

    外側のアプリの仕様には mount した API が入らないので, API だけの
    アプリの仕様のパスに頭を付ける. 自作の部品と手順のモジュールは読み
    込まない

    Parameters
    ----------
    None

    Returns
    ----------
    schema: dict[str, Any]
        OpenAPI の仕様
    """
    schema = create_api(procedures=[], units=[]).openapi()
    schema["paths"] = {
        API_PREFIX + path: item for path, item in schema["paths"].items()
    }

    return schema


def create_app(
    data_dir: Optional[Path] = None, token: Optional[str] = None,
    allowed_hosts: Optional[list[str]] = None,
    procedures: Optional[list[str]] = None,
    units: Optional[list[str]] = None
) -> FastAPI:
    """
    単独で起動するアプリを組み立てる関数

    トークン, Origin, ホスト名の確かめ, /api/auth, 画面の配信を持つ外側の
    アプリに, API だけのアプリ (create_api) を /api で mount する

    Parameters
    ----------
    data_dir: Optional[Path] = None
        アプリのデータの置き場, None の場合は既定の場所
    token: Optional[str] = None
        API の呼び出しに求めるトークン, None か空の場合は環境変数か
        ランダムに作成
    allowed_hosts: Optional[list[str]] = None
        受け付けるホスト名, None の場合は 127.0.0.1 と localhost
    procedures: Optional[list[str]] = None
        自作の繰り返しの手順を登録するモジュール, None の場合は環境変数
        ML_PROVIDE_PROCEDURES (起動の設定で明示したものだけを読む)
    units: Optional[list[str]] = None
        自作の部品を宣言するモジュール, None の場合は環境変数
        ML_PROVIDE_UNITS (起動の設定で明示したものだけを読む)

    Returns
    ----------
    app: FastAPI
        アプリ

    Raises
    ----------
    ValueError
        読めないモジュールがある場合 (起動を止める)
    """
    # 確かめと画面の配信を持つ外側のアプリ (仕様の画面は閉じる)
    app = FastAPI(
        title=TITLE, version=VERSION, docs_url=None, redoc_url=None,
        openapi_url=None
    )

    # トークンの無い呼び出し, 他のサイトからの書き込み,
    # 他のホスト名での接続 (DNS rebinding) の拒否
    app.state.token = get_token(token)
    if not token and not os.environ.get(TOKEN_ENV):
        # 開発用のサーバーなど, トークンを渡さずに (空で) 作った場合は
        # ランダムに作ったトークンでの開き方を表示
        print(
            "ML-Provide のトークン: "
            f"{AUTH_PATH}?token={app.state.token}",
            flush=True
        )
    app.middleware("http")(check_token)
    app.middleware("http")(check_origin)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=allowed_hosts or ALLOWED_HOSTS
    )

    # トークンの受け取り, API, 画面の登録 (/api/auth は API より先に照合)
    app.add_api_route(
        AUTH_PATH, authenticate, methods=["GET"], include_in_schema=False
    )
    app.mount(
        API_PREFIX,
        create_api(data_dir, procedures=procedures, units=units)
    )
    mount_frontend(app, STATIC_DIR)

    return app
