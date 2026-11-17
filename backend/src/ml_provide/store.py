import json
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# 登録した一式の一覧のファイル
BUNDLES_FILE = "bundles.json"


class Store:
    def __init__(self, home: Path):
        # 置き場を保存 (消すときの置き場の中かの確認を, リンクを辿った
        # 場所でそろえる)
        self.home = home.expanduser().resolve()
        self.lock = threading.Lock()

    @property
    def bundles_path(self) -> Path:
        """
        登録した一式の一覧のファイルの場所を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        path: Path
            bundles.json の場所
        """
        path = self.home / BUNDLES_FILE

        return path

    @property
    def results_dir(self) -> Path:
        """
        推論の結果を置くディレクトリを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        path: Path
            results の場所
        """
        path = self.home / "results"

        return path

    def read(self) -> list[dict[str, Any]]:
        """
        登録した一式の一覧を読む関数

        Parameters
        ----------
        None

        Returns
        ----------
        bundles: list[dict[str, Any]]
            登録した一式 (無い, 壊れている場合は空)
        """
        # 壊れたファイルは, 登録を失わないように .bak に退避してから空
        try:
            loaded = json.loads(self.bundles_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, ValueError):
            try:
                self.bundles_path.replace(self.bundles_path.with_suffix(".bak"))
            except OSError:
                pass
            return []
        bundles = [
            item for item in loaded if isinstance(item, dict)
        ] if isinstance(loaded, list) else []

        return bundles

    def write(self, bundles: list[dict[str, Any]]) -> None:
        """
        登録した一式の一覧を書く関数

        Parameters
        ----------
        bundles: list[dict[str, Any]]
            登録した一式

        Returns
        ----------
        None
        """
        # 一時ファイルに書いてから置き換え
        self.home.mkdir(parents=True, exist_ok=True)
        temporary = self.bundles_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(bundles, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.bundles_path)

    def find(self, bundle_id: str) -> Optional[dict[str, Any]]:
        """
        登録した一式を id で探す関数

        Parameters
        ----------
        bundle_id: str
            一式の id

        Returns
        ----------
        bundle: Optional[dict[str, Any]]
            登録 (無ければ None)
        """
        bundle = next(
            (item for item in self.read() if item.get("id") == bundle_id),
            None
        )

        return bundle

    def new_bundle_dir(self) -> tuple[str, Path]:
        """
        置き場で作る一式の id と場所を決める関数

        Parameters
        ----------
        None

        Returns
        ----------
        place: tuple[str, Path]
            id と, まだ無いディレクトリ
        """
        # 重ならない短い id
        bundle_id = uuid.uuid4().hex[:12]
        place = (bundle_id, self.home / "bundles" / bundle_id)

        return place

    def add(
        self, path: Path, name: str, managed: bool,
        bundle_id: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        一式を登録する関数

        Parameters
        ----------
        path: Path
            一式のディレクトリ
        name: str
            名前
        managed: bool
            置き場で作ったか (登録を消すと一式も消す)
        bundle_id: Optional[str] = None
            id (省略すると作る)
        extra: Optional[dict[str, Any]] = None
            登録に足す項目 (見本の新しいデータの場所など)

        Returns
        ----------
        bundle: dict[str, Any]
            登録

        Raises
        ----------
        ValueError
            同じ場所が既に登録されている場合
        """
        # 同じ場所の二重の登録は誤り
        with self.lock:
            bundles = self.read()
            resolved = str(path.resolve())
            if any(item.get("path") == resolved for item in bundles):
                raise ValueError(f"{resolved} は既に登録されています")
            bundle = {
                "id": bundle_id or uuid.uuid4().hex[:12],
                "name": name,
                "path": resolved,
                "managed": managed,
                "added_at": datetime.now().astimezone().isoformat(),
                **(extra or {}),
            }
            bundles.append(bundle)
            self.write(bundles)

        return bundle

    def relocate(self, bundle_id: str, path: Path) -> dict[str, Any]:
        """
        登録した一式の場所を, 置き場に写した一式に移す関数

        Parameters
        ----------
        bundle_id: str
            一式の id
        path: Path
            写した一式のディレクトリ (置き場の中)

        Returns
        ----------
        bundle: dict[str, Any]
            移した後の登録 (置き場で作った一式として扱う)

        Raises
        ----------
        KeyError
            登録が無い場合
        """
        # 場所と, 消すときに一式も消す印 (元の場所は登録に残す)
        with self.lock:
            bundles = self.read()
            bundle = next((item for item in bundles
                           if item.get("id") == bundle_id), None)
            if bundle is None:
                raise KeyError(bundle_id)
            bundle["copied_from"] = bundle["path"]
            bundle["path"] = str(path.resolve())
            bundle["managed"] = True
            self.write(bundles)

        return bundle

    def remove(self, bundle_id: str) -> bool:
        """
        一式の登録を消す関数 (置き場で作った一式は一式も消す)

        Parameters
        ----------
        bundle_id: str
            一式の id

        Returns
        ----------
        removed: bool
            消したか (無ければ False)
        """
        # 置き場の中の一式だけ消す
        with self.lock:
            bundles = self.read()
            target = next(
                (b for b in bundles if b.get("id") == bundle_id), None
            )
            if target is None:
                return False
            self.write([b for b in bundles if b is not target])
        place = Path(str(target["path"]))
        if target.get("managed") and (self.home / "bundles") in place.parents:
            shutil.rmtree(place, ignore_errors=True)
        removed = True

        return removed

    def new_result_dir(self) -> tuple[str, Path]:
        """
        結果の id と場所を決める関数

        Parameters
        ----------
        None

        Returns
        ----------
        place: tuple[str, Path]
            id (時刻の順に並ぶ) と, まだ無いディレクトリ
        """
        # 時刻と短い乱数の id
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
        place = (result_id, self.results_dir / result_id)

        return place

    def result_dir(self, result_id: str) -> Optional[Path]:
        """
        結果の id からディレクトリを返す関数

        Parameters
        ----------
        result_id: str
            結果の id

        Returns
        ----------
        path: Optional[Path]
            結果のディレクトリ (無い, id の形が違う場合は None)
        """
        # id は英数字とハイフンだけ (置き場の外を指せないように)
        if not result_id or not all(
            c.isalnum() or c == "-" for c in result_id
        ):
            return None
        path = self.results_dir / result_id
        if not (path / "result.yaml").is_file():
            return None

        return path

    def result_ids(self) -> list[str]:
        """
        結果の id を新しい順に返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        ids: list[str]
            結果の id
        """
        # result.yaml のある結果だけ
        if not self.results_dir.is_dir():
            return []
        ids = sorted(
            (p.name for p in self.results_dir.iterdir()
             if (p / "result.yaml").is_file()),
            reverse=True
        )

        return ids
