"""発注案の保存場所（TypeScript版の agent_proposals テーブルの代わりに、JSONファイルを使う）

エージェントは「承認待ち」の発注案を作ることしかできない。
承認できるのは、人がコマンドで approve を実行したときだけ。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path


class ProposalStore:
    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: list[dict]) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_pending(self, items: list[dict], note: str) -> str:
        """承認待ちで保存する（エージェントが使えるのはこれだけ）"""
        data = self._load()
        pid = uuid.uuid4().hex[:8]
        data.append({"id": pid, "created_at": datetime.now().isoformat(timespec="seconds"), "status": "pending", "note": note, "items": items})
        self._save(data)
        return pid

    def list(self) -> list[dict]:
        return self._load()

    def decide(self, pid: str, status: str) -> bool:
        """人が承認／却下する（承認待ちのものだけ変えられる）"""
        if status not in ("approved", "rejected"):
            raise ValueError("status は approved か rejected")
        data = self._load()
        for p in data:
            if p["id"] == pid and p["status"] == "pending":
                p["status"] = status
                p["decided_at"] = datetime.now().isoformat(timespec="seconds")
                self._save(data)
                return True
        return False
