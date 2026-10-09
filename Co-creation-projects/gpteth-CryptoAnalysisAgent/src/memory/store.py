"""Workspace memory: long-term MEMORY.md plus one markdown diary per day."""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Optional

from ..workspace.manager import WorkspaceManager

_DAILY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CAPTURE_TRIGGERS = ("记住", "记下", "记一下", "帮我记", "别忘了", "以后都", "remember")
_PREFERENCE_HINTS = ("偏好", "偏", "喜欢", "习惯", "风格", "风险", "关注", "prefer")


class MemoryStore:
    def __init__(self, workspace: WorkspaceManager):
        self.workspace = workspace

    @property
    def memory_path(self) -> str:
        return self.workspace.memory_path

    def _daily_path(self, date: str) -> str:
        return os.path.join(self.memory_path, f"{date}.md")

    def load_daily(self, filename: str) -> Optional[str]:
        date = filename[:-3] if filename.endswith(".md") else filename
        if not _DAILY_RE.match(date):
            return None
        path = self._daily_path(date)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()

    def append_daily(self, content: str, category: Optional[str] = None) -> str:
        os.makedirs(self.memory_path, exist_ok=True)
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        path = self._daily_path(date)
        tag = f" [{category.strip().lower()}]" if category and category.strip() else ""
        line = f"- {now.strftime('%H:%M')}{tag} {content.strip()}\n"
        is_new = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as fh:
            if is_new:
                fh.write(f"# {date}\n\n")
            fh.write(line)
        return path

    def append_longterm(self, content: str) -> None:
        current = (self.workspace.load_config("MEMORY") or "# MEMORY.md - 长期记忆\n").rstrip()
        stamp = datetime.now().strftime("%Y-%m-%d")
        self.workspace.save_config("MEMORY", f"{current}\n- {content.strip()}（{stamp}）\n")

    def list_files(self) -> list[dict]:
        files = []
        longterm = self.workspace.get_config_path("MEMORY")
        if os.path.isfile(longterm):
            files.append({"name": "MEMORY.md", "type": "longterm", "size": os.path.getsize(longterm)})
        for name in self._daily_files():
            path = os.path.join(self.memory_path, name)
            files.append({"name": name, "type": "daily", "size": os.path.getsize(path)})
        return files

    def search(self, keyword: str, context: int = 1) -> list[dict]:
        needle = (keyword or "").strip().lower()
        if not needle:
            return []
        sources = [("MEMORY.md", self.workspace.load_config("MEMORY") or "")]
        sources += [(name, self.load_daily(name) or "") for name in self._daily_files()]
        results = []
        for source, text in sources:
            lines = text.splitlines()
            matches = []
            for idx, line in enumerate(lines):
                if needle in line.lower():
                    start = max(0, idx - context)
                    end = min(len(lines), idx + context + 1)
                    matches.append({
                        "start_line": start + 1,
                        "end_line": end,
                        "content": "\n".join(lines[start:end]),
                    })
            if matches:
                results.append({"source": source, "matches": matches})
        return results

    def maybe_capture(self, message: str) -> bool:
        """Save explicit "remember this" requests to today's diary."""
        text = (message or "").strip()
        lowered = text.lower()
        if not text or not any(trigger in lowered for trigger in _CAPTURE_TRIGGERS):
            return False
        category = "preference" if any(hint in lowered for hint in _PREFERENCE_HINTS) else "fact"
        self.append_daily(text[:500], category=category)
        return True

    def recent_daily_text(self, days: int = 2, limit: int = 3000) -> str:
        today = datetime.now().date()
        parts = []
        for offset in range(days):
            date = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
            content = self.load_daily(date)
            if content and content.strip():
                parts.append(content.strip())
        text = "\n\n".join(parts)
        return text if len(text) <= limit else text[-limit:]

    def cleanup(self, days: int = 30) -> list[str]:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        deleted = []
        for name in self._daily_files():
            if name[:-3] < cutoff:
                try:
                    os.remove(os.path.join(self.memory_path, name))
                    deleted.append(name)
                except OSError:
                    continue
        return deleted

    def _daily_files(self) -> list[str]:
        if not os.path.isdir(self.memory_path):
            return []
        names = [n for n in os.listdir(self.memory_path) if n.endswith(".md") and _DAILY_RE.match(n[:-3])]
        return sorted(names, reverse=True)
