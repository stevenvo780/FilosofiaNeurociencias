"""Progress tracker — resumable JSON per chunk/phase."""
import json, threading
from pathlib import Path


class Progress:
    def __init__(self, work_dir: Path):
        self.path = work_dir / "progress.json"
        self.lock = threading.Lock()
        self.data = {"chunks": {}, "version": 6}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                pass

    def _flush(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)

    def done(self, cid: int, phase: str) -> bool:
        return self.data["chunks"].get(str(cid), {}).get(phase, False)

    def mark(self, cid: int, phase: str):
        with self.lock:
            self.data["chunks"].setdefault(str(cid), {})[phase] = True
            self._flush()
