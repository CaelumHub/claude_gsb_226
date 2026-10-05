"""
edit_session.py — In-memory edit sessions with undo / redo for the waveform
editor.

Why this exists
---------------
The plain edit endpoint (``/api/audio/<id>/edit``) renders every operation to a
brand-new library file, so a mistake cannot be taken back.  An *edit session*
instead keeps the chain of intermediate results inside one editing process:

    * each operation (trim / silence / gain / fade / normalize / reverse) is
      rendered to a full-snapshot WAV inside a private session directory;
    * undo / redo / "jump back to any step" are therefore pure pointer moves —
      every revision corresponds to exactly one on-disk result, no replaying
      or re-derivation, which keeps each step's result identical even after
      undoing to the middle of the chain and taking a different direction;
    * the session (and every snapshot) is ephemeral.  Only an explicit *commit*
      registers one WAV in the library, so the original file is never touched
      or overwritten; abandoning the session removes all scratch files.

Sessions intentionally live in process memory (not in library.json): the undo
chain is meaningful only within a single editing process, exactly like the
undo stack of a desktop editor.  Idle sessions are reaped automatically.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from . import audio_io
from .storage import new_id

# Cap on retained revisions: with a full WAV per revision we bound the scratch
# footprint.  Older revisions age out of the *undo* chain (the current result
# is always retained).
MAX_STATES = 100

# Sessions untouched for this long (seconds) are destroyed on the next access.
IDLE_TTL = 6 * 60 * 60


class EditSession:
    """One editing process: a revision chain plus a pointer to the current one."""

    def __init__(self, sid: str, source_id: str, source_name: str,
                 sessions_dir: str):
        self.sid = sid
        self.source_id = source_id
        self.source_name = source_name
        self.dir = os.path.join(sessions_dir, sid)
        os.makedirs(self.dir, exist_ok=True)
        # ``states`` is the linear history; ``pos`` indexes the current
        # revision.  Revisions beyond ``pos`` are the redo stack.  ``rev`` is a
        # monotonic counter used for snapshot filenames, so branching never
        # reuses (or overwrites) a file that still belongs to the chain.
        self.states: List[Dict[str, Any]] = []
        self.pos = -1
        self.rev = 0
        self.total_steps = 0  # operations ever applied on the current branch
        self.lock = threading.RLock()
        self.touched = time.time()

    # -- internal helpers ------------------------------------------------- #

    def _mark(self) -> None:
        self.touched = time.time()

    def _snapshot_path(self, rev: int) -> str:
        return os.path.join(self.dir, f"r_{rev:06d}.wav")

    def _probe(self, path: str) -> Dict[str, Any]:
        with audio_io.WavReader(path) as r:
            return {"sr": r.sr, "channels": r.channels, "frames": r.nframes,
                    "duration": r.duration, "size_bytes": os.path.getsize(path)}

    def _describe(self) -> Dict[str, Any]:
        history = []
        for i, st in enumerate(self.states):
            history.append({
                "step": i,
                "rev": st["rev"],
                "label": st["label"],
                "op": st["op"],
                "params": st.get("params", {}),
                "current": i == self.pos,
            })
        cur = self.states[self.pos]
        aged = len(self.states) < self.total_steps + 1
        return {
            "session_id": self.sid,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "step": self.pos,
            "steps": self.total_steps,
            "history_aged": aged,
            "can_undo": self.pos > 0,
            "can_redo": self.pos < len(self.states) - 1,
            "rev": cur["rev"],
            "label": cur["label"],
            "history": history,
            **{k: cur[k] for k in ("sr", "channels", "frames", "duration")},
        }

    # -- lifecycle -------------------------------------------------------- #

    def seed(self, src_path: str) -> Dict[str, Any]:
        """Snapshot the untouched source as revision 0."""
        with self.lock:
            self.rev = 0
            dst = self._snapshot_path(0)
            shutil.copyfile(src_path, dst)
            meta = self._probe(dst)
            self.states = [{
                "rev": 0, "label": "原始文件", "op": "original",
                "params": {}, "path": dst, **meta,
            }]
            self.pos = 0
            self._mark()
            return self._describe()

    # -- operations ------------------------------------------------------- #

    def apply(self, op: str, params: Dict[str, Any], label: str,
              apply_edit: Callable[[str, str, str, Dict[str, Any]], None]) -> Dict[str, Any]:
        """Apply ``op`` to the current revision, starting a new head.

        Any redo stack is discarded (classic undo semantics: editing after an
        undo opens a new branch) and its scratch files are removed.
        """
        with self.lock:
            branched = self.pos < len(self.states) - 1
            rev = self.rev + 1
            dst = self._snapshot_path(rev)
            try:
                apply_edit(self.states[self.pos]["path"], dst, op, params)
                meta = self._probe(dst)
            except Exception:
                # Don't leave a half-written, untracked snapshot behind.
                try:
                    os.unlink(dst)
                except OSError:
                    pass
                raise
            self.rev = rev

            # Drop the redo branch.
            for st in self.states[self.pos + 1:]:
                try:
                    os.unlink(st["path"])
                except OSError:
                    pass
            del self.states[self.pos + 1:]

            self.states.append({
                "rev": rev, "label": label, "op": op,
                "params": params or {}, "path": dst, **meta,
            })
            self.pos = len(self.states) - 1
            if branched:
                # Opening a new branch rewrites the tail; the operation count
                # is the surviving prefix length (aging may have dropped more).
                self.total_steps = self.pos
            else:
                self.total_steps += 1

            # Bound the scratch footprint: age out old revisions, but always
            # keep revision 0 (the original anchor) and the current head.
            while len(self.states) > MAX_STATES:
                victim = self.states[1]  # index 0 is the original
                try:
                    os.unlink(victim["path"])
                except OSError:
                    pass
                self.states.pop(1)
                self.pos -= 1

            self._mark()
            return self._describe()

    def goto_step(self, step: int) -> Optional[Dict[str, Any]]:
        """Move the current-revision pointer to any step of the chain."""
        with self.lock:
            if not (0 <= step < len(self.states)):
                return None
            self.pos = step
            self._mark()
            return self._describe()

    def undo(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            if self.pos <= 0:
                return None
            self.pos -= 1
            self._mark()
            return self._describe()

    def redo(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            if self.pos >= len(self.states) - 1:
                return None
            self.pos += 1
            self._mark()
            return self._describe()

    def current_path(self) -> str:
        with self.lock:
            return self.states[self.pos]["path"]

    def destroy(self) -> None:
        with self.lock:
            shutil.rmtree(self.dir, ignore_errors=True)


class EditSessionManager:
    """Process-wide registry of live edit sessions."""

    def __init__(self, root: str):
        self.sessions_dir = os.path.join(root, "edit_sessions")
        os.makedirs(self.sessions_dir, exist_ok=True)
        self._sessions: Dict[str, EditSession] = {}
        self._guard = threading.Lock()
        # Directories left behind by a previous process are stale scratch.
        for name in os.listdir(self.sessions_dir):
            p = os.path.join(self.sessions_dir, name)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)

    def _reap_idle(self) -> None:
        now = time.time()
        dead = [sid for sid, s in self._sessions.items()
                if now - s.touched > IDLE_TTL]
        for sid in dead:
            s = self._sessions.pop(sid, None)
            if s:
                s.destroy()

    def create(self, source_path: str, source_id: str,
               source_name: str) -> Dict[str, Any]:
        with self._guard:
            self._reap_idle()
            sid = new_id()
            session = EditSession(sid, source_id, source_name, self.sessions_dir)
            self._sessions[sid] = session
        # Sessions are always stored as PCM WAV; decode compressed sources
        # (e.g. an mp3 produced by the export page) up front.
        try:
            if not source_path.lower().endswith(".wav"):
                decoded = audio_io.decode_with_ffmpeg(source_path)
                try:
                    desc = session.seed(decoded)
                finally:
                    try:
                        os.unlink(decoded)
                    except OSError:
                        pass
            else:
                desc = session.seed(source_path)
        except Exception:
            with self._guard:
                self._sessions.pop(sid, None)
            session.destroy()
            raise
        return desc

    def get(self, sid: str) -> Optional[EditSession]:
        with self._guard:
            self._reap_idle()
            return self._sessions.get(sid)

    def drop(self, sid: str) -> bool:
        with self._guard:
            session = self._sessions.pop(sid, None)
        if session is None:
            return False
        session.destroy()
        return True
