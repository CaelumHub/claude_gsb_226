"""
edit_session.py — in-memory-style *edit sessions* for the waveform editor.

Why this exists
---------------
The waveform editor used to register a brand-new library file after every
single operation (trim / silence / gain / fade / normalize / reverse).  A
mistake could not be undone: the only way back was to start over from the
original file.

An edit session keeps the whole editing process -- a linear list of states
("steps") -- inside one private working area under ``data/edit_sessions/``::

    data/edit_sessions/<sid>/
        meta.json      session descriptor (source file, states, cursor ...)
        0000.wav       state 0: an untouched copy of the source file
        0001.wav       state 1: after the first operation
        ...

Only the current state (``states[cursor]``) is played back / displayed.
Undo/redo/seek simply move the cursor; applying a new operation after an undo
truncates the redo tail and appends the new branch -- exactly the familiar
"linear history with branching on edit" semantics.

The session never touches the original library file (``0000.wav`` is a
*copy*).  Nothing is added to the library until the user explicitly saves
("保存到音频库"), which registers the current state as one derived file.
Discarding a session removes its working directory; stale session dirs are
garbage-collected on startup.

Persistence / concurrency
-------------------------
``meta.json`` is rewritten atomically (temp file + ``os.replace``) under an
exclusive advisory lock (see :mod:`backend.storage`), so concurrent requests
from the browser serialise cleanly.  Each new state is written to a
*different* snapshot file (the index only moves forward until a branch
truncation), and snapshot files are removed under the meta lock, so a served
state file cannot vanish underneath an in-flight request.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional

from backend import audio_io, storage
from backend.storage import FileLock, atomic_write, new_id, now_iso, read_json

# Sessions older than this are swept away when a new session is created.
SESSION_TTL_SECONDS = 24 * 3600.0

OPS = {"trim", "silence", "gain", "fade", "normalize", "reverse"}

# Human-readable labels for the history list.
OP_LABELS = {
    "trim": "裁剪到选区",
    "silence": "静音选区",
    "gain": "增益",
    "fade": "淡入淡出",
    "normalize": "归一化",
    "reverse": "反转",
}


class SessionError(Exception):
    """Raised for unknown sessions / invalid cursor movement."""


# --------------------------------------------------------------------------- #
# Operation rendering (streaming; works on the full-resolution WAV on disk)
# --------------------------------------------------------------------------- #

def _render(src_path: str, dst_path: str, op: str, params: Dict[str, Any]) -> None:
    """Apply one operation and write the resulting WAV to ``dst_path``.

    ``trim`` really trims (only the selected range survives); the other
    operations preserve duration.
    """
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}")

    with audio_io.WavReader(src_path) as r:
        sr = r.sr
        ch = r.channels
        total = r.nframes

        start_f = max(0, min(int(float(params.get("start", 0.0)) * sr), total))
        end_f = max(start_f, min(int(float(params.get("end", total / sr)) * sr), total))
        gain = 10.0 ** (float(params.get("gain_db", 0.0)) / 20.0)
        fade_in = max(0.0, float(params.get("fade_in", 0.0))) * sr
        fade_out = max(0.0, float(params.get("fade_out", 0.0))) * sr

        if op == "reverse":
            bufs: List[List[float]] = [[] for _ in range(ch)]
            for chunk in r.iter_chunks():
                for c, data in enumerate(chunk):
                    bufs[c].extend(data)
            with audio_io.WavWriter(dst_path, sr, ch, 2) as w:
                w.write_chunk([b[::-1] for b in bufs])
            return

        if op == "trim":
            with audio_io.WavWriter(dst_path, sr, ch, 2) as w:
                r._w.setpos(start_f)
                remaining = end_f - start_f
                while remaining > 0:
                    chunk = r.read_chunk(min(1 << 16, remaining))
                    if chunk is None:
                        break
                    n = len(chunk[0])
                    remaining -= n
                    w.write_chunk(chunk)
            return

        # Normalize = gain derived from a full-file peak pre-scan.
        if op == "normalize":
            peak = 0.0
            with audio_io.WavReader(src_path) as rr:
                for chunk in rr.iter_chunks():
                    for data in chunk:
                        for v in data:
                            a = abs(v)
                            if a > peak:
                                peak = a
            target = float(params.get("peak", 0.99))
            gain = (target / peak) if peak > 1e-9 else 1.0
            op = "gain"

        with audio_io.WavWriter(dst_path, sr, ch, 2) as w:
            pos = 0
            while True:
                chunk = r.read_chunk(1 << 16)
                if chunk is None:
                    break
                n = len(chunk[0])
                out: List[List[float]] = []
                for data in chunk:
                    o: List[float] = []
                    for i, v in enumerate(data):
                        gpos = pos + i
                        x = v
                        if op == "gain":
                            x *= gain
                        elif op == "silence":
                            if start_f <= gpos < end_f:
                                x = 0.0
                        elif op == "fade":
                            if fade_in > 0 and gpos < fade_in:
                                x *= gpos / fade_in
                            if fade_out > 0 and gpos >= total - fade_out:
                                x *= max(0.0, (total - gpos) / fade_out)
                        o.append(x)
                    out.append(o)
                w.write_chunk(out)
                pos += n


def _describe(op: str, params: Dict[str, Any]) -> str:
    """One-line human description of an operation (with its parameters)."""
    if op == "trim":
        return f"裁剪 {float(params.get('start', 0)):.2f}s–{float(params.get('end', 0)):.2f}s"
    if op == "silence":
        return f"静音 {float(params.get('start', 0)):.2f}s–{float(params.get('end', 0)):.2f}s"
    if op == "gain":
        return f"增益 {float(params.get('gain_db', 0)):+.1f} dB"
    if op == "fade":
        return f"淡入 {float(params.get('fade_in', 0)):.1f}s / 淡出 {float(params.get('fade_out', 0)):.1f}s"
    if op == "normalize":
        return f"归一化到 {float(params.get('peak', 0.99)):.2f}"
    if op == "reverse":
        return "反转"
    return op


# --------------------------------------------------------------------------- #
# Session manager
# --------------------------------------------------------------------------- #

class EditSessionManager:
    """Owns the ``data/edit_sessions/`` working area and every session in it."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    # -- paths / locking -------------------------------------------------- #

    def _dir(self, sid: str) -> str:
        return os.path.join(self.root, sid)

    def _meta_path(self, sid: str) -> str:
        return os.path.join(self._dir(sid), "meta.json")

    def _snap_path(self, sid: str, index: int) -> str:
        return os.path.join(self._dir(sid), f"{index:04d}.wav")

    def _lock(self, sid: str) -> FileLock:
        # The lock lives inside the session dir; if the session is gone (or the
        # id is bogus), raise SessionError instead of FileNotFoundError.
        if not os.path.isdir(self._dir(sid)):
            raise SessionError("session not found")
        return FileLock(os.path.join(self._dir(sid), "meta.lock"))

    # -- garbage collection ---------------------------------------------- #

    def sweep(self, ttl: float = SESSION_TTL_SECONDS) -> None:
        """Remove session directories not touched within ``ttl`` seconds."""
        now = time.time()
        for name in os.listdir(self.root):
            d = os.path.join(self.root, name)
            mp = os.path.join(d, "meta.json")
            if not (os.path.isdir(d) and os.path.isfile(mp)):
                continue
            try:
                if now - os.path.getmtime(mp) > ttl:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass

    # -- raw meta helpers (must be called under the session lock) -------- #

    def _read(self, sid: str) -> Optional[Dict[str, Any]]:
        return read_json(self._meta_path(sid), None)

    def _write(self, sid: str, meta: Dict[str, Any]) -> None:
        meta["updated_at"] = now_iso()
        atomic_write(self._meta_path(sid), meta)

    @staticmethod
    def _public(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Session descriptor returned to the browser (no absolute paths)."""
        states = meta["states"]
        cursor = meta["cursor"]
        steps = [{
            "index": 0,
            "label": "原始文件",
            "op": None,
            "params": {},
        }]
        for st in states[1:]:
            steps.append({
                "index": st["index"],
                "label": st.get("label") or OP_LABELS.get(st.get("op"), st.get("op", "?")),
                "op": st.get("op"),
                "params": st.get("params", {}),
            })
        cur = states[cursor]
        return {
            "id": meta["id"],
            "source_id": meta["source_id"],
            "source_name": meta.get("source_name"),
            "cursor": cursor,
            "steps": steps,
            "can_undo": cursor > 0,
            "can_redo": cursor < len(states) - 1,
            "dirty": cursor > 0,
            "sr": cur.get("sr"),
            "channels": cur.get("channels"),
            "frames": cur.get("frames"),
            "duration": cur.get("duration"),
            "revision": meta.get("revision", 0),
        }

    # -- session lifecycle ----------------------------------------------- #

    def create(self, source_path: str, source_id: str, source_name: str) -> Dict[str, Any]:
        """Start a session: state 0 is an untouched copy of the source."""
        self.sweep()
        sid = new_id()
        os.makedirs(self._dir(sid), exist_ok=True)
        snap = self._snap_path(sid, 0)
        shutil.copyfile(source_path, snap)
        with audio_io.WavReader(snap) as r:
            sr, ch, frames = r.sr, r.channels, r.nframes
        meta = {
            "id": sid,
            "source_id": source_id,
            "source_name": source_name,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "revision": 0,
            "cursor": 0,
            "states": [{
                "index": 0,
                "wav": "0000.wav",
                "sr": sr,
                "channels": ch,
                "frames": frames,
                "duration": frames / sr if sr else 0.0,
            }],
        }
        with self._lock(sid):
            self._write(sid, meta)
        return self._public(meta)

    def get(self, sid: str) -> Dict[str, Any]:
        meta = self._read(sid)
        if meta is None:
            raise SessionError("session not found")
        return self._public(meta)

    def current_wav(self, sid: str) -> str:
        """Absolute path of the current state's WAV (for streaming/serving)."""
        meta = self._read(sid)
        if meta is None:
            raise SessionError("session not found")
        return os.path.join(self._dir(sid), meta["states"][meta["cursor"]]["wav"])

    # -- editing ---------------------------------------------------------- #

    def apply(self, sid: str, op: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if op not in OPS:
            raise ValueError(f"unknown op {op!r}")
        with self._lock(sid):
            meta = self._read(sid)
            if meta is None:
                raise SessionError("session not found")

            states = meta["states"]
            cursor = meta["cursor"]

            # Editing after an undo: the redo tail becomes a dead branch.
            for st in states[cursor + 1:]:
                p = os.path.join(self._dir(sid), st["wav"])
                if os.path.isfile(p):
                    os.unlink(p)
            del states[cursor + 1:]

            new_index = len(states)
            dst = self._snap_path(sid, new_index)
            src = os.path.join(self._dir(sid), states[cursor]["wav"])
            _render(src, dst, op, params or {})
            with audio_io.WavReader(dst) as r:
                sr, ch, frames = r.sr, r.channels, r.nframes
            states.append({
                "index": new_index,
                "wav": f"{new_index:04d}.wav",
                "op": op,
                "params": params or {},
                "label": _describe(op, params or {}),
                "sr": sr,
                "channels": ch,
                "frames": frames,
                "duration": frames / sr if sr else 0.0,
            })
            meta["cursor"] = new_index
            meta["revision"] = int(meta.get("revision", 0)) + 1
            self._write(sid, meta)
            return self._public(meta)

    def undo(self, sid: str) -> Dict[str, Any]:
        with self._lock(sid):
            meta = self._read(sid)
            if meta is None:
                raise SessionError("session not found")
            if meta["cursor"] <= 0:
                raise SessionError("nothing to undo")
            meta["cursor"] -= 1
            meta["revision"] = int(meta.get("revision", 0)) + 1
            self._write(sid, meta)
            return self._public(meta)

    def redo(self, sid: str) -> Dict[str, Any]:
        with self._lock(sid):
            meta = self._read(sid)
            if meta is None:
                raise SessionError("session not found")
            if meta["cursor"] >= len(meta["states"]) - 1:
                raise SessionError("nothing to redo")
            meta["cursor"] += 1
            meta["revision"] = int(meta.get("revision", 0)) + 1
            self._write(sid, meta)
            return self._public(meta)

    def seek(self, sid: str, index: int) -> Dict[str, Any]:
        """Jump to any existing step ("回到之前的任意一步")."""
        with self._lock(sid):
            meta = self._read(sid)
            if meta is None:
                raise SessionError("session not found")
            if not (0 <= index < len(meta["states"])):
                raise SessionError("step out of range")
            meta["cursor"] = index
            meta["revision"] = int(meta.get("revision", 0)) + 1
            self._write(sid, meta)
            return self._public(meta)

    # -- commit / discard ------------------------------------------------- #

    def commit(self, sid: str, register) -> Dict[str, Any]:
        """Save the current state as one derived library file; close session."""
        with self._lock(sid):
            meta = self._read(sid)
            if meta is None:
                raise SessionError("session not found")
            cur = meta["states"][meta["cursor"]]
            wav_path = os.path.join(self._dir(sid), cur["wav"])
            entry = register(meta["source_id"], meta.get("source_name") or "audio.wav",
                             wav_path, cur.get("op"), cur.get("params", {}),
                             cur.get("index", 0))
            shutil.rmtree(self._dir(sid), ignore_errors=True)
            return entry

    def discard(self, sid: str) -> None:
        """Throw the whole session away -- the original file is untouched."""
        d = self._dir(sid)
        if not os.path.isdir(d):
            raise SessionError("session not found")
        shutil.rmtree(d, ignore_errors=True)
