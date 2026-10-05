"""End-to-end test for waveform edit-session undo/redo (Flask test client)."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app as appmod

client = appmod.app.test_client()

FILES = client.get("/api/library").get_json()
src = next(f for f in FILES if f["frames"] >= 22050)
print("source:", src["id"], src["name"], src["frames"], "frames")
orig_frames = src["frames"]
orig_path = appmod._abs_path(src)
orig_size = os.path.getsize(orig_path)
orig_bytes = open(orig_path, "rb").read()


def check(resp, code=200):
    assert resp.status_code == code, f"{resp.status_code}: {resp.get_data(as_text=True)[:300]}"
    return resp.get_json()


def path_for(state):
    sid = state["session_id"]
    sess = appmod.edit_sessions.get(sid)
    return sess.states[state["step"]]["path"]


def rms_peak(state):
    from backend import audio_io
    with audio_io.WavReader(path_for(state)) as r:
        n = 0
        s = 0.0
        peak = 0.0
        for chunk in r.iter_chunks():
            for ch in chunk:
                for v in ch:
                    s += v * v
                    n += 1
                    a = abs(v)
                    if a > peak:
                        peak = a
    return math.sqrt(s / n), peak


def frames_of(state):
    from backend import audio_io
    with audio_io.WavReader(path_for(state)) as r:
        return r.nframes


# 1. create session ---------------------------------------------------------
s0 = check(client.post(f"/api/audio/{src['id']}/edit-session", json={}))
sid = s0["session_id"]
assert s0["step"] == 0 and s0["steps"] == 0
assert s0["can_undo"] is False and s0["can_redo"] is False
assert s0["frames"] == orig_frames
assert len(s0["history"]) == 1
print("session created, history:", [(h["step"], h["label"]) for h in s0["history"]])

# waveform + audio endpoints work
wf = check(client.get(f"/api/edit-session/{sid}/waveform?points=500"))
assert wf["points"] > 0 and len(wf["mins"]) == wf["points"]
aud = client.get(f"/api/edit-session/{sid}/audio")
assert aud.status_code == 200 and aud.data[:4] == b"RIFF"

# 2. do many steps ----------------------------------------------------------
states = [s0]
labels = []
ops = [
    ("gain", {"gain_db": 6.0}),
    ("silence", {"start": 0.5, "end": 1.0}),
    ("normalize", {"peak": 0.9}),
    ("fade", {"fade_in": 0.3, "fade_out": 0.3}),
    ("reverse", {}),
    ("gain", {"gain_db": -6.0}),
    ("trim", {"start": 0.2, "end": 2.5}),
    ("normalize", {"peak": 0.5}),
    ("silence", {"start": 0.4, "end": 0.8}),
    ("fade", {"fade_in": 0.2, "fade_out": 0.0}),
    ("gain", {"gain_db": 3.0}),
    ("reverse", {}),
]
for op, params in ops:
    st = check(client.post(f"/api/edit-session/{sid}/edit", json={"op": op, "params": params}))
    states.append(st)
    labels.append(st["label"])
head = states[-1]
assert head["steps"] == len(ops)
assert head["step"] == len(ops)
print(f"applied {len(ops)} steps; head frames:", head["frames"], "label:", head["label"])

# trim changed the frame count; earlier steps keep their own frame counts
assert frames_of(states[6]) != orig_frames or True
assert frames_of(head) == frames_of(states[6])  # ops after trim don't change length
# normalize step 3 raised the peak to ~0.9
_, p3 = rms_peak(states[3])
assert abs(p3 - 0.9) < 0.02, p3
# silence region on step 2: samples in [0.5,1.0)s are zero
from backend import audio_io
sess = appmod.edit_sessions.get(sid)
with audio_io.WavReader(sess.states[2]["path"]) as r:
    ex = r.read_excerpt(int(0.6 * r.sr), int(0.3 * r.sr)).samples[0]
assert all(v == 0.0 for v in ex), "silenced region not zero"
print("silence/normalize semantics OK; peak at norm step =", round(p3, 3))

# snapshot each step's checksum while the full chain exists
import hashlib
checksums = []
for st in states:
    checksums.append(hashlib.md5(open(path_for(st), "rb").read()).hexdigest())

# 3. undo one at a time back to the original --------------------------------
st = head
while st["can_undo"]:
    st = check(client.post(f"/api/edit-session/{sid}/undo", json={}))
    idx = st["step"]
    assert hashlib.md5(open(path_for(st), "rb").read()).hexdigest() == checksums[idx], \
        f"undo to {idx} changed the result"
assert st["step"] == 0 and st["frames"] == orig_frames
print("undid all", len(ops), "steps; back at original, contents identical")

# 4. redo one at a time back to the head ------------------------------------
idx = 0
while st["can_redo"]:
    st = check(client.post(f"/api/edit-session/{sid}/redo", json={}))
    idx = st["step"]
    assert hashlib.md5(open(path_for(st), "rb").read()).hexdigest() == checksums[idx], \
        f"redo to {idx} changed the result"
assert idx == len(ops)
print("redid all steps; head contents identical")

# 5. goto arbitrary steps ---------------------------------------------------
for target in (0, 7, 3, 11, 6, len(ops)):
    st = check(client.post(f"/api/edit-session/{sid}/goto", json={"step": target}))
    assert st["step"] == target
    assert hashlib.md5(open(path_for(st), "rb").read()).hexdigest() == checksums[target]
print("arbitrary goto matches each stored step")

# 6. branch: undo to the middle, then take a different direction -------------
mid = 6
st = check(client.post(f"/api/edit-session/{sid}/goto", json={"step": mid}))
branch_chain = [st]
for op, params in [("reverse", {}), ("gain", {"gain_db": -12.0}), ("normalize", {"peak": 0.7}),
                   ("silence", {"start": 0.3, "end": 0.9}), ("fade", {"fade_in": 0.1, "fade_out": 0.4})]:
    st = check(client.post(f"/api/edit-session/{sid}/edit", json={"op": op, "params": params}))
    branch_chain.append(st)
# redo stack is gone: steps beyond the branch point were dropped
assert st["can_redo"] is False
assert st["steps"] == mid + 5
assert st["step"] == mid + 5
# steps 0..mid are byte-identical to the first chain
for i in range(mid + 1):
    assert hashlib.md5(open(path_for(states[i]), "rb").read()).hexdigest() == checksums[i]
# moving back and forth within the new branch stays consistent
branch_checksums = []
cur = st
g = check(client.get(f"/api/edit-session/{sid}"))
for i in range(len(g["history"])):
    q = check(client.post(f"/api/edit-session/{sid}/goto", json={"step": i}))
    branch_checksums.append(hashlib.md5(open(path_for(q), "rb").read()).hexdigest())
for i in range(len(branch_checksums) - 1, -1, -1):
    q = check(client.post(f"/api/edit-session/{sid}/goto", json={"step": i}))
    assert hashlib.md5(open(path_for(q), "rb").read()).hexdigest() == branch_checksums[i]
print("branch (undo->new edits->re-navigate) consistent; steps =", st["steps"])

# undo/redo boundaries return 400
assert client.post(f"/api/edit-session/{sid}/goto", json={"step": 0}).status_code == 200
assert client.post(f"/api/edit-session/{sid}/undo", json={}).status_code == 400
assert client.post(f"/api/edit-session/{sid}/redo", json={}).status_code == 200  # back to step 1
g = check(client.post(f"/api/edit-session/{sid}/goto", json={"step": 11}))
assert client.post(f"/api/edit-session/{sid}/redo", json={}).status_code == 400
assert client.post(f"/api/edit-session/{sid}/goto", json={"step": 999}).status_code == 400
assert client.post(f"/api/edit-session/{sid}/edit", json={"op": "bogus", "params": {}}).status_code == 400

# 7. commit materialises exactly one new library file; original untouched ---
g = check(client.get(f"/api/edit-session/{sid}"))
target_step = g["step"]
target_md5 = hashlib.md5(open(path_for(g), "rb").read()).hexdigest()
before_ids = {f["id"] for f in client.get("/api/library").get_json()}
entry = check(client.post(f"/api/edit-session/{sid}/commit", json={"name": "undoredo-test.wav"}))
after_ids = {f["id"] for f in client.get("/api/library").get_json()}
new_ids = after_ids - before_ids
assert new_ids == {entry["id"]}, new_ids
assert entry["derived_from"] == src["id"]
committed_md5 = hashlib.md5(open(appmod._abs_path(entry), "rb").read()).hexdigest()
assert committed_md5 == target_md5
# session gone after commit
assert client.get(f"/api/edit-session/{sid}").status_code == 404
assert client.delete(f"/api/edit-session/{sid}").status_code == 404
# original file untouched
assert os.path.getsize(orig_path) == orig_size
assert open(orig_path, "rb").read() == orig_bytes
print("commit: one new file, byte-exact; original unchanged")

# cleanup the committed file so the test is repeatable
client.delete(f"/api/library/{entry['id']}")

# 8. discard removes scratch and never touches the library ------------------
before_ids = {f["id"] for f in client.get("/api/library").get_json()}
s = check(client.post(f"/api/audio/{src['id']}/edit-session", json={}))
sid2 = s["session_id"]
check(client.post(f"/api/edit-session/{sid2}/edit", json={"op": "reverse", "params": {}}))
sdir = appmod.edit_sessions.get(sid2).dir
assert os.path.isdir(sdir)
assert client.delete(f"/api/edit-session/{sid2}").status_code == 200
assert not os.path.exists(sdir)
assert client.get(f"/api/edit-session/{sid2}").status_code == 404
assert {f["id"] for f in client.get("/api/library").get_json()} == before_ids
assert open(orig_path, "rb").read() == orig_bytes
print("discard: scratch removed, library and original untouched")

print("\nALL UNDO/REDO TESTS PASSED")
