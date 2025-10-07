"""
io_pipeline.py — Overmind Event Bus PRO + HMAC
- Kanały: psyche, auto, memo, prog, images, travel, crypto, files, misc
- Metryki, rate-limit, idempotency, DLQ, rotacja, eksport/import
- HMAC-SHA256 podpis zdarzeń (AUTH_TOKEN). Wymaganie: REQUIRE_HMAC=1
- verify_hmac()/emit_secure() dla API
"""

from __future__ import annotations

import gzip as _gzip
import hashlib
import hmac
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/mrd69")
PIPE_DIR = ROOT / "out" / "pipeline"
PIPE_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = PIPE_DIR / "events.jsonl"
DLQ_FILE = PIPE_DIR / "dead_letter.jsonl"
ROTATE_MB = int(os.getenv("BUS_ROTATE_MB", "48"))
MAX_RING = int(os.getenv("BUS_RING", "10000"))
MAX_PAYLOAD = int(os.getenv("BUS_MAX_PAYLOAD", "200_000"))
MAX_DEPTH = int(os.getenv("BUS_MAX_DEPTH", "6"))
RATE_PER_MIN = int(os.getenv("BUS_RATE_PER_MIN", "300"))
REQUIRE_HMAC = os.getenv("REQUIRE_HMAC", "0").lower() in ("1", "true", "yes")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

_HAS_MEM = _HAS_PSY = _HAS_AN = False
memory = psychika = autonauka = None
try:
    from . import memory as _m

    memory = _m
    _HAS_MEM = True
except Exception:
    pass
try:
    from . import psychika as _p

    psychika = _p
    _HAS_PSY = True
except Exception:
    pass
try:
    from . import autonauka as _a

    autonauka = _a
    _HAS_AN = True
except Exception:
    pass


def _now_ms() -> int:
    return int(time.time() * 1000)


def _size(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False))
    except Exception:
        return 0


def _deep(obj: Any, lvl: int = 0) -> int:
    if lvl > MAX_DEPTH:
        return lvl
    if isinstance(obj, dict):
        return max([lvl] + [_deep(v, lvl + 1) for v in obj.values()]) if obj else lvl
    if isinstance(obj, list):
        return max([lvl] + [_deep(v, lvl + 1) for v in obj]) if obj else lvl
    return lvl


@dataclass
class Event:
    ts: int
    channel: str
    type: str
    user: str
    data: dict[str, Any]
    id: str
    idem: str | None = None
    attempts: int = 0
    prev_ts: int | None = None


_ALLOWED_CH = {
    "psyche",
    "auto",
    "memo",
    "prog",
    "images",
    "travel",
    "crypto",
    "files",
    "misc",
}


def _validate(channel: str, type_: str, data: dict[str, Any]) -> tuple[bool, str]:
    if channel not in _ALLOWED_CH:
        return False, "invalid_channel"
    if not isinstance(data, dict):
        return False, "data_not_dict"
    if _size(data) > MAX_PAYLOAD:
        return False, "payload_too_big"
    if _deep(data) > MAX_DEPTH:
        return False, "payload_too_deep"
    if not re.match(r"^[a-z0-9_\-\.]{1,64}$", type_):
        return False, "invalid_type"
    return True, ""


class Metrics:
    def __init__(self):
        self.lock = threading.RLock()
        self.by_chan: dict[str, dict[str, int]] = {}
        self.last_ts: dict[str, int] = {}
        self.rate: dict[tuple[str, str], deque] = {}
        self.dropped: int = 0
        self.dlq: int = 0

    def incr(self, chan: str, key: str, v: int = 1):
        with self.lock:
            m = self.by_chan.setdefault(chan, {})
            m[key] = m.get(key, 0) + v
            self.last_ts[chan] = _now_ms()

    def allow_rate(self, user: str, chan: str) -> bool:
        from collections import deque

        now = _now_ms()
        with self.lock:
            dq = self.rate.setdefault((user, chan), deque())
            while dq and now - dq[0] > 60_000:
                dq.popleft()
            if len(dq) >= RATE_PER_MIN:
                return False
            dq.append(now)
            return True

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "by_channel": self.by_chan,
                "last_ts": self.last_ts,
                "dropped": self.dropped,
                "dlq": self.dlq,
            }


METRICS = Metrics()


class EventBus:
    def __init__(self, log_path: Path):

        self.log_path = log_path
        self._lock = threading.RLock()
        self._subs: dict[str, list[Callable[[Event], None]]] = {}
        self._ring: deque[Event] = deque(maxlen=MAX_RING)
        self._idem: dict[str, str] = {}
        self._before: list[Callable[[Event], None]] = []
        self._after: list[Callable[[Event], None]] = []

    def on_before(self, cb: Callable[[Event], None]) -> None:
        with self._lock:
            self._before.append(cb)

    def on_after(self, cb: Callable[[Event], None]) -> None:
        with self._lock:
            self._after.append(cb)

    def _rotate_if_needed(self):
        try:
            if (
                self.log_path.exists()
                and self.log_path.stat().st_size > ROTATE_MB * 1024 * 1024
            ):
                ts = int(time.time())
                dst = self.log_path.with_name(f"events_{ts}.jsonl.gz")
                with open(self.log_path, "rb") as src, _gzip.open(
                    dst, "wb", compresslevel=5
                ) as gz:
                    shutil.copyfileobj(src, gz)
                self.log_path.unlink(missing_ok=True)
        except Exception:
            pass

    def publish(
        self,
        channel: str,
        type_: str,
        data: dict[str, Any] | None = None,
        user: str = "global",
        idem_key: str | None = None,
    ) -> Event:
        ok, err = _validate(channel or "misc", type_ or "event", data or {})
        if not ok:
            METRICS.dropped += 1
            return Event(
                ts=_now_ms(),
                channel="misc",
                type="drop",
                user=user,
                data={"error": err},
                id=str(uuid.uuid4()),
            )
        if not METRICS.allow_rate(user or "global", channel):
            METRICS.dropped += 1
            return Event(
                ts=_now_ms(),
                channel="misc",
                type="rate_limited",
                user=user,
                data={},
                id=str(uuid.uuid4()),
            )
        if idem_key:
            with self._lock:
                if idem_key in self._idem:  # duplikat
                    return Event(
                        ts=_now_ms(),
                        channel=channel,
                        type=type_,
                        user=user,
                        data={"dup_of": self._idem[idem_key]},
                        id=str(uuid.uuid4()),
                        idem=idem_key,
                    )
        ev = Event(
            ts=_now_ms(),
            channel=channel,
            type=type_,
            user=user,
            data=(data or {}),
            id=str(uuid.uuid4()),
            idem=idem_key or None,
        )
        with self._lock:
            self._rotate_if_needed()
            for cb in self._before:
                try:
                    cb(ev)
                except Exception:
                    pass
            self._ring.append(ev)
            line = {
                "ts": ev.ts,
                "id": ev.id,
                "channel": ev.channel,
                "type": ev.type,
                "user": ev.user,
                "data": ev.data,
                "idem": ev.idem,
            }
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            for cb in self._subs.get(ev.channel, []) + self._subs.get("*", []):
                try:
                    cb(ev)
                except Exception:
                    pass
            for cb in self._after:
                try:
                    cb(ev)
                except Exception:
                    pass
            if idem_key:
                self._idem[idem_key] = ev.id
        METRICS.incr(channel, "published", 1)
        return ev

    def subscribe(self, channel: str, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._subs.setdefault(channel, []).append(callback)

    def recent(
        self,
        channel: str | None = None,
        limit: int = 100,
        since_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._ring)
        if channel:
            items = [e for e in items if e.channel == channel]
        if since_ms:
            items = [e for e in items if e.ts >= since_ms]
        items = items[-int(max(1, min(2000, limit))) :]
        return [
            {
                "ts": e.ts,
                "id": e.id,
                "channel": e.channel,
                "type": e.type,
                "user": e.user,
                "data": e.data,
                "idem": e.idem,
            }
            for e in items
        ]


BUS = EventBus(LOG_FILE)


def _to_dlq(ev: Event, reason: str) -> None:
    try:
        with open(DLQ_FILE, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": _now_ms(),
                        "reason": reason,
                        "event": {
                            "ts": ev.ts,
                            "id": ev.id,
                            "channel": ev.channel,
                            "type": ev.type,
                            "user": ev.user,
                            "data": ev.data,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        METRICS.dlq += 1
    except Exception:
        pass


def _route_psyche(ev: Event) -> None:
    if not _HAS_PSY or not psychika:
        return
    try:
        if ev.type == "event":
            handler = getattr(psychika, "handle_event", None)
            if handler and callable(handler):
                event_name = ev.data.get("event")
                if not event_name:
                    event_name = "generic"
                res = handler(event_name, ev.data, ev.user)
                if res:
                    BUS.publish(
                        "memo",
                        "note",
                        {
                            "title": f"[PSY]{ev.data.get('event','generic')}",
                            "sources": [{"text": str(res)}],
                        },
                        ev.user,
                    )
        elif ev.type in ("autopilot", "tick", "cycle"):
            try:
                autopilot = getattr(psychika, "autopilot_cycle", None)
                if autopilot is not None and callable(autopilot):
                    res = autopilot(ev.user)
                    if res:
                        BUS.publish(
                            "memo",
                            "note",
                            {
                                "title": "[PSY]autopilot",
                                "sources": [{"text": str(res)}],
                            },
                            ev.user,
                        )
            except Exception as e:
                _to_dlq(ev, f"psyche_autopilot_error:{e}")
    except Exception as e:
        _to_dlq(ev, f"psyche_error:{e}")


def _route_auto(ev: Event) -> None:
    if not _HAS_AN or not autonauka:
        return
    try:
        if ev.type in ("learn", "study", "ingest"):
            topic = ev.data.get("topic") or ev.data.get("q") or ""
            learner = getattr(autonauka, "learn", None)
            if not callable(learner):
                return
                res = learner(topic, ev.user)
                if res:
                    BUS.publish(
                        "memo",
                        "note",
                        {
                            "title": f"[AUTO]{topic or 'learn'}",
                            "sources": [{"text": str(res)}],
                        },
                        ev.user,
                    )
    except Exception as e:
        _to_dlq(ev, f"auto_error:{e}")


def _route_memo(ev: Event) -> None:
    if not _HAS_MEM or not memory:
        return
    try:
        if ev.type in ("note", "save", "ltm"):
            title = ev.data.get("title") or f"[{ev.channel}] {ev.type}"
            src = ev.data.get("sources") or [ev.data]
            tags = ev.data.get("tags") or [ev.channel]
            add_func = getattr(memory, "add", None)
            if add_func is not None and callable(add_func):
                add_func(title, sources=src, user=ev.user, tags=tags)
    except Exception as e:
        _to_dlq(ev, f"memory_error:{e}")


BUS.subscribe("psyche", _route_psyche)
BUS.subscribe("auto", _route_auto)
BUS.subscribe("memo", _route_memo)
BUS.on_before(lambda ev: METRICS.incr(ev.channel, "in", 1))


# === HMAC (API) ===
def _sig_base(
    ts_ms: int, channel: str, type_: str, user: str, data: dict[str, Any]
) -> bytes:
    payload = json.dumps(data or {}, ensure_ascii=False, separators=(",", ":"))
    s = f"{int(ts_ms)}.{channel}.{type_}.{user}.{payload}"
    return s.encode("utf-8")


def make_hmac(
    ts_ms: int, channel: str, type_: str, user: str, data: dict[str, Any]
) -> str:
    if not AUTH_TOKEN:
        return ""
    return hmac.new(
        AUTH_TOKEN.encode("utf-8"),
        _sig_base(ts_ms, channel, type_, user, data),
        hashlib.sha256,
    ).hexdigest()


def verify_hmac(
    ts_ms: int,
    channel: str,
    type_: str,
    user: str,
    data: dict[str, Any],
    signature: str,
) -> bool:
    if not REQUIRE_HMAC:
        return True
    if not (AUTH_TOKEN and signature):
        return False
    expect = make_hmac(ts_ms, channel, type_, user, data)
    try:
        return hmac.compare_digest(expect, signature)
    except Exception:
        return False


# użycie in-proc
def emit(
    channel: str,
    type_: str,
    data: dict[str, Any] | None = None,
    user: str = "global",
    idem_key: str | None = None,
) -> dict[str, Any]:
    ev = BUS.publish(channel, type_, data or {}, user, idem_key=idem_key)
    return {"ok": True, "ts": ev.ts, "id": ev.id}


# użycie z API (czas/ podpis z klienta)
def emit_secure(
    ts_ms: int,
    channel: str,
    type_: str,
    data: dict[str, Any] | None,
    user: str,
    signature: str,
    idem_key: str | None = None,
) -> dict[str, Any]:
    if not verify_hmac(ts_ms, channel, type_, user, data or {}, signature):
        return {"ok": False, "error": "invalid_signature"}
    # używamy klientowskiego timestampu dla spójności
    ev = Event(
        ts=int(ts_ms),
        channel=channel,
        type=type_,
        user=user or "global",
        data=(data or {}),
        id=str(uuid.uuid4()),
        idem=idem_key or None,
    )
    # wstaw do ring + log tak jak publish, ale z zewn. ts
    ok, err = _validate(channel, type_, ev.data)
    if not ok:
        return {"ok": False, "error": err}
    if not METRICS.allow_rate(ev.user, channel):
        return {"ok": False, "error": "rate_limited"}
    with BUS._lock:  # noqa
        BUS._rotate_if_needed()
        BUS._ring.append(ev)
        line = {
            "ts": ev.ts,
            "id": ev.id,
            "channel": ev.channel,
            "type": ev.type,
            "user": ev.user,
            "data": ev.data,
            "idem": ev.idem,
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        for cb in BUS._subs.get(ev.channel, []) + BUS._subs.get("*", []):
            try:
                cb(ev)
            except Exception:
                pass
    METRICS.incr(channel, "published", 1)
    return {"ok": True, "ts": ev.ts, "id": ev.id}


def tail(channel: str | None = None, n: int = 50) -> list[dict[str, Any]]:
    return BUS.recent(channel, limit=n)


def metrics() -> dict[str, Any]:
    return METRICS.snapshot()


def export_gz(out_path: str) -> str:
    src = LOG_FILE
    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f, _gzip.open(dst, "wb", compresslevel=6) as gz:
        shutil.copyfileobj(f, gz)
    return str(dst)


def import_jsonl(path: str) -> int:
    cnt = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                BUS.publish(
                    obj.get("channel", "misc"),
                    obj.get("type", "event"),
                    obj.get("data") or {},
                    obj.get("user", "global"),
                )
                cnt += 1
            except Exception:
                continue
    return cnt


# CLI (debug)
def _main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="io_pipeline PRO + HMAC")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("emit")
    g.add_argument("--chan", required=True)
    g.add_argument("--type", required=True)
    g.add_argument("--user", default="global")
    g.add_argument("--json", required=True)
    g.add_argument("--idem", default=None)
    t = sub.add_parser("tail")
    t.add_argument("--chan", default=None)
    t.add_argument("-n", type=int, default=50)
    sub.add_parser("metrics")
    e = sub.add_parser("export")
    e.add_argument("--out", required=True)
    i = sub.add_parser("import")
    i.add_argument("--path", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "emit":
        try:
            payload = json.loads(args.json)
        except Exception:
            payload = {"raw": args.json}
        print(
            json.dumps(
                emit(args.chan, args.type, payload, args.user, idem_key=args.idem),
                ensure_ascii=False,
            )
        )
        return 0
    if args.cmd == "tail":
        print(json.dumps(tail(args.chan, args.n), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "metrics":
        print(json.dumps(metrics(), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "export":
        print(json.dumps({"gz": export_gz(args.out)}, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "import":
        print(
            json.dumps(
                {"imported": import_jsonl(args.path)}, ensure_ascii=False, indent=2
            )
        )
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
