"""tests/test_service_warmup.py — the service answers /health while its models load.

The lifespan loads the models in a daemon thread and serves at once: /health is 200 from the
start, /ready and the classification endpoints are 503 until the models are loaded. A first
start downloads about 2.4 GB, and loading them before serving left /health unanswered all that
time, so a container's health check could fail before the start finished.

These run the real lifespan (a TestClient used as a context manager) with a warm-up held on an
Event, and real uvicorn processes for the SIGTERM cases.
"""

import io
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from tests.openapi_contract_data import prepare  # noqa: E402

prepare()
from service import api  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class _HeldManager:
    """A model manager whose warm-up waits until the test releases it."""

    device = "cpu"
    available_versions = ["v1.4", "v2.4", "v3.4", "v4.4", "v5.4"]

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def warmup(self, versions=None):
        self.started.set()
        self.release.wait(10)

    def get_model_details(self, version):
        return f"stub ({version})"

    def predict_detailed(self, image, version, topn):
        result = [{"label": "TEXT", "score": 0.9}][:topn]
        return {"predictions": result, "models_used": [version], "per_model": {version: result}}


@pytest.fixture
def held(monkeypatch):
    manager = _HeldManager()
    monkeypatch.setattr(api, "manager", manager)
    saved = (api._state.warm, api._state.draining, api._warming, api._warmup_thread)
    yield manager
    manager.release.set()
    api._state.warm, api._state.draining, api._warming, api._warmup_thread = saved


def _png():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color="white").save(buf, format="PNG")
    return buf.getvalue()


def _classify(client):
    return client.post("/predict_image", files={"file": ("page.png", _png(), "image/png")})


def _wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def test_health_answers_while_the_models_load_and_classification_waits(held):
    with TestClient(api.app) as client:
        assert held.started.wait(5), "the lifespan did not start the warm-up"

        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        assert (ready.status_code, ready.json()["status"]) == (503, "starting")
        refused = _classify(client)
        assert refused.status_code == 503
        assert "still loading" in refused.json()["detail"]

        held.release.set()
        assert _wait_for(lambda: api._state.warm), "the warm-up never marked the service warm"
        assert client.get("/ready").status_code == 200
        assert _classify(client).status_code == 200


def test_a_failed_warm_up_still_ends_the_start(held, monkeypatch):
    """warmup() handles each model's failure itself; an error past that must not leave /ready
    at 503 for good. The models then load on their first request, as a failed one does now."""

    def _broken(versions=None):
        raise RuntimeError("disk full")

    monkeypatch.setattr(held, "warmup", _broken)
    with TestClient(api.app) as client:
        assert _wait_for(lambda: api._state.warm)
        assert client.get("/ready").status_code == 200
        assert _classify(client).status_code == 200


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _status(url):
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except OSError:
        return None


def test_sigterm_during_the_warm_up_exits_promptly():
    """The warm-up runs in a daemon thread. With asyncio.to_thread the server did not answer
    until the models were loaded, and the executor's thread was joined at exit, so a SIGTERM
    during the download waited for it; a container's stop then ran into its kill timeout."""
    pytest.importorskip("uvicorn")
    port = _free_port()
    script = textwrap.dedent(
        f"""
        import threading
        from tests.openapi_contract_data import prepare
        prepare()
        from service import api
        api.manager.warmup = lambda versions=None: threading.Event().wait(600)
        import uvicorn
        uvicorn.run(api.app, host="127.0.0.1", port={port}, log_level="warning")
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        assert _wait_for(lambda: _status(f"{base}/health") == 200, timeout=60), "no /health answer during warm-up"
        assert _status(f"{base}/ready") == 503

        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
            pytest.fail("the service did not exit within 15 s of SIGTERM during its warm-up")


class _Exited(Exception):
    """Raised by the stand-in for os._exit, so the test sees the exit code instead of ending."""


def _stub_exit(monkeypatch):
    def _exit(code):
        raise _Exited(code)

    monkeypatch.setattr(os, "_exit", _exit)
    monkeypatch.setattr(api.logging, "shutdown", lambda: None)  # pytest's own log handlers stay open


def test_a_stop_during_the_warm_up_ends_the_process_without_teardown(monkeypatch):
    """The server stopped while the warm-up thread still loads: os._exit(0), no interpreter
    teardown beside that thread's native code (exit 139 in the `api` image on 11-10 and 11-14)."""
    _stub_exit(monkeypatch)
    release = threading.Event()
    thread = threading.Thread(target=release.wait, args=(10,), daemon=True)
    thread.start()
    monkeypatch.setattr(api, "_warmup_thread", thread)
    try:
        with pytest.raises(_Exited) as exited:
            api._exit_now_if_warming()
        assert exited.value.args == (0,)
    finally:
        release.set()
        thread.join(5)


def test_a_finished_or_absent_warm_up_leaves_the_exit_to_python(monkeypatch):
    _stub_exit(monkeypatch)
    monkeypatch.setattr(api, "_warmup_thread", None)
    api._exit_now_if_warming()
    finished = threading.Thread(target=lambda: None)
    finished.start()
    finished.join()
    monkeypatch.setattr(api, "_warmup_thread", finished)
    api._exit_now_if_warming()


def test_a_stop_during_the_warm_up_as_pid_1_exits_cleanly():
    """In the container the service is PID 1, which ignores the SIGTERM uvicorn raises again after
    its graceful exit, so uvicorn.run() returns. Here SIGTERM is ignored before uvicorn starts,
    which has the same effect; the service's own entry point then calls _exit_now_if_warming()."""
    pytest.importorskip("uvicorn")
    port = _free_port()
    script = textwrap.dedent(
        f"""
        import signal
        import threading
        from tests.openapi_contract_data import prepare
        prepare()
        from service import api
        api.manager.warmup = lambda versions=None: threading.Event().wait(600)
        import uvicorn
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        uvicorn.run(api.app, host="127.0.0.1", port={port}, log_level="warning")
        print("RUN RETURNED", flush=True)
        api._exit_now_if_warming()
        print("TEARDOWN REACHED", flush=True)
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        assert _wait_for(lambda: _status(f"{base}/health") == 200, timeout=60), "no /health answer during warm-up"
        assert _status(f"{base}/ready") == 503

        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
            pytest.fail("the service did not exit within 30 s of SIGTERM during its warm-up")
    assert "RUN RETURNED" in out, err[-2000:]
    assert "TEARDOWN REACHED" not in out, "the process went on to Python's teardown with the warm-up running"
    assert proc.returncode == 0, err[-2000:]
