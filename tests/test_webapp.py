"""Smoke test for the FastAPI UI."""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from tests.fixtures import make_synthetic_plan


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Point the app at a fresh jobs dir before it is imported.
    monkeypatch.setenv("JOBS_DIR", str(tmp_path / "jobs"))
    # Reload module so it picks up the env var.
    import importlib
    import webapp.main as main_module
    importlib.reload(main_module)
    return TestClient(main_module.app)


def _plan_bytes(tmp_path: Path) -> bytes:
    p = tmp_path / "plan.png"
    make_synthetic_plan(p)
    return p.read_bytes()


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Векторизация" in r.text or "vectoriz" in r.text.lower()


def test_upload_and_download(client, tmp_path):
    img_bytes = _plan_bytes(tmp_path)
    files = {"plan": ("plan.png", io.BytesIO(img_bytes), "image/png")}
    data = {"floor_id": "ui_test"}

    r = client.post("/", files=files, data=data, follow_redirects=False)
    assert r.status_code == 303
    location = r.headers["location"]
    assert location.startswith("/jobs/")
    job_id = location.rsplit("/", 1)[-1]

    # Poll status until done.
    deadline = time.time() + 30
    state = "pending"
    while time.time() < deadline:
        r = client.get(f"/jobs/{job_id}/status")
        assert r.status_code == 200
        state = r.json()["state"]
        if state in ("done", "error"):
            break
        time.sleep(0.3)
    assert state == "done", f"job did not finish: {state}"

    # Files list.
    r = client.get(f"/jobs/{job_id}")
    assert r.status_code == 200
    assert "lots.svg" in r.text

    # Direct file fetch.
    r = client.get(f"/jobs/{job_id}/files/lots.svg")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg+xml")
    assert b"<polygon" in r.content

    # Zip bundle.
    r = client.get(f"/jobs/{job_id}/download")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert len(r.content) > 0
