"""Focused unit coverage for the Hugging Face integration boundary.

These tests call the route functions directly so the asynchronous push worker,
token fallback, and job ownership checks are tested without requiring a live
FastAPI server or the Hugging Face service.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from iam_client.models import UserContext


def _user(org_id: str | None = "org-a") -> UserContext:
    return UserContext(
        user_id="user-1",
        email="user@example.test",
        roles=[],
        permissions=["model.use"],
        valid=True,
        org_id=org_id,
    )


@pytest.fixture
def hf_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(tmp_path / "registry"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    import omnibioai_model_registry.hf_routes as module

    module._JOBS.clear()
    return module


def test_run_push_success_uploads_immutable_version_and_card(hf_module, tmp_path, monkeypatch):
    vdir = tmp_path / "v1"
    vdir.mkdir()
    (vdir / "model.pt").write_bytes(b"weights")
    fake_api = MagicMock()
    fake_hf = SimpleNamespace(HfApi=MagicMock(return_value=fake_api))
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hf)

    hf_module._run_push(
        "job-1", vdir, "org/model", True, "secret", "test card", "mit",
    )

    assert hf_module._get_job("job-1") == {
        "status": "success",
        "url": "https://huggingface.co/org/model",
    }
    fake_api.create_repo.assert_called_once_with(
        repo_id="org/model", repo_type="model", private=True, exist_ok=True,
    )
    fake_api.upload_folder.assert_called_once_with(
        folder_path=str(vdir), repo_id="org/model", repo_type="model", token="secret",
    )
    card = fake_api.upload_file.call_args.kwargs["path_or_fileobj"].decode()
    assert "license: mit" in card
    assert "test card" in card
    assert not (vdir / "README.md").exists()


def test_run_push_records_downstream_failure(hf_module, tmp_path, monkeypatch):
    fake_api = MagicMock()
    fake_api.create_repo.side_effect = RuntimeError("HF unavailable")
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=MagicMock(return_value=fake_api)),
    )

    hf_module._run_push("job-2", tmp_path, "org/model", False, "secret", "", "apache-2.0")

    job = hf_module._get_job("job-2")
    assert job["status"] == "error"
    assert job["error"] == "HF unavailable"


def test_hf_push_requires_request_or_server_token(hf_module, monkeypatch):
    request = hf_module.HFPushRequest(
        task="t", model_name="m", version="v1", repo_id="org/m",
    )

    with pytest.raises(HTTPException) as exc_info:
        hf_module.hf_push(request, _user())

    assert exc_info.value.status_code == 400
    assert "HF_TOKEN" in exc_info.value.detail


def test_hf_push_uses_server_token_and_records_org_scoped_job(hf_module, monkeypatch):
    request = hf_module.HFPushRequest(
        task="t", model_name="m", version="v1", repo_id="org/m",
    )
    monkeypatch.setenv("HF_TOKEN", "server-secret")
    monkeypatch.setattr(hf_module._registry, "resolve_model", lambda **kwargs: Path("/vdir"))
    monkeypatch.setattr(hf_module.uuid, "uuid4", lambda: "job-fixed")
    audit = MagicMock()
    monkeypatch.setattr(hf_module, "_audit", audit)

    captured = {}

    class ImmediateThread:
        def __init__(self, *, target, args, daemon):
            captured.update(target=target, args=args, daemon=daemon)

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(hf_module.threading, "Thread", ImmediateThread)

    response = hf_module.hf_push(request, _user("org-a"))

    assert response.ok is True
    assert response.job_id == "job-fixed"
    assert hf_module._get_job("job-fixed")["organization_id"] == "org-a"
    assert captured["started"] is True
    assert captured["args"][4] == "server-secret"
    audit.log_event.assert_called_once()


def test_hf_status_hides_jobs_from_other_organizations(hf_module):
    hf_module._set_job("job-a", status="success", organization_id="org-a", url="safe")

    with pytest.raises(HTTPException) as exc_info:
        hf_module.hf_push_status("job-a", _user("org-b"))

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Unknown job_id"

    result = hf_module.hf_push_status("job-a", _user("org-a"))
    assert result.ok is True
    assert result.status == "success"
    assert result.url == "safe"


def test_hf_status_unknown_job_is_non_enumerating(hf_module):
    with pytest.raises(HTTPException) as exc_info:
        hf_module.hf_push_status("does-not-exist", _user())

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Unknown job_id"
