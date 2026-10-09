from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import src.__main__ as app_module
from src.config.settings import settings
from src.services.agent.cell_edit import CellEditFailed
from src.web.routes.cell import router as cell_router

BODY = {
    "block_id": "c1",
    "openrouter_api_key": "k",
    "model": "m",
    "context": {"user_id": "u", "workspace_id": "w", "document_id": "d", "user_token": "SECRET-TOKEN"},
}
HEADERS = {"x-handshake-token": settings.AI_HANDSHAKE_TOKEN}


@pytest.fixture
def client():
    return TestClient(app_module.app)  # no lifespan: no Redis or Qdrant needed


@pytest.fixture
def service(mocker):
    instance = mocker.patch.object(cell_router, "CellService").return_value
    instance.edit = AsyncMock()
    instance.fix = AsyncMock()
    return instance


def test_edit_reports_only_that_the_cell_was_changed_and_never_echoes_the_token(client, service):
    r = client.post("/sql/edit", json={**BODY, "prompt": "make it 2"}, headers=HEADERS)

    assert r.status_code == 200
    assert r.json() == {"cell_id": "c1", "updated": True}
    assert "SECRET-TOKEN" not in r.text
    service.edit.assert_awaited_once_with("make it 2")


def test_fix_is_offered_for_code_and_sql_but_not_markdown(client, service):
    assert client.post("/code/fix", json={**BODY, "error_message": "e"}, headers=HEADERS).status_code == 200
    assert client.post("/sql/fix", json={**BODY, "error_message": "e"}, headers=HEADERS).status_code == 200
    assert client.post("/markdown/fix", json={**BODY, "error_message": "e"}, headers=HEADERS).status_code == 404


def test_a_missing_user_token_is_a_400(client, service):
    service.edit.side_effect = ValueError("needs the signed-in user's token")

    r = client.post("/code/edit", json={**BODY, "prompt": "x"}, headers=HEADERS)

    assert r.status_code == 400 and "token" in r.json()["detail"]


def test_an_edit_the_notebook_server_refused_is_a_502(client, service):
    service.edit.side_effect = CellEditFailed("cell is running")

    r = client.post("/markdown/edit", json={**BODY, "prompt": "x"}, headers=HEADERS)

    assert r.status_code == 502 and "cell is running" in r.json()["detail"]


def test_requests_without_a_valid_handshake_token_are_refused(client, service):
    assert client.post("/sql/edit", json={**BODY, "prompt": "x"}).status_code == 400  # header missing
    wrong = client.post("/sql/edit", json={**BODY, "prompt": "x"}, headers={"x-handshake-token": "nope"})
    assert wrong.status_code == 401
    service.edit.assert_not_awaited()
