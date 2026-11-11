"""Conversation identities are persistent, dataset-bound, and reused by the UI."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
import pytest

import agent
import api
from conversations import resolve_conversation


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
    runner = Mock(return_value={"messages": [AIMessage(content="done")]})
    monkeypatch.setattr(api, "run_dataset_agent", runner)
    with TestClient(api.app) as client:
        yield client, runner


def upload(client):
    response = client.post("/upload", files={"file": ("sample.csv", "revenue\n1\n", "text/csv")})
    assert response.status_code == 200
    return response.json()["dataset_id"]


def test_first_question_creates_id_and_followup_reuses_it(client):
    client, runner = client
    dataset = upload(client)
    first = client.post("/ask", json={"question": "first", "dataset_id": dataset})
    assert first.status_code == 200
    conversation = first.json()["conversation_id"]
    assert str(UUID(conversation)) == conversation
    second = client.post("/ask", json={"question": "next", "dataset_id": dataset,
                                       "conversation_id": conversation})
    assert second.status_code == 200
    assert second.json()["conversation_id"] == conversation
    assert runner.call_count == 2
    assert runner.call_args.kwargs["question"] == "next"
    assert runner.call_args.kwargs["dataset_path"] == api.UPLOAD_DIR / f"{dataset}.csv"
    assert [m.content for m in runner.call_args.kwargs["history"]] == ["first", "done"]


def test_omitting_id_starts_separate_conversations_on_same_dataset(client):
    client, _ = client
    dataset = upload(client)
    ids = [client.post("/ask", json={"question": "test", "dataset_id": dataset}).json()["conversation_id"]
           for _ in range(2)]
    assert ids[0] != ids[1]


def test_conversation_cannot_switch_dataset(client):
    client, runner = client
    first, second = upload(client), upload(client)
    conversation = client.post("/ask", json={"question": "first", "dataset_id": first}).json()["conversation_id"]
    runner.reset_mock()
    response = client.post("/ask", json={"question": "test", "dataset_id": second,
                                         "conversation_id": conversation})
    assert response.status_code == 409
    runner.assert_not_called()


def test_unknown_id_is_rejected_before_model_call(client):
    client, runner = client
    response = client.post("/ask", json={"question": "test", "dataset_id": upload(client),
                                         "conversation_id": str(uuid4())})
    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found"
    runner.assert_not_called()


def test_invalid_id_is_rejected(client):
    client, runner = client
    response = client.post("/ask", json={"question": "test", "dataset_id": upload(client),
                                         "conversation_id": "not-a-uuid"})
    assert response.status_code == 422
    runner.assert_not_called()


def test_binding_survives_new_connections_and_concurrent_creations(tmp_path):
    path = tmp_path / "conversations.db"
    dataset = uuid4()
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(lambda _: resolve_conversation(path, dataset, None), range(8)))
    assert len(set(ids)) == 8
    for conversation in ids:
        assert resolve_conversation(path, dataset, conversation) == conversation


def test_frontend_reuses_id_and_new_conversation_resets_it(monkeypatch):
    import requests
    from streamlit.testing.v1 import AppTest

    first, second = str(uuid4()), str(uuid4())
    responses = []
    for conversation in (first, first, second):
        response = Mock()
        response.json.return_value = {"answer": "done", "conversation_id": conversation}
        responses.append(response)
    post = Mock(side_effect=responses)
    monkeypatch.setattr(requests, "post", post)
    ui = AppTest.from_file(str(agent.PROJECT_DIR / "app.py"))
    ui.session_state["dataset_id"] = "sample-dataset"
    ui.run()
    ui.chat_input[0].set_value("first").run()
    assert not ui.exception
    assert post.call_args.kwargs["json"]["conversation_id"] is None
    assert ui.session_state["conversation_id"] == first
    ui.chat_input[0].set_value("follow up").run()
    assert not ui.exception
    assert post.call_args.kwargs["json"]["conversation_id"] == first
    ui.button(key="new_conversation").click().run()
    assert not ui.exception
    assert ui.session_state["conversation_id"] is None
    assert ui.session_state["messages"] == []
    assert ui.session_state["dataset_id"] == "sample-dataset"
    ui.chat_input[0].set_value("new first question").run()
    assert not ui.exception
    assert post.call_args.kwargs["json"]["conversation_id"] is None
    assert ui.session_state["conversation_id"] == second


def test_loading_dataset_clears_old_conversation(monkeypatch):
    import requests
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    uploaded = Mock()
    uploaded.name = "new.csv"
    uploaded.getvalue.return_value = b"revenue\n1\n"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    response = Mock()
    response.json.return_value = {"dataset_id": "new-dataset", "row_count": 1, "columns": ["revenue"]}
    monkeypatch.setattr(requests, "post", Mock(return_value=response))
    ui = AppTest.from_file(str(agent.PROJECT_DIR / "app.py"))
    ui.session_state["dataset_id"] = "old-dataset"
    ui.session_state["conversation_id"] = str(uuid4())
    ui.session_state["messages"] = [{"role": "assistant", "content": "old"}]
    ui.run()
    next(button for button in ui.button if button.label == "Load Dataset").click().run()
    assert not ui.exception
    assert ui.session_state["dataset_id"] == "new-dataset"
    assert ui.session_state["conversation_id"] is None
    assert ui.session_state["messages"] == []
