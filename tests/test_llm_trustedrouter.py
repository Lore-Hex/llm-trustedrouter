import json

import llm
import pytest
from click.testing import CliRunner
from llm.cli import cli

import llm_trustedrouter

MODELS = {
    "data": [
        {
            "id": "trustedrouter/auto",
            "name": "TrustedRouter Auto",
            "context_length": 200000,
            "architecture": {"input_modalities": ["text"]},
            "supported_parameters": ["max_tokens"],
        },
        {
            "id": "openai/gpt-4o-mini",
            "name": "GPT-4o mini",
            "context_length": 128000,
            "architecture": {"input_modalities": ["text", "image"]},
            "supported_parameters": ["tools", "structured_outputs"],
        },
    ]
}


@pytest.fixture(autouse=True)
def user_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    monkeypatch.setenv("TRUSTEDROUTER_API_KEY", "test-key")
    return tmp_path


def test_model_ids_are_not_double_prefixed():
    assert llm_trustedrouter.llm_model_id("trustedrouter/auto") == "trustedrouter/auto"
    assert (
        llm_trustedrouter.llm_model_id("openai/gpt-4o-mini")
        == "trustedrouter/openai/gpt-4o-mini"
    )


def test_registers_models_from_catalog(httpx_mock):
    httpx_mock.add_response(url=llm_trustedrouter.MODELS_URL, json=MODELS)
    model = llm.get_model("trustedrouter/openai/gpt-4o-mini")
    assert isinstance(model, llm_trustedrouter.TrustedRouterChat)
    assert model.model_name == "openai/gpt-4o-mini"
    assert model.api_base == "https://api.trustedrouter.com/v1"
    assert model.needs_key == "trustedrouter"
    assert model.key_env_var == "TRUSTEDROUTER_API_KEY"
    assert model.vision and model.supports_tools and model.supports_schema
    auto = llm.get_model("trustedrouter/auto")
    assert not auto.vision and not auto.supports_tools
    assert isinstance(
        llm.get_async_model("trustedrouter/auto"),
        llm_trustedrouter.TrustedRouterAsyncChat,
    )


def test_no_models_without_key(monkeypatch, httpx_mock):
    monkeypatch.delenv("TRUSTEDROUTER_API_KEY")
    ids = [m.model_id for m in llm.get_models()]
    assert not any(i.startswith("trustedrouter/") for i in ids)
    assert httpx_mock.get_requests() == []


def test_model_list_is_cached(httpx_mock, user_dir):
    httpx_mock.add_response(url=llm_trustedrouter.MODELS_URL, json=MODELS)
    llm_trustedrouter.get_trustedrouter_models()
    llm_trustedrouter.get_trustedrouter_models()
    assert len(httpx_mock.get_requests()) == 1
    assert json.loads((user_dir / "trustedrouter_models.json").read_text()) == MODELS


def test_falls_back_to_stale_cache_on_error(httpx_mock, user_dir):
    (user_dir / "trustedrouter_models.json").write_text(json.dumps(MODELS))
    httpx_mock.add_response(url=llm_trustedrouter.MODELS_URL, status_code=503)
    models = llm_trustedrouter.get_trustedrouter_models(skip_cache=True)
    assert [m["id"] for m in models] == ["trustedrouter/auto", "openai/gpt-4o-mini"]


def test_prompt_sends_request_to_trustedrouter(httpx_mock, monkeypatch):
    # The openai client may use httpx or httpx2 internally; give it a mock
    # transport from whichever module it uses so no request leaves the test.
    import openai
    import openai._base_client as base_client

    httpx_mod = getattr(base_client, "httpx2", None) or base_client.httpx
    captured = []

    def handler(request):
        captured.append(request)
        return httpx_mod.Response(
            200,
            json={
                "id": "gen-1",
                "object": "chat.completion",
                "created": 0,
                "model": "trustedrouter/auto",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    real_openai = openai.OpenAI

    def mocked_openai(**kwargs):
        kwargs["http_client"] = httpx_mod.Client(
            transport=httpx_mod.MockTransport(handler)
        )
        return real_openai(**kwargs)

    monkeypatch.setattr(openai, "OpenAI", mocked_openai)
    httpx_mock.add_response(url=llm_trustedrouter.MODELS_URL, json=MODELS)
    model = llm.get_model("trustedrouter/auto")
    assert model.prompt("Hi", stream=False).text() == "Hello!"
    assert len(captured) == 1
    request = captured[0]
    assert str(request.url) == "https://api.trustedrouter.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert json.loads(request.content)["model"] == "trustedrouter/auto"


def test_models_command(httpx_mock):
    httpx_mock.add_response(url=llm_trustedrouter.MODELS_URL, json=MODELS)
    result = CliRunner().invoke(cli, ["trustedrouter", "models"])
    assert result.exit_code == 0, result.output
    assert "- id: openai/gpt-4o-mini" in result.output
