import json
import time
from pathlib import Path

import click
import httpx
import llm
from llm.default_plugins.openai_models import AsyncChat, Chat

API_BASE = "https://api.trustedrouter.com/v1"
MODELS_URL = API_BASE + "/models"
KEY_NAME = "trustedrouter"
KEY_ENV_VAR = "TRUSTEDROUTER_API_KEY"
CACHE_SECONDS = 3600


class DownloadError(Exception):
    pass


def cache_path():
    return llm.user_dir() / "trustedrouter_models.json"


def fetch_cached_json(url, path, cache_timeout):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and time.time() - path.stat().st_mtime < cache_timeout:
        return json.loads(path.read_text())
    try:
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()
        data = response.json()
        path.write_text(json.dumps(data))
        return data
    except httpx.HTTPError:
        if path.is_file():
            return json.loads(path.read_text())
        raise DownloadError(
            f"Failed to download {url} and no cache is available at {path}"
        )


def get_trustedrouter_models(skip_cache=False):
    return fetch_cached_json(
        MODELS_URL, cache_path(), 0 if skip_cache else CACHE_SECONDS
    )["data"]


def llm_model_id(model_name):
    if model_name.startswith("trustedrouter/"):
        return model_name
    return "trustedrouter/" + model_name


def has_parameter(model_definition, parameter):
    return parameter in (model_definition.get("supported_parameters") or [])


def supports_images(model_definition):
    modalities = (model_definition.get("architecture") or {}).get(
        "input_modalities"
    ) or []
    return "image" in modalities


class TrustedRouterChat(Chat):
    needs_key = KEY_NAME
    key_env_var = KEY_ENV_VAR

    def __str__(self):
        return "TrustedRouter: {}".format(self.model_id)


class TrustedRouterAsyncChat(AsyncChat):
    needs_key = KEY_NAME
    key_env_var = KEY_ENV_VAR

    def __str__(self):
        return "TrustedRouter: {}".format(self.model_id)


def model_kwargs(model_definition):
    return dict(
        model_id=llm_model_id(model_definition["id"]),
        model_name=model_definition["id"],
        api_base=API_BASE,
        vision=supports_images(model_definition),
        reasoning=has_parameter(model_definition, "reasoning"),
        supports_schema=has_parameter(model_definition, "structured_outputs"),
        supports_tools=has_parameter(model_definition, "tools"),
    )


@llm.hookimpl
def register_models(register):
    # Only register models once a key is configured, like llm-openrouter
    if not llm.get_key("", KEY_NAME, KEY_ENV_VAR):
        return
    try:
        models = get_trustedrouter_models()
    except DownloadError:
        return
    for model_definition in models:
        kwargs = model_kwargs(model_definition)
        register(TrustedRouterChat(**kwargs), TrustedRouterAsyncChat(**kwargs))


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def trustedrouter():
        "Commands relating to the llm-trustedrouter plugin"

    @trustedrouter.command()
    @click.option("json_", "--json", is_flag=True, help="Output as JSON")
    def models(json_):
        "List available TrustedRouter models"
        models = get_trustedrouter_models()
        if json_:
            click.echo(json.dumps(models, indent=2))
            return
        for model in models:
            click.echo("- id: {}".format(model["id"]))
            click.echo("  name: {}".format(model.get("name", "")))
            click.echo("  context_length: {}".format(model.get("context_length")))

    @trustedrouter.command()
    def refresh():
        "Refresh the cached list of TrustedRouter models"
        before = {m["id"] for m in json.loads(cache_path().read_text())["data"]} if cache_path().is_file() else set()
        after = {m["id"] for m in get_trustedrouter_models(skip_cache=True)}
        added = sorted(after - before)
        removed = sorted(before - after)
        click.echo("{} models; added {}, removed {}".format(len(after), len(added), len(removed)))
