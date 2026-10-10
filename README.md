# llm-trustedrouter

[![Tests](https://github.com/Lore-Hex/llm-trustedrouter/actions/workflows/test.yml/badge.svg)](https://github.com/Lore-Hex/llm-trustedrouter/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/Lore-Hex/llm-trustedrouter/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin for models available through [TrustedRouter](https://trustedrouter.com), an OpenAI-compatible LLM router.

## Installation

```bash
llm install llm-trustedrouter
```

## Configuration

Set your API key:

```bash
llm keys set trustedrouter
# Paste key here
```

or set the `TRUSTEDROUTER_API_KEY` environment variable.

## Usage

Models are registered from the public catalog at `https://api.trustedrouter.com/v1/models` once a key is configured. The list is cached for one hour in `trustedrouter_models.json` in your LLM user directory.

```bash
llm models -q trustedrouter
llm -m trustedrouter/auto "Five names for a pet pelican"
```

TrustedRouter model IDs are already prefixed (`trustedrouter/auto`) and are used as-is; other catalog IDs get a `trustedrouter/` prefix (`trustedrouter/openai/gpt-4o-mini`).

Plugin commands:

```bash
llm trustedrouter models          # list catalog models
llm trustedrouter models --json   # full catalog JSON
llm trustedrouter refresh         # refresh the cached catalog
```

## Verifying the gateway

The gateway runs as an attested Google Confidential Space workload. The attestation is served at https://api.trustedrouter.com/attestation and the verification procedure is at https://trustedrouter.com/trust (source: https://github.com/Lore-Hex/quill-cloud-proxy).

## Development

```bash
python -m venv venv && source venv/bin/activate
pip install -e '.[test]'
python -m pytest
```

Tests mock all HTTP and make no network calls.
