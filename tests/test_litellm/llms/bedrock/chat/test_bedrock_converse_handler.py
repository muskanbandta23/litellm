"""Tests for `BedrockConverseLLM.completion`.

AWS credential resolution is stubbed so nothing reaches STS.
"""

from __future__ import annotations

import asyncio
import binascii
import json
import logging
import struct
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Final
from unittest.mock import MagicMock, patch

import boto3
import httpx
import pytest
from botocore.credentials import Credentials
from botocore.exceptions import ClientError

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.bedrock.chat.converse_handler import BedrockConverseLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.rust_bridge import configuration
from litellm.types.utils import ModelResponse
from tests.test_litellm.llms.bedrock.event_loop_probe import EventLoopProbe

RESOLVED_CREDENTIALS = Credentials(
    access_key="AKIARESOLVED",
    secret_key="resolved-secret",
    token="resolved-token",
)


@pytest.fixture(autouse=True)
def reset_rust_configuration(monkeypatch):
    monkeypatch.setenv("LITELLM_RUST", "1")
    configuration.reset_rust_configuration()
    yield
    configuration.reset_rust_configuration()


def _completion_kwargs(**overrides):
    kwargs = {
        "model": "bedrock/us-east-1/anthropic.claude-sonnet-4-5-v1:0",
        "messages": [{"role": "user", "content": "hi"}],
        "api_base": None,
        "custom_prompt_dict": {},
        "model_response": ModelResponse(),
        "encoding": None,
        "logging_obj": MagicMock(),
        "optional_params": {"maxTokens": 16},
        "acompletion": False,
        "timeout": 30.0,
        "litellm_params": {},
        "extra_headers": None,
        "client": None,
        "api_key": None,
    }
    kwargs.update(overrides)
    return kwargs


def _run(*, credentials: Credentials | None = RESOLVED_CREDENTIALS, **overrides):
    with patch.object(BedrockConverseLLM, "get_credentials", return_value=credentials):
        return BedrockConverseLLM().completion(**_completion_kwargs(**overrides))


CONVERSE_RESPONSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7},
}


async def _drive_async_completion(
    *,
    skip_pre_call_logging: bool,
    logging_obj,
    credentials: Credentials = RESOLVED_CREDENTIALS,
    outer_dispatch: bool = False,
):
    """Run the real `async_completion` with a stubbed transport."""
    import httpx as _httpx

    client = MagicMock()

    async def post(**_kwargs):
        return _httpx.Response(
            200,
            json=CONVERSE_RESPONSE,
            request=_httpx.Request("POST", "https://bedrock-runtime.us-west-2.amazonaws.com"),
        )

    client.post = post
    client.__class__ = AsyncHTTPHandler

    if outer_dispatch:
        return await _run(credentials=credentials, acompletion=True, client=client, logging_obj=logging_obj)

    return await BedrockConverseLLM().async_completion(
        model="anthropic.claude-sonnet-4-5-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        api_base="https://bedrock-runtime.us-west-2.amazonaws.com/model/m/converse",
        model_response=ModelResponse(),
        timeout=30.0,
        encoding=None,
        logging_obj=logging_obj,
        stream=None,
        optional_params={"maxTokens": 16},
        litellm_params={"aws_region_name": "us-west-2"},
        credentials=credentials,
        headers={},
        client=client,
        skip_pre_call_logging=skip_pre_call_logging,
    )


@pytest.mark.asyncio
async def test_async_completion_honors_the_pre_call_suppression():
    logging_obj = MagicMock()
    await _drive_async_completion(skip_pre_call_logging=True, logging_obj=logging_obj)
    assert logging_obj.pre_call.call_count == 0


@pytest.mark.asyncio
async def test_async_completion_logs_pre_call_by_default():
    """The suppression must be opt-in, so every existing caller keeps its log."""
    logging_obj = MagicMock()
    await _drive_async_completion(skip_pre_call_logging=False, logging_obj=logging_obj)
    assert logging_obj.pre_call.call_count == 1


@pytest.mark.asyncio
async def test_async_completion_signs_off_the_event_loop(monkeypatch):
    """Regression for issue #40165: botocore refreshes expiring credentials inside SigV4 signing with a
    blocking HTTP call, so `async_completion` must sign on a worker thread to keep the loop serving."""
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    probe = EventLoopProbe()
    release = asyncio.create_task(probe.release_refresh_from_the_loop())

    response = await _drive_async_completion(
        skip_pre_call_logging=False, logging_obj=MagicMock(), credentials=probe.credentials()
    )
    await release

    assert response.choices[0].message.content == "hi"
    assert probe.served_during_refresh is True


@pytest.mark.asyncio
@pytest.mark.parametrize("rust_enabled", (False, True))
async def test_python_only_async_dispatch_refreshes_credentials_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, rust_enabled: bool
) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("LITELLM_RUST", "1" if rust_enabled else "0")
    configuration.rust(rust_enabled)
    probe: Final = EventLoopProbe()
    release: Final = asyncio.create_task(probe.release_refresh_from_the_loop())

    response: Final = await _drive_async_completion(
        skip_pre_call_logging=False, logging_obj=MagicMock(), credentials=probe.credentials(), outer_dispatch=True
    )
    await release

    assert response.choices[0].message.content == "hi"
    assert probe.served_during_refresh is True


def _sync_client_returning_converse_response():
    client = MagicMock()
    client.post.side_effect = lambda **_kwargs: httpx.Response(
        200,
        json=CONVERSE_RESPONSE,
        request=httpx.Request("POST", "https://bedrock-runtime.us-west-2.amazonaws.com"),
    )
    client.__class__ = HTTPHandler
    return client


def test_the_sync_python_path_logs_pre_call_once():
    logging_obj = MagicMock()
    response = _run(
        logging_obj=logging_obj,
        client=_sync_client_returning_converse_response(),
    )

    assert response.choices[0].message.content == "hi"
    assert logging_obj.pre_call.call_count == 1


def test_bearer_token_auth_serves_when_boto3_resolves_no_sigv4_credentials(monkeypatch):
    """With only `AWS_BEARER_TOKEN_BEDROCK` configured boto3 resolves no
    credentials at all. The handler must not dereference that None: the bearer
    token signs the request on its own."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-bearer-token")
    client = _sync_client_returning_converse_response()

    response = _run(credentials=None, litellm_params={}, client=client)

    assert response.choices[0].message.content == "hi"
    sent_headers = client.post.call_args.kwargs["headers"]
    assert sent_headers["Authorization"] == "Bearer bedrock-bearer-token"


@pytest.mark.parametrize("configured_through", ["env_var", "api_key"])
def test_bearer_token_auth_never_runs_the_sigv4_credential_chain(monkeypatch, configured_through):
    """The deployment's AWS profile does not exist, so resolving SigV4 credentials
    raises; a bearer-token deployment must still serve the request, since the
    bearer token alone signs it."""
    if configured_through == "env_var":
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-bearer-token")
    else:
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    client = _sync_client_returning_converse_response()

    response = BedrockConverseLLM().completion(
        **_completion_kwargs(
            optional_params={"maxTokens": 16, "aws_profile_name": "litellm-no-such-aws-profile"},
            litellm_params={},
            client=client,
            api_key="bedrock-bearer-token" if configured_through == "api_key" else None,
        )
    )

    assert response.choices[0].message.content == "hi"
    assert client.post.call_args.kwargs["headers"]["Authorization"] == "Bearer bedrock-bearer-token"


def test_session_tags_sign_the_request_and_stay_out_of_the_body(monkeypatch):
    """The tagged STS session signs the Converse call and the tags never reach the request body (#34069)."""
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_WEB_IDENTITY_TOKEN_FILE", raising=False)
    monkeypatch.delenv("AWS_ROLE_ARN", raising=False)
    tags = [{"Key": "team", "Value": "genai"}]

    class FakeSTSClient:
        def get_caller_identity(self):
            return {"Arn": "arn:aws:iam::111111111111:user/litellm-proxy-pod"}

        def assume_role(self, **params):
            if list(params.get("Tags", ())) != tags:
                raise ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": "is not authorized to perform: sts:TagSession"}},
                    "AssumeRole",
                )
            return {
                "Credentials": {
                    "AccessKeyId": "ASIACONVERSETAGGED",
                    "SecretAccessKey": "assumed-secret",
                    "SessionToken": "assumed-session-token",
                    "Expiration": datetime.now(timezone.utc) + timedelta(minutes=30),
                }
            }

    client = _sync_client_returning_converse_response()
    with patch.object(boto3, "client", return_value=FakeSTSClient()):
        response = BedrockConverseLLM().completion(
            **_completion_kwargs(
                optional_params={
                    "maxTokens": 16,
                    "aws_region_name": "us-east-1",
                    "aws_access_key_id": "AKIACONVERSECALLER",
                    "aws_secret_access_key": "pod-caller-secret",
                    "aws_role_name": "arn:aws:iam::999999999999:role/litellm-converse-role",
                    "aws_session_name": "litellm-converse-session",
                    "aws_session_tags": tags,
                },
                litellm_params={},
                client=client,
            )
        )

    assert response.choices[0].message.content == "hi"
    sent = client.post.call_args.kwargs
    assert "Credential=ASIACONVERSETAGGED/" in sent["headers"]["Authorization"]
    assert "aws_session_tags" not in sent["data"]


_CONVERSE_MODEL: Final = "bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
_MESSAGES: Final[list[dict[str, str]]] = [{"role": "user", "content": "hi"}]
_PROBE: Final = {"_litellm_probe": 1}
_AWS_TEST_KWARGS: Final = {"aws_access_key_id": "test", "aws_secret_access_key": "test", "aws_region_name": "us-west-2"}


class _PreCallBodyRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.bodies: list[object] = []

    def log_pre_api_call(self, model: str, messages: object, kwargs: Mapping[str, object]) -> None:
        additional_args: Final = kwargs["additional_args"]
        assert isinstance(additional_args, Mapping)
        self.bodies.append(additional_args["complete_input_dict"])


def _recording_transport(wire: list[dict[str, object]], response: httpx.Response) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        wire.append(json.loads(request.content))
        return response

    return httpx.MockTransport(handle)


def _converse_event_frame(event_type: str, payload: Mapping[str, object]) -> bytes:
    def header(name: str, value: str) -> bytes:
        return bytes([len(name)]) + name.encode() + bytes([7]) + struct.pack(">H", len(value)) + value.encode()

    headers: Final = (
        header(":event-type", event_type)
        + header(":content-type", "application/json")
        + header(":message-type", "event")
    )
    body: Final = json.dumps(payload).encode()
    prelude: Final = struct.pack(">II", 12 + len(headers) + len(body) + 4, len(headers))
    framed: Final = prelude + struct.pack(">I", binascii.crc32(prelude)) + headers + body
    return framed + struct.pack(">I", binascii.crc32(framed))


_CONVERSE_STREAM: Final = b"".join(
    (
        _converse_event_frame("messageStart", {"role": "assistant"}),
        _converse_event_frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "hi"}}),
        _converse_event_frame("contentBlockStop", {"contentBlockIndex": 0}),
        _converse_event_frame("messageStop", {"stopReason": "end_turn"}),
        _converse_event_frame(
            "metadata", {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}
        ),
    )
)


def _assert_pre_call_body_is_the_wire_body(recorder: _PreCallBodyRecorder, wire: list[dict[str, object]]) -> None:
    assert len(wire) == 1 and len(recorder.bodies) == 1, (wire, recorder.bodies)
    assert isinstance(recorder.bodies[0], Mapping), type(recorder.bodies[0])
    assert json.loads(json.dumps(recorder.bodies[0])) == wire[0]


def test_converse_sync_pre_call_body_is_the_mapping_sent_on_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    wire: Final[list[dict[str, object]]] = []
    recorder: Final = _PreCallBodyRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    client: Final = HTTPHandler(
        client=httpx.Client(transport=_recording_transport(wire, httpx.Response(200, json=CONVERSE_RESPONSE)))
    )
    litellm.completion(model=_CONVERSE_MODEL, messages=_MESSAGES, client=client, extra_body=_PROBE, **_AWS_TEST_KWARGS)
    _assert_pre_call_body_is_the_wire_body(recorder, wire)


@pytest.mark.asyncio
async def test_converse_async_pre_call_body_is_the_mapping_sent_on_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    wire: Final[list[dict[str, object]]] = []
    recorder: Final = _PreCallBodyRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    client: Final = AsyncHTTPHandler(transport=_recording_transport(wire, httpx.Response(200, json=CONVERSE_RESPONSE)))
    await litellm.acompletion(
        model=_CONVERSE_MODEL, messages=_MESSAGES, client=client, extra_body=_PROBE, **_AWS_TEST_KWARGS
    )
    _assert_pre_call_body_is_the_wire_body(recorder, wire)


@pytest.mark.asyncio
async def test_converse_async_stream_pre_call_body_is_the_mapping_sent_on_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire: Final[list[dict[str, object]]] = []
    recorder: Final = _PreCallBodyRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    stream_response: Final = httpx.Response(
        200, content=_CONVERSE_STREAM, headers={"content-type": "application/vnd.amazon.eventstream"}
    )
    client: Final = AsyncHTTPHandler(transport=_recording_transport(wire, stream_response))
    stream: Final = await litellm.acompletion(
        model=_CONVERSE_MODEL, messages=_MESSAGES, stream=True, client=client, extra_body=_PROBE, **_AWS_TEST_KWARGS
    )
    chunks: Final = [chunk async for chunk in stream]
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "hi"
    _assert_pre_call_body_is_the_wire_body(recorder, wire)


def test_converse_top_level_owned_kwarg_folds_into_additional_fields_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    wire: Final[list[dict[str, object]]] = []
    client: Final = HTTPHandler(
        client=httpx.Client(transport=_recording_transport(wire, httpx.Response(200, json=CONVERSE_RESPONSE)))
    )
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        litellm.completion(
            model=_CONVERSE_MODEL, messages=_MESSAGES, client=client, _litellm_probe=1, **_AWS_TEST_KWARGS
        )
    warnings: Final = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and record.getMessage().startswith("LiteLLM-owned keys")
    ]
    assert len(warnings) == 1, warnings
    assert warnings[0].split("keys=")[-1].split(".")[-1] == "_litellm_probe"
    additional_fields: Final = wire[0]["additionalModelRequestFields"]
    assert isinstance(additional_fields, dict)
    assert additional_fields["_litellm_probe"] == 1
