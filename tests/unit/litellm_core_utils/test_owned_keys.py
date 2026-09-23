import json
from collections.abc import Mapping
from typing import Final

import pytest

from litellm.litellm_core_utils.owned_keys import owned_keys_in, request_body_view


def test_top_level_owned_key_is_flagged() -> None:
    body: Final = {"model": "gpt-5.4", "model_info": {"id": "x"}}

    assert owned_keys_in(body) == ("model_info",)


@pytest.mark.parametrize("key", ("_litellm_probe", "_litellm_zzz"))
def test_prefixed_key_is_flagged(key: str) -> None:
    assert owned_keys_in({"model": "gpt-5.4", key: 1}) == (key,)


@pytest.mark.parametrize(
    ("container", "inner", "expected"),
    (
        ("metadata", {"user_api_key_hash": "h"}, ("metadata.user_api_key_hash",)),
        ("extra_body", {"_litellm_probe": 1}, ("_litellm_probe",)),
        ("litellm_metadata", {"user_api_key_hash": "h"}, ("litellm_metadata", "litellm_metadata.user_api_key_hash")),
        (
            "additionalModelRequestFields",
            {"_litellm_probe": 1},
            ("additionalModelRequestFields._litellm_probe",),
        ),
    ),
)
def test_owned_keys_one_level_under_any_mapping_are_flagged(
    container: str, inner: Mapping[str, object], expected: tuple[str, ...]
) -> None:
    assert owned_keys_in({"model": "gpt-5.4", container: inner}) == expected


def test_extra_body_is_flattened_before_walk() -> None:
    body: Final = {"model": "gpt-5.4", "extra_body": {"metadata": {"user_api_key_hash": "h"}}}

    assert owned_keys_in(body) == ("metadata.user_api_key_hash",)


@pytest.mark.parametrize("extra_body", ("user_api_key_hash", ["user_api_key_hash"], [{"user_api_key_hash": "h"}], None))
def test_non_mapping_extra_body_is_left_alone(extra_body: object) -> None:
    body: Final = {"model": "gpt-5.4", "extra_body": extra_body}

    assert request_body_view(body) is body
    assert owned_keys_in(body) == ()


def test_extra_body_flatten_reports_every_owned_key_once() -> None:
    body: Final = {
        "model": "gpt-5.4",
        "model_info": {},
        "extra_body": {"_litellm_probe": 1, "metadata": {"user_api_key_hash": "h", "litellm_call_id": "c"}},
    }

    assert owned_keys_in(body) == (
        "_litellm_probe",
        "metadata.litellm_call_id",
        "metadata.user_api_key_hash",
        "model_info",
    )


def test_key_two_levels_deep_is_not_flagged() -> None:
    body: Final = {
        "model": "gpt-5.4",
        "additionalModelRequestFields": {"extra_body": {"_litellm_probe": 1}},
    }

    assert owned_keys_in(body) == ()


def test_owned_key_inside_a_list_held_mapping_is_not_flagged() -> None:
    body: Final = {
        "model": "gpt-5.4",
        "messages": [{"_litellm_probe": 1, "user_api_key_hash": "h"}],
    }

    assert owned_keys_in(body) == ()


def test_owned_names_inside_tool_schemas_and_tool_arguments_are_not_flagged() -> None:
    arguments: Final = json.dumps({"model_info": {}, "rpm": 1, "tpm": 1, "user_api_key_hash": "h"})
    body: Final = {
        "model": "gpt-5.4",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": arguments}}],
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_info": {"type": "string"},
                            "rpm": {"type": "integer"},
                            "tpm": {"type": "integer"},
                            "user_api_key_hash": {"type": "string"},
                            "_litellm_probe": {"type": "string"},
                        },
                    },
                },
            }
        ],
    }

    assert owned_keys_in(body) == ()


def test_plain_metadata_is_not_flagged() -> None:
    assert owned_keys_in({"model": "gpt-5.4", "metadata": {"foo": "bar"}}) == ()


_OWNED_KEYS: Final = (
    "user_api_key_hash",
    "user_api_key_alias",
    "user_api_key_team_id",
    "user_api_key_user_id",
    "user_api_key_org_id",
    "user_api_key_end_user_id",
    "litellm_api_version",
    "litellm_call_id",
    "litellm_logging_obj",
    "litellm_metadata",
    "litellm_trace_id",
    "proxy_server_request",
    "global_max_parallel_requests",
    "model_info",
)


@pytest.mark.parametrize("key", _OWNED_KEYS)
def test_every_owned_key_is_flagged_at_top_level(key: str) -> None:
    assert owned_keys_in({"model": "gpt-5.4", key: 1}) == (key,)


@pytest.mark.parametrize(
    "key", ("user", "stream_chunk_size", "litellm_probe", "_litellmx", "_litellm", "x_litellm_y", "x_litellm_")
)
def test_non_owned_and_prefix_boundary_keys_are_not_flagged(key: str) -> None:
    assert owned_keys_in({"model": "gpt-5.4", key: 1}) == ()


def test_result_is_sorted_regardless_of_insertion_order() -> None:
    expected: Final = (
        "_litellm_probe",
        "metadata.litellm_call_id",
        "metadata.user_api_key_hash",
        "model_info",
    )
    forward: Final = {
        "extra_body": {"_litellm_probe": 1},
        "metadata": {"user_api_key_hash": "h", "litellm_call_id": "c"},
        "model_info": {},
    }
    reversed_order: Final = {
        "model_info": {},
        "metadata": {"litellm_call_id": "c", "user_api_key_hash": "h"},
        "extra_body": {"_litellm_probe": 1},
    }

    assert owned_keys_in(forward) == expected
    assert owned_keys_in(reversed_order) == expected
    assert owned_keys_in(forward) == owned_keys_in(reversed_order)
