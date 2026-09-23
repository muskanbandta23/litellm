from collections.abc import Mapping
from typing import Final

OWNED_KEYS: Final[frozenset[str]] = frozenset(
    (
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
)
OWNED_PREFIX: Final = "_litellm_"


def is_owned_key(key: str) -> bool:
    return key in OWNED_KEYS or key.startswith(OWNED_PREFIX)


def owned_keys_in(body: Mapping[str, object]) -> tuple[str, ...]:
    top_level: Final = (key for key in body if isinstance(key, str) and is_owned_key(key))
    nested: Final = (
        f"{container}.{key}"
        for container, inner in body.items()
        if isinstance(container, str) and isinstance(inner, Mapping)
        for key in inner
        if isinstance(key, str) and is_owned_key(key)
    )
    return tuple(sorted((*top_level, *nested)))
