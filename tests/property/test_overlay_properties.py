"""Property-based tests for aria.overlay using Hypothesis.

All network I/O is replaced by a lightweight in-process mock so no real
overlay node is required.  The mock is injected via the `client` parameter
that every overlay class accepts for exactly this purpose.

Invariants tested:
    - TopicManager.submit always returns AdmittanceResult.
    - admitted == (len(admitted_outputs) > 0), always.
    - admitted_outputs is always list[int].
    - URL trailing slashes are stripped in _base.
    - topic name is always preserved end-to-end.
    - txid from the HTTP response always appears in AdmittanceResult.
    - LookupService.lookup always returns list[LookupResult].
    - LookupResult fields are correctly typed.
    - OverlayClient.find_epochs query always contains system_id.
    - OverlayClient.find_epochs includes epoch_id only when provided.
    - api_key is forwarded in Authorization: Bearer header.
    - Payloads sent over the wire are always JSON-serializable dicts.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aria.overlay import (
    AdmittanceResult,
    LookupResult,
    LookupService,
    OverlayClient,
    TopicManager,
)


# ---------------------------------------------------------------------------
# Minimal async httpx mock — no real network, no dependency on unittest.mock
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Mimics the subset of httpx.Response used by aria.overlay."""

    def __init__(self, data: dict[str, Any], status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code
        self.is_success = status_code < 400
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._data


class _FakeClient:
    """Async httpx.AsyncClient stand-in that records calls and returns canned data."""

    def __init__(self, response_data: dict[str, Any], status_code: int = 200) -> None:
        self._response_data = response_data
        self._status_code = status_code
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return _FakeResponse(self._response_data, self._status_code)


def _submit_response(topic: str, outputs: list[int], txid: str = "aa" * 32) -> dict:
    """Build a minimal BRC-31 submit response."""
    return {
        "txid": txid,
        "topics": {topic: {"outputsToAdmit": outputs, "coinsToRetain": []}},
    }


def _lookup_response(results: list[dict]) -> dict:
    """Build a minimal BRC-31 lookup response."""
    return {"results": results}


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid topic names: letters + underscores, like "tm_aria_epochs"
topic_name = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz_",
    min_size=1,
    max_size=40,
)

# Arbitrary hex strings as stand-ins for raw transactions
raw_hex = st.text(alphabet="0123456789abcdef", min_size=2, max_size=200)

# 64-char lowercase hex txids
txid_hex = st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)

# Simple URL with optional trailing slash(es)
base_url = st.one_of(
    st.just("http://localhost:8080"),
    st.just("http://localhost:8080/"),
    st.just("http://overlay.example.com/node/"),
    st.just("https://api.example.com"),
    st.just("https://api.example.com///"),
)

# Arbitrary output index lists
output_index_list = st.lists(
    st.integers(min_value=0, max_value=9),
    min_size=0,
    max_size=5,
)

# Arbitrary string suitable as a system/epoch identifier
ident = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
    min_size=1,
    max_size=50,
)

# JSON-safe scalar for arbitrary payload fields
json_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-999, max_value=999),
    st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=False),
    st.text(max_size=20),
)

payload_dict = st.dictionaries(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10),
    json_scalar,
    max_size=5,
)


# ---------------------------------------------------------------------------
# TopicManager properties
# ---------------------------------------------------------------------------


@given(topic=topic_name, outputs=output_index_list, txid=txid_hex)
@settings(max_examples=60)
def test_topic_manager_submit_always_returns_admittance_result(
    topic: str, outputs: list[int], txid: str
) -> None:
    """TopicManager.submit always returns an AdmittanceResult, regardless of topic."""
    client = _FakeClient(_submit_response(topic, outputs, txid))
    tm = TopicManager(base_url="http://example.com", topic=topic, client=client)
    result = asyncio.run(tm.submit("deadbeef"))
    assert isinstance(result, AdmittanceResult)


@given(topic=topic_name, outputs=output_index_list)
@settings(max_examples=60)
def test_topic_manager_admitted_iff_outputs_non_empty(
    topic: str, outputs: list[int]
) -> None:
    """admitted is True exactly when admitted_outputs is non-empty."""
    client = _FakeClient(_submit_response(topic, outputs))
    tm = TopicManager(base_url="http://example.com", topic=topic, client=client)
    result = asyncio.run(tm.submit("deadbeef"))
    assert result.admitted == (len(outputs) > 0)


@given(topic=topic_name, outputs=output_index_list)
@settings(max_examples=60)
def test_topic_manager_admitted_outputs_is_list_of_int(
    topic: str, outputs: list[int]
) -> None:
    """admitted_outputs is always a list of integers."""
    client = _FakeClient(_submit_response(topic, outputs))
    tm = TopicManager(base_url="http://example.com", topic=topic, client=client)
    result = asyncio.run(tm.submit("deadbeef"))
    assert isinstance(result.admitted_outputs, list)
    assert all(isinstance(x, int) for x in result.admitted_outputs)


@given(topic=topic_name, txid=txid_hex)
@settings(max_examples=60)
def test_topic_manager_txid_from_response_appears_in_result(
    topic: str, txid: str
) -> None:
    """The txid returned by the overlay node appears in the AdmittanceResult."""
    client = _FakeClient(_submit_response(topic, [0], txid))
    tm = TopicManager(base_url="http://example.com", topic=topic, client=client)
    result = asyncio.run(tm.submit("cafebabe"))
    assert result.txid == txid


@given(topic=topic_name)
@settings(max_examples=40)
def test_topic_manager_preserves_topic_name(topic: str) -> None:
    """The topic property always returns the topic passed to the constructor."""
    tm = TopicManager(base_url="http://example.com", topic=topic)
    assert tm.topic == topic


@given(url=base_url, topic=topic_name)
@settings(max_examples=40)
def test_topic_manager_strips_trailing_slash_from_url(url: str, topic: str) -> None:
    """_base always strips trailing slashes from the provided base URL."""
    tm = TopicManager(base_url=url, topic=topic)
    assert not tm._base.endswith("/"), f"_base should have no trailing slash: {tm._base!r}"


@given(topic=topic_name, outputs=output_index_list, raw_tx=raw_hex)
@settings(max_examples=60)
def test_topic_manager_posts_raw_tx_in_body(
    topic: str, outputs: list[int], raw_tx: str
) -> None:
    """The raw_tx value is included in the POST body sent to the overlay node."""
    client = _FakeClient(_submit_response(topic, outputs))
    tm = TopicManager(base_url="http://example.com", topic=topic, client=client)
    asyncio.run(tm.submit(raw_tx))
    assert len(client.calls) == 1
    body = client.calls[0]["json"]
    assert body["rawTx"] == raw_tx


@given(topic=topic_name, outputs=output_index_list)
@settings(max_examples=40)
def test_topic_manager_posts_json_serializable_body(
    topic: str, outputs: list[int]
) -> None:
    """The body posted to the overlay node is always JSON-serializable."""
    client = _FakeClient(_submit_response(topic, outputs))
    tm = TopicManager(base_url="http://example.com", topic=topic, client=client)
    asyncio.run(tm.submit("deadbeef"))
    assert len(client.calls) == 1
    body = client.calls[0]["json"]
    # Will raise if the body is not JSON-serializable
    round_tripped = json.loads(json.dumps(body))
    assert round_tripped["rawTx"] == "deadbeef"


# ---------------------------------------------------------------------------
# LookupService properties
# ---------------------------------------------------------------------------


@given(system_id=ident, epoch_id=ident)
@settings(max_examples=60)
def test_lookup_service_returns_list_of_lookup_results(
    system_id: str, epoch_id: str
) -> None:
    """LookupService.lookup always returns a list of LookupResult objects."""
    fake_result = {
        "txid": "bb" * 32,
        "outputIndex": 0,
        "beef": None,
        "data": {"system_id": system_id, "epoch_id": epoch_id},
    }
    client = _FakeClient(_lookup_response([fake_result]))
    ls = LookupService(base_url="http://example.com", client=client)
    results = asyncio.run(ls.lookup({"system_id": system_id, "epoch_id": epoch_id}))
    assert isinstance(results, list)
    assert all(isinstance(r, LookupResult) for r in results)


@given(txid=txid_hex, output_index=st.integers(min_value=0, max_value=99))
@settings(max_examples=60)
def test_lookup_result_fields_are_correctly_typed(
    txid: str, output_index: int
) -> None:
    """LookupResult.txid is a str and output_index is an int."""
    fake_result = {
        "txid": txid,
        "outputIndex": output_index,
        "beef": None,
        "data": {},
    }
    client = _FakeClient(_lookup_response([fake_result]))
    ls = LookupService(base_url="http://example.com", client=client)
    results = asyncio.run(ls.lookup({"any": "query"}))
    assert len(results) == 1
    r = results[0]
    assert isinstance(r.txid, str)
    assert isinstance(r.output_index, int)
    assert r.txid == txid
    assert r.output_index == output_index


@given(n=st.integers(min_value=0, max_value=10))
@settings(max_examples=40)
def test_lookup_service_returns_exactly_n_results(n: int) -> None:
    """LookupService returns as many LookupResult objects as items in 'results'."""
    fake_results = [
        {"txid": f"{i:0>62}ff", "outputIndex": i, "beef": None, "data": {}}
        for i in range(n)
    ]
    client = _FakeClient(_lookup_response(fake_results))
    ls = LookupService(base_url="http://example.com", client=client)
    results = asyncio.run(ls.lookup({}))
    assert len(results) == n


@given(url=base_url)
@settings(max_examples=40)
def test_lookup_service_strips_trailing_slash_from_url(url: str) -> None:
    """LookupService strips trailing slashes from the base URL."""
    ls = LookupService(base_url=url)
    assert not ls._base.endswith("/")


# ---------------------------------------------------------------------------
# OverlayClient properties
# ---------------------------------------------------------------------------


@given(system_id=ident)
@settings(max_examples=60)
def test_overlay_client_find_epochs_always_includes_system_id(
    system_id: str,
) -> None:
    """find_epochs always includes system_id in the lookup query body."""
    client = _FakeClient(_lookup_response([]))
    oc = OverlayClient(base_url="http://example.com", client=client)
    asyncio.run(oc.find_epochs(system_id))
    assert len(client.calls) == 1
    body = client.calls[0]["json"]
    assert body["query"]["system_id"] == system_id


@given(system_id=ident, epoch_id=ident)
@settings(max_examples=60)
def test_overlay_client_find_epochs_includes_epoch_id_when_provided(
    system_id: str, epoch_id: str
) -> None:
    """When epoch_id is given, find_epochs includes it in the lookup query."""
    client = _FakeClient(_lookup_response([]))
    oc = OverlayClient(base_url="http://example.com", client=client)
    asyncio.run(oc.find_epochs(system_id, epoch_id=epoch_id))
    body = client.calls[0]["json"]
    assert body["query"]["epoch_id"] == epoch_id


@given(system_id=ident)
@settings(max_examples=60)
def test_overlay_client_find_epochs_omits_epoch_id_when_not_provided(
    system_id: str,
) -> None:
    """When epoch_id is omitted, find_epochs does NOT include it in the query."""
    client = _FakeClient(_lookup_response([]))
    oc = OverlayClient(base_url="http://example.com", client=client)
    asyncio.run(oc.find_epochs(system_id))
    body = client.calls[0]["json"]
    assert "epoch_id" not in body["query"]


@given(
    api_key=st.text(
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        min_size=8,
        max_size=64,
    ),
    topic=topic_name,
)
@settings(max_examples=40)
def test_api_key_forwarded_in_authorization_header(
    api_key: str, topic: str
) -> None:
    """TopicManager forwards api_key as 'Authorization: Bearer <key>' in every request."""
    client = _FakeClient(_submit_response(topic, [0]))
    tm = TopicManager(
        base_url="http://example.com",
        topic=topic,
        api_key=api_key,
        client=client,
    )
    asyncio.run(tm.submit("deadbeef"))
    headers = client.calls[0]["headers"]
    assert headers.get("Authorization") == f"Bearer {api_key}"


@given(topic=topic_name)
@settings(max_examples=40)
def test_no_api_key_means_no_authorization_header(topic: str) -> None:
    """When no api_key is provided, the Authorization header is absent."""
    client = _FakeClient(_submit_response(topic, []))
    tm = TopicManager(base_url="http://example.com", topic=topic, client=client)
    asyncio.run(tm.submit("deadbeef"))
    headers = client.calls[0]["headers"]
    assert "Authorization" not in headers
