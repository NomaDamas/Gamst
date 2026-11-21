import asyncio
import json

import pytest

from gamst.engine import Gamst
from gamst.mcp import MCPServer
from gamst.schema import timestamp


def run(coro):
    return asyncio.run(coro)


RISK = {"signal": "risk.score", "op": "gt", "value": 0.5, "within_ms": 60_000}


async def push(human, name, value, ts):
    await human.push_signal(name, value, "video", timestamp_ms=ts)


def test_fires_with_contributions_context_cooldown_and_rate_cap():
    async def scenario():
        async with Gamst(consent=True) as human:
            base = timestamp() - 120_000
            watch = human.watches.create(owner="user", description="위험도가 0.5를 넘으면 알려줘", rule={
                "conditions": [RISK], "cooldown_ms": 1_000, "max_per_minute": 2, "context_window_ms": 5_000})
            await push(human, "risk.score", 0.2, base)
            await push(human, "risk.score", 0.9, base + 100)    # fires
            await push(human, "risk.score", 0.9, base + 500)    # cooldown
            await push(human, "risk.score", 0.9, base + 2_000)  # fires
            await push(human, "risk.score", 0.9, base + 4_000)  # rate cap (2/min)
            await push(human, "risk.score", 0.9, base + 61_000)  # window rolled: fires
            return watch, human.watches.get_alarms(limit=10)["items"][::-1], human.watches.list()["watches"][0]

    watch, alarms, state = run(scenario())
    assert [a["fired_ms"] - alarms[0]["fired_ms"] for a in alarms] == [0, 1_900, 60_900]
    first = alarms[0]
    assert first["version"] == watch["version"] and first["recipient"] == "user"
    assert first["contributions"][0]["observed"] == 0.9
    assert first["contributions"][0]["contribution"] == 1.0
    values = [r["observation"]["value"] for r in first["context"]["records"] if r["type"] == "observation.signal"]
    assert values == [0.2, 0.9]
    assert state["stats"]["suppressed_cooldown"] == 1 and state["stats"]["suppressed_rate"] == 1


def test_weighted_threshold_records_each_evidence():
    async def scenario():
        async with Gamst(consent=True) as human:
            base = timestamp() - 10_000
            human.watches.create(owner="jev", description="person near and loud", rule={
                "threshold": 0.6, "cooldown_ms": 0,
                "conditions": [dict(RISK, weight=2.0),
                               {"signal": "audio.level", "op": "gte", "value": 70, "weight": 1.0, "within_ms": 1_000}]})
            await push(human, "audio.level", 80, base)            # alone: score 1/3
            await push(human, "risk.score", 0.8, base + 5_000)    # audio stale: score 2/3 >= .6
            return human.watches.get_alarms()["items"]

    alarms = run(scenario())
    assert len(alarms) == 1
    risk, audio = alarms[0]["contributions"]
    assert (risk["satisfied"], risk["contribution"]) == (True, pytest.approx(2 / 3, abs=1e-6))
    assert (audio["status"], audio["contribution"]) == ("stale", 0.0)
    assert alarms[0]["recipient"] == "jev"


def test_only_owner_updates_and_jev_proposals_are_not_applied():
    async def scenario():
        async with Gamst(consent=True) as human:
            watches = human.watches
            watch = watches.create(owner="user", description="risk", rule={"conditions": [RISK]})
            with pytest.raises(ValueError, match="owned by user"):
                watches.update(watch_id=watch["id"], actor="jev", expected_version=watch["version"],
                               changes={"threshold": 0.5})
            proposal = watches.propose(watch_id=watch["id"], proposer="jev",
                                       changes={"cooldown_ms": 30_000}, reason="false alarm burst")
            assert watches.list()["watches"][0]["version"] == watch["version"]
            with pytest.raises(ValueError, match="expected_version"):
                watches.update(watch_id=watch["id"], actor="user", expected_version="0" * 64,
                               changes={"cooldown_ms": 30_000})
            updated = watches.update(watch_id=watch["id"], actor="user", expected_version=watch["version"],
                                     changes=proposal["changes"], reason="accept", proposal_id=proposal["id"])
            return watch, updated, watches.list(include_audit=True)

    watch, updated, listing = run(scenario())
    assert updated["version"] != watch["version"] and updated["rule"]["cooldown_ms"] == 30_000
    assert [e["action"] for e in listing["audit"]] == ["create", "propose", "update"]
    assert listing["audit"][-1]["previous_version"] == watch["version"]
    assert listing["open_proposals"] == []


def test_mcp_tools_push_alarm_notification():
    async def scenario():
        async with Gamst(consent=True) as human:
            server = MCPServer(human)
            response = await server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                "name": "create_watch", "arguments": {"owner": "user", "description": "risk",
                                                      "rule": {"conditions": [RISK]}}}})
            assert response["result"]["isError"] is False
            await push(human, "risk.score", 0.7, timestamp() - 1_000)
            message = server.notifications.get_nowait()
            listed = await server.call_tool("list_watches", {})
            await server.close()
            return message, listed

    message, listed = run(scenario())
    assert message["method"] == "notifications/gamst/watch_alarm"
    assert message["params"]["alarm"]["contributions"][0]["observed"] == 0.7
    assert listed["watches"][0]["stats"]["fired"] == 1
    json.dumps(message, allow_nan=False)
