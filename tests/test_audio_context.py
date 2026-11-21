import asyncio
import json

from gamst.capabilities import SensorMetadata
from gamst.engine import Gamst
from gamst.mcp import MCPServer
from gamst.plugins import InputFrame
from gamst.providers.audio_context import AudioContext


def test_audio_context_derives_silence_barge_in_and_double_talk():
    provider = AudioContext()
    sensor = SensorMetadata(source_id="mic", sensor_id="audio", sequence=0, channels=("microphone",))
    first = provider.process(InputFrame({"speech_active": True, "assistant_playing": True,
                                         "sound_events": [{"label": "door", "score": .8}],
                                         "prosody": {"f0_hz": 180}}, "audio", 1000, sensor=sensor))
    second = provider.process(InputFrame({"speech_active": False, "assistant_playing": False}, "audio", 1200,
                                         sensor=SensorMetadata(source_id="mic", sensor_id="audio", sequence=1, channels=("microphone",))))
    assert first["context"]["barge_in"] is True
    assert first["context"]["double_talk"] is True
    assert first["context"]["sound_events"][0]["label"] == "door"
    assert second["context"]["silence_ms"] == 200
    assert second["context"]["barge_in"] is False


def test_mcp_voice_backend_is_disabled_and_network_free():
    async def run():
        async with Gamst(consent=True) as human:
            server = MCPServer(human)
            status = await server.call_tool("get_voice_backend_status")
            configured = await server.call_tool("configure_voice_backend", {
                "enabled": True, "provider": "custom", "endpoint": "https://voice.invalid/realtime",
                "model": "test", "api_key_env": "VOICE_KEY", "context_only": True})
            return status, configured
    status, configured = asyncio.run(run())
    assert status == {"enabled": False, "provider": "none", "model": None,
                      "context_only": True, "network_started": False}
    assert configured["enabled"] is True and configured["network_started"] is False
    json.dumps(configured)
