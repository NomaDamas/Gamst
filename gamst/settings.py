"""Runtime model switches shared by the dashboard and MCP."""
import inspect


def feature_status(human):
    return [{"name": p.name, "enabled": (p.plugin.enabled if p.name == "local_vlm_scene" else getattr(p, "runtime_enabled", True)),
             "modalities": list(p.modalities)} for p in human.plugins] + [{"name": "text_companion", "enabled": getattr(human, "text_companion_enabled", True), "modalities": ["text"]}]


async def configure_feature(human, name, enabled):
    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean")
    if not human.paused:
        raise ValueError("Stop capture before changing model switches")
    if name == "text_companion":
        human.text_companion_enabled = enabled
        return feature_status(human)
    plugin = next((p for p in human.plugins if p.name == name), None)
    if plugin is None:
        raise ValueError("Unknown installed model")
    if name == "local_vlm_scene":
        from .realtime import configure_scene
        await configure_scene(human, {"enabled": enabled})
        return feature_status(human)
    # Gate before resetting queued/native work; never reload model weights.
    plugin.runtime_enabled = False
    plugin.runtime_generation = getattr(plugin, "runtime_generation", 0) + 1
    reset = getattr(plugin, "reset", None)
    if reset:
        result = reset()
        if inspect.isawaitable(result):
            await result
    plugin.runtime_enabled = enabled
    return feature_status(human)
