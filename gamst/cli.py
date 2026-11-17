"""Small command line interface; all demos are explicitly identified."""
import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys


_CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_COMMANDS = {
    "install": _CHECKOUT_ROOT / "scripts" / "local_models.py",
    "run": _CHECKOUT_ROOT / "scripts" / "run_local.py",
}


def _dispatch_local_command(command, extra):
    """Run an explicitly requested checkout-owned local command.

    ``gamst run`` never installs or downloads anything implicitly.  The
    installer is a separate, explicit command and both commands fail with a
    useful message when the source checkout helpers are not present (for
    example, in a wheel that intentionally omits the development scripts).
    """
    script = _LOCAL_COMMANDS[command]
    if not script.is_file():
        raise SystemExit(
            f"gamst {command} requires the checkout helper {script}; "
            "use the source checkout or run the documented setup command."
        )
    forwarded = list(extra)
    # The approved user-facing form is ``gamst install``.  The underlying
    # script also exposes status/remove/profile actions, so preserve an
    # explicit action while defaulting the bare command to its install action.
    if command == "install" and (not forwarded or forwarded[0].startswith("-")):
        forwarded.insert(0, "install")
    try:
        result = subprocess.run(
            [sys.executable, str(script), *forwarded],
            cwd=str(_CHECKOUT_ROOT),
            check=False,
        )
    except OSError as exc:
        raise SystemExit(f"unable to start gamst {command}: {exc}") from exc
    if result.returncode:
        raise SystemExit(result.returncode)
    return result.returncode


async def demo():
    from . import Gamst
    async with Gamst(consent=True) as human:
        await human.push_text("Could you explain that again?", subject_id="demo-user")
        result = await human.push_text("Could you explain that again?", subject_id="demo-user")
        await human.push_claim("preferred_language", "English", subject_id="demo-user", scope="demo")
        conflict = await human.push_claim("preferred_language", "Korean", subject_id="demo-user", scope="demo")
        from .schema import timestamp
        now = timestamp()
        for index in range(5):
            await human.push_signal("demo.activity", 2.0, "video", subject_id="demo-user",
                                    timestamp_ms=now - 60_000 + index * 100, model_id="synthetic-demo", model_version="1")
        await human.push_signal("demo.activity", 3.0, "video", subject_id="demo-user",
                                timestamp_ms=now, model_id="synthetic-demo", model_version="1")
        trends = await human.get_trends(subject_id="demo-user", window="30s")
        print(json.dumps({"demo": True, "synthetic_behavior": trends, "conversation": result, "consistency": conflict,
                          "context": await human.get_context()}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="gamst real-time sensory context SDK")
    parser.add_argument("command", choices=["demo", "mcp", "dashboard", "websocket", "install", "run"])
    args, extra = parser.parse_known_args()
    if args.command == "demo":
        if extra:
            parser.error("demo accepts no extra arguments")
        asyncio.run(demo())
    elif args.command in _LOCAL_COMMANDS:
        _dispatch_local_command(args.command, extra)
    else:
        sys.argv = [sys.argv[0]] + extra
        if args.command == "mcp":
            from .mcp import main as run
        elif args.command == "dashboard":
            from .dashboard import main as run
        else:
            from .server import main as run
        run()
