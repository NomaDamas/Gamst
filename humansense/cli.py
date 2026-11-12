"""Small command line interface; all demos are explicitly identified."""
import argparse
import asyncio
import json


async def demo():
    from . import HumanSense
    async with HumanSense(consent=True) as human:
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
    parser = argparse.ArgumentParser(description="HumanSense local interaction SDK")
    parser.add_argument("command", choices=["demo", "mcp", "dashboard", "websocket"])
    args, extra = parser.parse_known_args()
    if args.command == "demo":
        if extra:
            parser.error("demo accepts no extra arguments")
        asyncio.run(demo())
    else:
        import sys
        sys.argv = [sys.argv[0]] + extra
        if args.command == "mcp":
            from .mcp import main as run
        elif args.command == "dashboard":
            from .dashboard import main as run
        else:
            from .server import main as run
        run()
