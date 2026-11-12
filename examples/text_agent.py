"""Run: python examples/text_agent.py (after installing the package)."""
import asyncio
import json
from humansense import HumanSense


async def main():
    async with HumanSense(consent=True, store="sqlite:///humansense.db") as human:
        result = await human.push_text("설명이 어려워요. 다시 설명해 주세요.")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(await human.get_state(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
