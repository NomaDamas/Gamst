"""A conversation in which an LLM answers only through GAMST's MCP tools: no frame, no sample of sound reaches it.

This is a scripted MCP client. It calls ``MCPServer.call_tool`` exactly as an LLM's tool calls would arrive and prints
each call and its bounded JSON answer. **The answer sentences are a template that stands in for the LLM**: they are
filled from the tool results only, so a reader can check every number against the JSON above it.

  python docs/demo/conversation/conversation.py --clips DIR --store RUN.sqlite \
      [--profile PROFILE.json] [--words-store STORED.sqlite --words-report STORED.json]

Session 1 ("감스트가 샷건을 몇 번 쳤어? 언제?"): compose_recipe composes the recipe from a structured request (the
caller writes the movement as a rule; GAMST has no hidden LLM), starts it on the clip and waits; count_events and
get_timeline answer. This runs the pose, face and appearance models locally (run it through try-it/guard.sh).

Session 2 ("뽀또라고 몇 번 말했어?"): compose_recipe composes the word recipe but it is NOT started: its recogniser is
hosted and paid, which needs the owner's OK. count_events reads the records an earlier run of the same recipe stored
(--words-store). Without them the words session is skipped and says so.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

SHOTGUN_TEXT = '샷건: 주먹을 어깨 위로 들었다가 빠르게 아래로 내리친다'
# The rule the LLM writes for that text (gamst/providers/movement_rule.py). It is the rule measured 16/16 on this clip
# (docs/measurements/testbed-2026-11-26.md); here it stands in for what an LLM would write.
SHOTGUN_RULE = {"channels": "vision.pose_channels", "steps": [
    {"when": {"measure": {"height": ["right_wrist", "right_shoulder"]}, "op": "gt", "value": 0.0}},
    {"when": {"measure": {"rate": {"height": ["right_wrist", "right_shoulder"]}}, "op": "lt", "value": -10},
     "within_ms": 300},
    {"when": {"all": [{"measure": {"height": ["right_wrist", "right_shoulder"]}, "op": "lt", "value": -0.8},
                      {"measure": {"rate": {"height": ["right_wrist", "right_shoulder"]}}, "op": "gt", "value": -3}]},
     "within_ms": 150}], "margin": 0}
LLM = '[LLM stand-in: sentence template filled from the tool results above]'


def bounded(value, *, items=6, text=160, depth=0):
    """What is printed of an answer: lists cut to ``items`` (the rest counted), long strings cut, deep objects folded."""
    if isinstance(value, dict):
        if depth >= 5:
            return f'{{... {len(value)} keys}}'
        return {key: bounded(item, items=items, text=text, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        shown = [bounded(item, items=items, text=text, depth=depth + 1) for item in value[:items]]
        return shown + ([f'... {len(value) - items} more'] if len(value) > items else [])
    if isinstance(value, str) and len(value) > text:
        return value[:text] + f'... ({len(value)} chars)'
    return value


def composed_view(result):
    """compose_recipe's answer without the recipe body (it is saved; its modules are listed)."""
    if not isinstance(result, dict) or 'recipe' not in result:
        return result
    recipe = result['recipe']
    return dict(result, recipe=f"{{... saved: {len(recipe['modules'])} module items, {len(recipe['inputs'])} input(s), "
                               f"references {[item['name'] for item in recipe.get('references', ())]}}}")


async def call(server, name, arguments, *, shown=None, items=6, view=None):
    print(f'\n>>> LLM calls {name}')
    print(json.dumps(shown if shown is not None else arguments, ensure_ascii=False, indent=1))
    try:
        result = await server.call_tool(name, arguments)
    except Exception as error:          # an MCP tool error is an answer too
        print(f'<<< error: {error}')
        return None
    size = len(json.dumps(result, ensure_ascii=False, default=str).encode())
    print(f'<<< {name}: {size} bytes of JSON (printed bounded)')
    print(json.dumps(bounded(view(result) if view else result, items=items), ensure_ascii=False, indent=1, default=str))
    return result


def seconds(ms, start):
    return f'{(ms - start) / 1000:.2f}초'


async def shotgun(args):
    from gamst import Gamst
    from gamst.mcp import MCPServer
    print('=' * 100)
    print('USER: 감스트가 샷건을 몇 번 쳤어? 언제?')
    print('=' * 100)
    async with Gamst(consent=True, store=args.store, sensor_inference_interval_ms=1000, max_records=100000,
                     max_payload_bytes=8192 * 100000) as human:
        server = MCPServer(human, clips=[args.clips], run_profile=args.profile)
        request = dict(name='shotgun-gamst', inputs=dict(video='b.mp4'),
                       who=dict(name='감스트', person='gamst.person.json'),
                       what=dict(movement=dict(text=SHOTGUN_TEXT, rule=SHOTGUN_RULE)),
                       mode='count', timing='post', start=True, wait=True)
        composed = await call(server, 'compose_recipe', request,
                              shown=dict(request, what=dict(movement=dict(text=SHOTGUN_TEXT, rule='{... the rule, 3 steps}'))),
                              view=composed_view, items=8)
        if not composed or not composed.get('ok') or not (composed.get('report') or {}).get('ok'):
            print('\nthe run did not finish; nothing to answer from')
            return 1
        start = composed['report']['start_ms']
        counted = await call(server, 'count_events', dict(capability_id='temporal.exemplar_events', event='exemplar.matched',
                                                          who='감스트', window='1h', max_evidence=40), items=3)
        everyone = await call(server, 'count_events', dict(capability_id='temporal.exemplar_events', event='exemplar.matched',
                                                           window='1h', group_by='who', max_evidence=0))
        seen = await call(server, 'get_timeline', dict(start_ms=start, end_ms=start + 12000, step_ms=2000,
                                                       capabilities=['temporal.exemplar_events'], who='감스트',
                                                       max_bytes=8192), items=2)
        times = sorted(item['event']['end_ms'] for item in counted['evidence'])
        on_screen = (seen or {}).get('who', {}).get('seen') or []
        print('\n' + LLM)
        print(f"ANSWER: 감스트는 샷건을 {counted['count']}번 쳤어요" +
              ('' if counted.get('complete', True) else ' (잃어버린 프레임이 있어 정확하지 않을 수 있어요)') + '. '
              f"영상 시작부터 각 동작이 끝난 때는 {', '.join(seconds(t, start) for t in times)}입니다." +
              (" 사람별로는 " + ', '.join(f"{who} {count}번" for who, count in everyone['groups'].items()) + '.'
               if everyone and everyone.get('groups') else '') +
              (f" 감스트가 화면에 보인 구간: " + ', '.join(f"{seconds(item['start_ms'], start)}-{seconds(item['end_ms'], start)}"
                                                    for item in on_screen[:4]) + '.' if on_screen else ''))
    return 0


async def words(args):
    from gamst import Gamst
    from gamst.mcp import MCPServer
    print('\n' + '=' * 100)
    print('USER: 뽀또라고 몇 번 말했어?')
    print('=' * 100)
    if not args.words_store or not Path(args.words_store).is_file():
        print('\nSKIPPED: no stored word run (--words-store), and a new one would send the sound to a paid hosted '
              'recogniser, which needs the owner\'s OK.')
        return 0
    # The stored run is from 2026-11-26: a retention long enough to keep it, a copy so that the original is untouched.
    async with Gamst(consent=True, store=args.words_store, retention_hours=24 * 60, max_records=100000,
                     max_payload_bytes=8192 * 100000) as human:
        server = MCPServer(human, clips=[args.clips])
        request = dict(name='bbo-gamst', inputs=dict(audio='a.wav'),
                       who=dict(name='감스트', person='gamst.person.json', voice='voice.json'),
                       what=dict(words=['뽀또']), mode='count', timing='realtime',
                       **{'with': ['verification', 'vote', 'tse']})
        await call(server, 'compose_recipe', request, items=8, view=composed_view)
        print('\nNOT STARTED: the recipe would send the sound to hosted, paid recognisers (before_run above); that needs '
              "the owner's OK. The answer below reads the records that an earlier run of this same recipe stored "
              '(try-it/testbed/tse/runs/m1, 2026-11-26), through the same MCP tools.')
        counted = await call(server, 'count_events', dict(capability_id='speech.term_events', event='term.spoken',
                                                          window='60d', max_evidence=20), items=3)
        uncertain = await call(server, 'count_events', dict(capability_id='speech.term_events', event='term.spoken',
                                                            window='60d', where={'state': 'uncertain'}, max_evidence=0))
        start = json.loads(Path(args.words_report).read_text())['start_ms'] if args.words_report else None
        times = sorted(item['event']['start_ms'] for item in counted['evidence'])
        print('\n' + LLM)
        when = (f" 녹음 시작부터 {', '.join(seconds(t, start) for t in times)}에 말했어요"
                " (녹음 시작 시각은 그 실행의 보고서 파일에서 읽었고 MCP 도구에서 온 것이 아니에요)." if start else '')
        states = counted.get('states') or {}
        print(f"ANSWER: 감스트는 '뽀또'를 {counted['count']}번 말했어요 (확정 {states.get('final', 0)}, 잠정 "
              f"{states.get('provisional', 0)})." + when +
              (f" 목소리가 겹쳐 아직 판단하지 못한 것이 {(uncertain or {}).get('states', {}).get('uncertain', 0)}개 있고, "
               '세지 않았지만 \'말하지 않았다\'는 뜻은 아니에요.' if (uncertain or {}).get('states', {}).get('uncertain') else ''))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--clips', required=True, help="the owner's folder: b.mp4, a.wav, gamst.person.json, voice.json")
    parser.add_argument('--store', required=True, help='SQLite store of the shotgun run (written)')
    parser.add_argument('--profile', help='setup profile of the run (pose, face and appearance models)')
    parser.add_argument('--words-store', help='a COPY of a stored person-term-count run (opened read-write)')
    parser.add_argument('--words-report', help="that run's report (its start_ms anchors the times)")
    parser.add_argument('--only', choices=['shotgun', 'words'])
    args = parser.parse_args(argv)
    code = 0
    if args.only != 'words':
        code = asyncio.run(shotgun(args))
    if args.only != 'shotgun':
        code = asyncio.run(words(args)) or code
    return code


if __name__ == '__main__':
    sys.exit(main())
