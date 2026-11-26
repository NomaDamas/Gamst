"""Words by example: a speech recognizer writes down what is said in the owner's clips; the owner confirms or edits it.

``gamst recipe enroll RECIPE --kind terms --video CLIP[@start-end] --audio CLIP ...`` runs the recipe's own
transcribing module over the clips and adds one proposal (:mod:`gamst.movement_proposals`): the term and the
different ways it was written. Nothing is counted by a proposal. After ``gamst recipe proposals confirm ID``
(``ID=word,other spelling`` edits it) the same command with ``--out`` saves a terms file for ``--terms``.
Clips are read once, in memory; no audio is kept.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys

from .movement import PROPOSALS, SOURCES, parse_clip


def transcriber(recipe):
    """(the recipe's transcribing module, reading the sound input itself; that input)."""
    from . import PARTS, inputs_of, upstream
    item = next((item for item in recipe['modules'] if PARTS[item['use']].transcribes), None)
    if item is None:
        raise SystemExit(f"{recipe['name']} has no transcribing module to write down words with")
    place = next(entry for entry in inputs_of(recipe, upstream(recipe['modules'], item['name']))
                 if entry['use'] == 'microphone')
    return dict(item, consumes=[place['id']]), place


async def heard(recipe, args, clip):
    """The text recognized in one clip (finals in order); only the named part of the file is read."""
    from .. import Gamst
    from ..inputs import AudioFileInput, SoundTrackInput
    from ..revisions import record_span
    from . import build_pipeline
    item, place = transcriber(recipe)
    part = {} if clip.start is None else dict(start_s=clip.start, end_s=clip.end)
    source = (SoundTrackInput if clip.kind == 'video' else AudioFileInput)(clip.path, **part)
    run = argparse.Namespace(**dict(vars(args), offset_ms=None, input=None, **{name: None for name in SOURCES}))
    async with Gamst(consent=True) as human:
        pipeline = await build_pipeline(dict(recipe, references=[]), run, human, sources={place['id']: source},
                                        modules=[item], lanes=False)
        try:
            await pipeline.run(stop_on_failure=True)
            records = human.store.query('local-user', kind='observation.signal', limit=1000, registered_only=True,
                                        capability_id=pipeline.modules[item['name']]['spec'].id)
        except (RuntimeError, OSError, ValueError) as error:
            raise SystemExit(f'{clip.name()}: {error}') from error
        finally:
            await pipeline.close()
    finals = {}
    for record in records:
        value = record['observation']['value']
        if value.get('status') == 'final':
            finals[value.get('utterance_id') or record['id']] = (record_span(record) or (0, 0), value['text'])
    return ' '.join(text.strip() for _, text in sorted(finals.values()) if text.strip())


async def enroll_terms(recipe, args):
    from ..movement_proposals import Proposals
    clips = [parse_clip(kind, value) for kind in ('video', 'audio') for value in getattr(args, kind) or []]
    place = args.proposals or PROPOSALS
    store = Proposals(place)
    if clips:
        written = []
        for clip in clips:
            text = await heard(recipe, args, clip)
            if not text:
                raise SystemExit(f'nothing was recognized in {clip.name()}')
            written.append(dict(clip=clip.name(), heard=text[:500]))
        variants = [text for text, _ in Counter(item['heard'] for item in written).most_common()]
        item, _ = transcriber(recipe)
        try:
            proposal = store.propose_term(recipe=recipe['name'], spellings=([args.text] if args.text else []) + variants,
                                          clips=written, model=f"{item['use']}:{item.get('candidate') or 'default'}")
        except ValueError as error:
            raise SystemExit(f'{error}; name the part of each clip that is the word: FILE@start-end') from error
        print(json.dumps(proposal, ensure_ascii=False), flush=True)
        if not (args.no_confirm and args.out):
            print(f"nothing is counted yet: gamst recipe proposals confirm {proposal['id'][:12]} --proposals {place} "
                  f"(or {proposal['id'][:12]}=word,other spelling), then enroll --kind terms --out FILE", file=sys.stderr)
            return proposal
    if not args.out:
        raise SystemExit('give clips (--video/--audio FILE[@start-end]) to propose a word, or --out FILE to save confirmed ones')
    wanted = ('confirmed', 'open') if args.no_confirm else ('confirmed',)
    try:
        found = store.list(text=args.text if not clips else None, limit=512)['proposals']
    except ValueError as error:
        raise SystemExit(f'{place}: {error}') from error
    found = [item for item in found if item['kind'] == 'term' and item['recipe'] == recipe['name'] and item['status'] in wanted
             and (not args.proposal or any(item['id'].startswith(key) for key in args.proposal))]
    if not found:
        raise SystemExit(f'no confirmed word for {recipe["name"]} in {place} (gamst recipe proposals list, then confirm ID)')
    terms = {}
    for item in found:
        terms.setdefault(item['text'], [])
        terms[item['text']] += [value for value in item['spellings'] if value not in terms[item['text']]]
    saved = dict(kind='terms', format=1, recipe=recipe['name'], terms=terms, method='examples',
                 confirmed=all(item['status'] == 'confirmed' for item in found), proposals=[item['id'] for item in found])
    from .enrollment import _save
    _save(args.out, saved)
    print(json.dumps(dict(out=args.out, terms=terms, confirmed=saved['confirmed']), ensure_ascii=False), flush=True)
    return saved
