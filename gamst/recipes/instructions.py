"""What a running recipe looks for, changed while it runs: by an LLM over MCP, or by the application (``Gamst.set_instruction``).

An instruction is the recipe's reference of kind ``terms`` (words) or ``exemplar`` (a movement or pose). It is given
as words, as a rule, as a file the owner enrolled, as proposals the owner confirmed, or as example clips. Clips and
files are only read from the folders the owner named at the start (``--clips DIR``); clips go through the same
enrollment as ``gamst recipe enroll``, in a thread of their own, so that capture does not wait for it.

A change takes effect at a stated capture time (:meth:`gamst.pipeline.Pipeline.set_reference`). Persons and voices are
the owner's: they are enrolled with consent before the run and cannot be set here. Who set an instruction (``actor``) is
what the caller says of itself, as with Watches and proposals; GAMST does not verify it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import threading

from ..movement_finder import movement_text
from ..movement_proposals import Proposals, checked_spellings
from ..pipeline import INSTRUCTABLE
from ..plugins import InputFrame
from ..redact import public_text
from ..watch import OWNERS

MAX_TERMS, MAX_IDS, MAX_FILE, MAX_CLIP_FILE = 64, 32, 4 << 20, 2 << 30
MAX_LISTED, MAX_WALKED, MAX_FOLDERS = 200, 2000, 8
ENROLL_S = 300.
PERSONAL = ('voice', 'person', 'face', 'body')
COUNTERS = dict(terms='terms', exemplar='exemplar')     # part -> the kind of reference it reads
VIDEO, SOUND, SAMPLES = ('.mp4', '.mov', '.m4v', '.mkv', '.webm', '.avi'), ('.wav',), ('.csv', '.jsonl', '.ndjson')
SOURCES = ('terms', 'file', 'proposals', 'rule', 'clips')


def _media(head):
    """What a file's first bytes say it is: a container with its own samples, never a list of places to read from."""
    return head[4:8] in (b'ftyp', b'moov', b'mdat', b'free', b'wide', b'skip') or head[:4] in (b'RIFF', b'\x1aE\xdf\xa3')


class Instructions:
    """The instructions of one running recipe (``human.instructions``)."""

    def __init__(self, recipe, args, pipeline, *, raw=None, folders=(), proposals=None, on_capability=None):
        """``raw``: the whole recipe, with what was left out; ``on_capability(id)``: a counting module was added."""
        from . import read_recipe
        self.recipe, self.args, self.pipeline = recipe, args, pipeline
        self.raw = raw or read_recipe(getattr(args, 'recipe', None) or recipe['name'])[1]
        if len(folders) > MAX_FOLDERS:
            raise SystemExit(f'--clips: at most {MAX_FOLDERS} folders')
        self.folders = []
        for folder in folders:
            path = Path(folder).expanduser()
            if not path.is_dir():
                raise SystemExit(f'--clips {folder}: not a folder')
            self.folders.append(path.resolve(strict=True))
        self.proposals, self.on_capability, self._busy = proposals, on_capability, None
        self.modules = list(pipeline.recipe_context['modules'])

    # ------------------------------------------------------------------ what there is

    def _references(self, kind=None):
        """``{name: kind}`` of the recipe's words and movements, also of parts that were left out."""
        return {item['name']: item['kind'] for item in self.raw.get('references', ())
                if item['kind'] in INSTRUCTABLE and kind in (None, item['kind'])}

    def _name(self, kind, reference):
        if kind not in INSTRUCTABLE:
            raise PermissionError(f"{kind}: only {' and '.join(INSTRUCTABLE)} are set while a recipe runs; a person or a voice is "
                                  'enrolled by the owner, with consent, before the run')
        names = self._references(kind)
        if reference is None and len(names) == 1:
            return next(iter(names))
        if reference not in names:
            raise ValueError(f"{self.raw['name']} has no {kind} reference {reference or ''}".rstrip() + f"; it has: {', '.join(names) or 'none'}")
        return reference

    def _counter(self, name, kind):
        return next((item for item in self.raw['modules'] if COUNTERS.get(item['use']) == kind
                     and item.get('reference', item['use']) == name), None)

    async def list(self):
        pipeline, shown = self.pipeline, []
        for name, kind in self._references().items():
            reference, readers = pipeline.references.get(name), pipeline._reading(name)
            entry = dict(reference=name, kind=kind)
            if reference is None or 'revision' not in reference:
                missing = self._unopened(name, kind)
                entry.update(state='left_out', **({'needs_input': missing} if missing else {}))
            else:
                states = [item['module'].status() for item in readers.values()]
                entry.update(
                    state='removed' if reference.get('removed') else 'set', label=str(reference['label'])[:200],
                    reference_id=reference['id'], reference_revision=reference['revision'], set_by=reference['set_by'],
                    source=reference['source'], set_at_ms=reference['set_at_ms'], effective_from_ms=reference['from_ms'],
                    complete_from_ms=max([reference['from_ms'], *(item for item in (
                        pipeline._complete_from(key) for key in readers) if item is not None)]),
                    applied=all(state.get('from_ms') == reference['from_ms'] for state in states) if kind == 'exemplar' else True,
                    modules=sorted(readers), capabilities=sorted(filter(None, (pipeline._capability(item) for item in readers.values()))))
                if reference['value'] is not None and kind == 'terms':
                    from ..providers.term_events import parse_terms
                    words, origin = parse_terms(reference['value'])
                    entry.update(terms={term[:100]: spellings[:16] for term, spellings in list(words.items())[:MAX_TERMS]},
                                 confirmed=origin['confirmed'])
                elif isinstance(reference['value'], dict):
                    entry.update({key: reference['value'][key] for key in ('mode', 'method', 'confirmed', 'text', 'enter', 'exit')
                                  if key in reference['value']})
            shown.append(entry)
        changes = [dict(change, known_ms=known) for known, change in pipeline.human.instruction_changes()][-32:]
        return dict(recipe=self.raw['name'], instructions=shown, changes=changes, preparing=self._busy is not None,
                    owner_only=sorted({item['kind'] for item in self.raw.get('references', ()) if item['kind'] in PERSONAL}),
                    clips=await asyncio.to_thread(self._files), proposals=self.proposals is not None,
                    actor='set_by is what each caller said of itself; GAMST does not verify it')

    def _files(self):
        found, walked = [], 0
        for folder in self.folders:
            for place, names, files in os.walk(folder):
                names.sort()
                for file in sorted(files):
                    walked += 1
                    if walked > MAX_WALKED or len(found) >= MAX_LISTED:
                        return dict(files=found, truncated=True)
                    if Path(file).suffix.lower() in (*VIDEO, *SOUND, *SAMPLES, '.json'):
                        path = Path(place) / file
                        found.append(dict(file=str(path.relative_to(folder)), folder=folder.name))
        return dict(files=found, truncated=False)

    # ------------------------------------------------------------------ files of the owner's folders

    def _inside(self, file, suffixes, limit):
        """The one regular file of that name in the owner's folders, links resolved; anything else is refused."""
        if not self.folders:
            raise ValueError('no folder to read from: the owner starts the run with --clips DIR')
        if not isinstance(file, str) or not 1 <= len(file) <= 1024 or '://' in file or '\0' in file:
            raise ValueError('a file is named by its path inside a --clips folder')
        found = set()
        for folder in self.folders:
            try:
                path = (folder / file).resolve(strict=True)
                if path.is_relative_to(folder) and path.is_file():
                    found.add(path)
            except (OSError, RuntimeError):
                continue
        if len(found) != 1:
            raise ValueError(f"{file}: {'in more than one' if found else 'no such file in a'} --clips folder")
        path = found.pop()
        if path.suffix.lower() not in suffixes:
            raise ValueError(f"{file}: give one of {', '.join(suffixes)}")
        if path.stat().st_size > limit:
            raise ValueError(f'{file} is larger than {limit >> 20} MB')
        return path

    def _clip(self, item, place):
        from .movement import MAX_CLIP_S, Clip
        if type(item) is not dict or set(item) - {'file', 'start_s', 'end_s'} or ('start_s' in item) != ('end_s' in item):
            raise ValueError('a clip is {"file", and start_s with end_s if only a part of it is the movement}')
        suffixes = SAMPLES if place['use'] == 'sensor' else (*SOUND, *VIDEO) if place['use'] == 'microphone' else VIDEO
        path = self._inside(item.get('file'), suffixes, MAX_CLIP_FILE)
        if path.suffix.lower() not in SAMPLES:
            with open(path, 'rb') as source:
                if not _media(source.read(16)):
                    raise ValueError(f"{item['file']} is not a media file")
        kind = 'sensor' if place['use'] == 'sensor' else 'audio' if path.suffix.lower() in SOUND else 'video'
        start, end = item.get('start_s'), item.get('end_s')
        if start is not None and not (type(start) in (int, float) and type(end) in (int, float) and 0 <= start < end
                                      and end - start <= MAX_CLIP_S):
            raise ValueError(f'{item["file"]}: 0 <= start_s < end_s, at most {MAX_CLIP_S:g} s')
        return Clip(kind, str(path), None if start is None else float(start), None if end is None else float(end))

    # ------------------------------------------------------------------ enrollment

    async def _enrolled(self, job):
        """``job()`` in a thread of its own (with its own event loop and model worker): capture does not wait for it."""
        loop = asyncio.get_running_loop()
        done = loop.create_future()
        done.add_done_callback(lambda future: future.cancelled() or future.exception())     # also when nobody waits any more

        def finish(value, error):
            self._busy = None
            if not done.done():
                done.set_exception(error) if error is not None else done.set_result(value)

        def work():
            value = error = None
            try:
                value = job()
            except (Exception, SystemExit) as problem:
                error = ValueError(public_text(problem))
            try:
                loop.call_soon_threadsafe(finish, value, error)
            except RuntimeError:        # the run ended meanwhile
                pass

        self._busy = threading.Thread(target=work, daemon=True, name='gamst-enroll')
        self._busy.start()
        try:
            return await asyncio.wait_for(asyncio.shield(done), ENROLL_S)
        except asyncio.TimeoutError:
            raise ValueError(f'the clips were not read within {ENROLL_S:g} s; nothing was changed') from None

    def _enroll(self, **options):
        from . import SOURCES as GIVEN
        from .movement import enroll
        given = {**vars(self.args), **{name: None for name in GIVEN}, **dict(
            offset_ms=None, input=None, text=None, name=None, rule=None, negative=[], proposals=None, proposal=[], no_confirm=False,
            mode=None, start=None, seconds=None, capture_fps=None, out=None, clips=None), **options}
        return asyncio.run(enroll(self.raw, argparse.Namespace(**given)))

    async def _movement(self, *, file, proposals, rule, clips, negatives, mode, text, actor):
        from .movement import _rule, exemplar_input
        from ..providers.movement_match import MAX_EXAMPLES
        text = None if text is None else movement_text(text)
        if rule is not None:
            if text is None:
                raise ValueError('a rule needs the text it stands for')
            return _rule(self.raw, argparse.Namespace(name=None, text=text, out=None), rule, confirmed=actor == OWNERS[0])
        if proposals is not None:
            chosen, videos = await asyncio.to_thread(self._proposals().selected, ids=proposals)
            for item in chosen:
                if item['kind'] == 'interval':      # a clip like any other: only from the owner's folders
                    self._inside(videos[item['video']]['path'], VIDEO, MAX_CLIP_FILE)
            return await self._enrolled(lambda: self._enroll(proposals=self.proposals, proposal=proposals, text=text, mode=mode))
        place = exemplar_input(self.raw)
        groups = [clips, negatives or []]
        if not clips or any(type(group) is not list or len(group) > MAX_EXAMPLES for group in groups):
            raise ValueError(f'1..{MAX_EXAMPLES} clips, and at most as many negatives')
        found = [[self._clip(item, place) for item in group] for group in groups]
        kinds = {clip.kind for group in found for clip in group}
        if len(kinds) > 1:
            raise ValueError('clips of one movement are all video files or all sound files')
        saved = await self._enrolled(lambda: self._enroll(clips=found, text=text, name=text, mode=mode))
        return dict(saved, confirmed=saved['confirmed'] and actor == OWNERS[0])

    def _proposals(self):
        if self.proposals is None:
            raise ValueError('no proposals file; the owner starts the run with --proposals FILE')
        return Proposals(self.proposals)

    def _saved(self, kind, file):
        from . import _reference_value
        path = self._inside(file, ('.json',), MAX_FILE)
        saved = json.loads(path.read_text())
        if not isinstance(saved, dict) or saved.get('kind') != kind:
            raise ValueError(f'{file} is not an enrolled {kind} file')
        return _reference_value(dict(kind=kind, name=kind), argparse.Namespace(exemplar=str(path), terms=str(path)))[0]

    async def _words(self, *, terms, proposals, actor, patterns):
        if proposals is not None:
            chosen, _ = await asyncio.to_thread(self._proposals().selected, ids=proposals)
            if any(item['kind'] != 'term' or item['status'] != 'confirmed' for item in chosen):
                raise ValueError('only words the owner confirmed (kind term) can be counted')
            found = {}
            for item in chosen:
                found.setdefault(item['text'], [])
                found[item['text']] += [value for value in item['spellings'] if value not in found[item['text']]]
            return dict(kind='terms', terms=found, method='examples', confirmed=True)
        if not isinstance(terms, (list, dict)) or not 1 <= len(terms) <= MAX_TERMS:
            raise ValueError(f'terms: 1..{MAX_TERMS} words, or {{word: [spellings]}}')
        items = terms.items() if isinstance(terms, dict) else ((term, [term]) for term in terms)
        mapping = {}
        for term, spellings in items:
            term = ' '.join(str(term).split())
            if not 1 <= len(term) <= 100 or not isinstance(spellings, list):
                raise ValueError('a word has 1..100 characters and a list of spellings')
            mapping[term] = checked_spellings(spellings or [term], patterns=patterns)
        return dict(kind='terms', terms=mapping, method='text', confirmed=actor == OWNERS[0])

    # ------------------------------------------------------------------ changes

    async def set(self, kind, *, reference=None, terms=None, file=None, proposals=None, rule=None, clips=None, negatives=None,
                  mode=None, text=None, actor=OWNERS[0], patterns=False):
        """Set or replace the words or the movement of ``reference``: exactly one of ``terms``, ``file``, ``proposals``
        (IDs the owner confirmed), ``rule`` (with ``text``) or ``clips`` (with ``negatives``, ``mode``, ``text``)."""
        if actor not in OWNERS:
            raise ValueError(f'actor must be one of {OWNERS}')
        name = self._name(kind, reference)
        given = dict(terms=terms, file=file, proposals=proposals, rule=rule, clips=clips)
        source = [key for key in SOURCES if given[key] is not None]
        allowed = ('terms', 'file', 'proposals') if kind == 'terms' else SOURCES[1:]
        if len(source) != 1 or source[0] not in allowed or negatives and source != ['clips']:
            raise ValueError(f"{kind}: give exactly one of {', '.join(allowed)}" + (' (negatives go with clips)' if negatives else ''))
        if proposals is not None and (type(proposals) is not list or not 1 <= len(proposals) <= MAX_IDS):
            raise ValueError(f'proposals: 1..{MAX_IDS} IDs')
        if mode not in (None, 'hold', 'sequence'):
            raise ValueError("mode is 'hold' or 'sequence'")
        if self._busy is not None:
            raise ValueError('another instruction is being prepared; try again when list_instructions says it is not')
        self.pipeline._instructable()
        try:
            if file is not None:
                value = await asyncio.to_thread(self._saved, kind, file)
            elif kind == 'terms':
                value = await self._words(terms=terms, proposals=proposals, actor=actor, patterns=patterns)
            else:
                value = await self._movement(file=file, proposals=proposals, rule=rule, clips=clips, negatives=negatives,
                                             mode=mode, text=text, actor=actor)
            label = value.get('label') if isinstance(value, dict) else None
            label = label or text or (', '.join(value['terms'])[:100] if kind == 'terms' and isinstance(value, dict) else None)
            result = await self._apply(name, kind, value, label=label or name, set_by=actor, source=source[0])
        except SystemExit as error:
            raise ValueError(str(error)) from None
        return dict(result, confirmed=value.get('confirmed', True) if isinstance(value, dict) else True)

    async def remove(self, reference, *, actor=OWNERS[0]):
        if actor not in OWNERS:
            raise ValueError(f'actor must be one of {OWNERS}')
        return await self.pipeline.remove_reference(reference, set_by=actor)

    async def _apply(self, name, kind, value, **about):
        if self.pipeline._reading(name):
            return await self.pipeline.set_reference(name, kind, value, **about)
        self._busy = 'building'     # one at a time: a worker that starts takes seconds
        try:
            items, references, dropped = self._plan(name, kind)
            added, built = [], []
            try:
                from . import build_module, module_options
                context = dict(self.pipeline.recipe_context, modules=[*self.modules, *items],
                               recipe=dict(self.recipe, inputs=self.raw['inputs']))
                for reference, owned, consent in references:
                    self.pipeline.add_reference(reference['name'], reference['kind'], owned, label=reference.get('label'), consent=consent)
                for item in items:
                    made = await build_module(item, context)
                    built.append(made['module'])
                    added.append((item['name'], made['module'], module_options(item, made)))
                result = await self.pipeline.set_reference(name, kind, value, add=added, **about)
            except BaseException:
                for reference, _, _ in references:
                    self.pipeline.references.pop(reference['name'], None)
                for module in built:
                    close = getattr(module, 'close', None)
                    closed = close() if callable(close) else None
                    if hasattr(closed, '__await__'):
                        await closed
                raise
        finally:
            self._busy = None
        self.modules += items
        names = {item['name'] for item in items}
        for item in self.raw.get('records', ()):
            wanted = item.get('for') if isinstance(item, dict) else None
            if wanted is not None and names & set([wanted] if isinstance(wanted, str) else wanted) and self.on_capability:
                self.on_capability(item['id'])
        return dict(result, added_modules=sorted(names), **({'not_limited_to': dropped} if dropped else {}))

    def _unopened(self, name, kind):
        """The inputs a counting module would need that this run has not opened."""
        from . import inputs_of, upstream
        item = self._counter(name, kind)
        chain = upstream(self.raw['modules'], item['name']) if item is not None else []
        return [entry['id'] for entry in inputs_of(self.raw, chain) if entry['id'] not in self.pipeline.inputs]

    def _plan(self, name, kind):
        """(modules to add, references to a person they need with consent, what they are then not limited to)."""
        from . import _listed, _reference_value, choose, inputs_of, upstream
        target = self._counter(name, kind)
        if target is None:
            raise ValueError(f"{self.raw['name']} has no module that counts {kind}")
        chain = upstream(self.raw['modules'], target['name'])
        personal = {item['name']: item for item in self.raw.get('references', ()) if item['kind'] in PERSONAL}
        left_out = set(getattr(self.args, 'exclude', None) or ()) - set(_listed(target.get('parts')))
        exclude, references, dropped = set(), {}, set()
        for item in chain:
            reference = personal.get(item.get('reference'))
            if reference is None or reference['name'] in self.pipeline.references or reference['name'] in references:
                continue
            if not left_out & {item['name'], *_listed(item.get('parts'))}:
                try:        # the owner gave it (with consent) when the run was started
                    references[reference['name']] = (reference, *_reference_value(reference, self.args))
                    continue
                except SystemExit:
                    pass
            if item.get('optional'):
                exclude.add(item['name'])
                dropped.add(reference['name'])
            else:
                raise ValueError(f"{item['name']} needs an enrolled {reference['kind']}: the owner gives it before the run")
        # What reads a module that is left out goes with it (a speaker check, a separated voice).
        chosen = choose(dict(self.raw, modules=chain), exclude=sorted(exclude))
        missing = [entry['id'] for entry in inputs_of(dict(self.raw, modules=chosen['modules']), chosen['modules'])
                   if entry['id'] not in self.pipeline.inputs]
        if missing:
            options = {entry['id']: entry['sources'] for entry in self.raw['inputs']}
            raise ValueError('; '.join(f"{key} is not open in this run, and no tool opens a device: the owner starts it with "
                                       + ' or '.join(f'--{option}' for option in options[key]) for key in missing))
        items = [item for item in chosen['modules'] if item['name'] not in self.pipeline.modules and not item.get('in')]
        return items, list(references.values()), sorted(dropped)

    # ------------------------------------------------------------------ after the fact

    async def search(self, reference, *, window='1h', max_events=50):
        """Look for the current instruction in records retained from before it took effect. No model runs, nothing is
        stored; the results are separate from the live counts."""
        from ..counting import GAP_CAPABILITY, gaps_of, lost
        from ..engine import _window_ms, timestamp
        pipeline = self.pipeline
        wanted, readers = pipeline.references.get(reference), pipeline._reading(reference)
        if wanted is None or 'revision' not in wanted or wanted.get('removed') or not readers:
            raise ValueError(f'{reference!r} is not set')
        if not wanted['from_ms']:
            raise ValueError(f'{reference!r} has been looked for since the start; nothing earlier was captured in this run')
        if type(max_events) is not int or not 1 <= max_events <= 100:
            raise ValueError('max_events must be 1..100')
        entry = next(iter(readers.values()))
        module = entry['module'].offline(wanted)
        sources = sorted({output or pipeline._capability(pipeline.modules[source]) for source, output in entry['links']
                          if source in pipeline.modules} - {None})
        store, now = pipeline.human.store, timestamp()
        start, end, known = max(0, now - _window_ms(window)), wanted['from_ms'] - 1, store.known_now()
        rows, limit = [], 20_000
        for step in store.scan('local-user', start, end, known_until=known, any_of=sources) if sources else ():
            rows += step
            if len(rows) > limit:
                break
            await asyncio.sleep(0)
        gone = store.superseded('local-user')
        gaps = [record for record in store.query('local-user', start, end, 'observation.signal', limit=1000, registered_only=True,
                                                 capability_id=GAP_CAPABILITY) if any(
            gaps_of([record], source) for source in sources)]
        result = await asyncio.to_thread(_search, module, rows[:limit], gone, max_events)
        removed = store.lost_until_ms
        return dict(result, retrospective=True, in_live_counts=False, reference=reference, reference_id=wanted['id'],
                    reference_revision=wanted['revision'], searched=sources, start_ms=start, end_ms=end,
                    lost=lost(gaps), removed_until_ms=removed, truncated=len(rows) > limit,
                    complete=bool(result['records']) and not gaps and len(rows) <= limit and not (removed and removed >= start),
                    note='found in records that were kept, without running a model; what was not kept was not searched')


def _search(module, rows, gone, max_events):
    """Run a counter over kept records, oldest first (a thread: no Store, no runtime)."""
    from ..timeline import rest
    events, count, seen, first, last = [], 0, 0, None, None

    def keep(values, record_id):
        nonlocal count
        for value in values if isinstance(values, list) else [values] if values is not None else []:
            for event in value['events']:
                ended = str(event.get('event')).endswith('.ended')
                count += not ended
                if len(events) < max_events:
                    events.append(dict(event, retrospective=True, record_id=record_id))

    identifier = None
    for payload, _ in reversed(rows):
        rest()
        record = json.loads(payload)
        if record['id'] in gone:
            continue
        observation, identifier = record['observation'], record['id']
        seen, first, last = seen + 1, first or record['timestamp_ms'], record['timestamp_ms']
        keep(module.process(InputFrame(observation['value'], observation.get('modality'), record['timestamp_ms'],
                                       subject_id=record['subject_id'])), identifier)
    finish = getattr(module, 'finish', None)
    keep(finish() if callable(finish) else None, identifier)
    return dict(count=count, events=events, events_truncated=count > len(events), records=seen,
                retained_from_ms=first, retained_until_ms=last)
