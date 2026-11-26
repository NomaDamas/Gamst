"""Proposed movements and words waiting for the owner: intervals of a text, a rule an agent wrote, or a term heard in clips.

A proposal never counts anything. An interval becomes a positive example only after the owner confirms it,
a negative one after the owner rejects it (``gamst recipe enroll --proposals``). A term (what a speech recognizer wrote for
example clips, with its spelling variants) becomes countable only after the owner confirms or edits it. As with Watches, a proposal
belongs to ``user``; ``jev`` (the application's model) or the finder may propose, only the owner resolves,
and must name the version that was looked at.

The file is JSON (mode 0600) and holds intervals, the text, scores and the identity of the video (path, size
and SHA256), never frames, joints, audio or text written by a VLM.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid

from .capabilities import identifier
from .movement_finder import movement_text
from .providers.movement_rule import validate_rule
from .watch import OWNERS

OWNER = 'user'
PROPOSERS = OWNERS + ('finder',)
STATUSES = ('open', 'confirmed', 'rejected')
MAX_PROPOSALS, MAX_VIDEOS, MAX_FILE = 512, 64, 4 << 20
MAX_INTERVAL_S, MAX_SPELLINGS = 30., 16
_VERSIONED = ('id', 'kind', 'recipe', 'text', 'video', 'start_s', 'end_s', 'score', 'rule', 'proposer', 'status')


def _version(item):
    keys = _VERSIONED + (('spellings',) if item.get('kind') == 'term' else ())
    data = json.dumps({key: item.get(key) for key in keys}, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(data.encode()).hexdigest()


def video_identity(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return dict(path=str(path), name=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest())


def checked_interval(start_s, end_s, duration_s):
    for value in (start_s, end_s):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('an interval is two finite numbers of seconds')
    if not 0 <= start_s < end_s or end_s - start_s > MAX_INTERVAL_S:
        raise ValueError(f'an interval is 0 <= start < end, at most {MAX_INTERVAL_S:g} s long')
    if duration_s is not None and end_s > duration_s + 1e-3:
        raise ValueError('the interval ends after the video')
    return round(float(start_s), 3), round(float(end_s), 3)


def checked_spellings(spellings, *, patterns=False):
    """Distinct written forms of one term, the first naming it; ``re:`` patterns only where the owner wrote them."""
    from .providers.term_events import normalize, check_spelling
    found = []
    for value in spellings if isinstance(spellings, (list, tuple)) else ():
        value = ' '.join(str(value).split())
        if not 1 <= len(value) <= 100:
            raise ValueError('a spelling has 1..100 characters')
        if value.startswith('re:') and not patterns:
            raise ValueError('a proposed spelling is text, not a pattern')
        check_spelling(value)
        if normalize(value) not in [normalize(other) for other in found]:
            found.append(value)
    if not 1 <= len(found) <= MAX_SPELLINGS:
        raise ValueError(f'a term has 1..{MAX_SPELLINGS} spellings')
    return found


class Proposals:
    """The proposals file. Every change is written atomically under a lock; a missing file is empty."""

    def __init__(self, path, *, clock=lambda: time.time_ns() // 1_000_000):
        self.path, self._clock = Path(path), clock

    # ------------------------------------------------------------------ file

    def _read(self):
        if not self.path.exists():
            return dict(version=1, videos={}, proposals=[])
        if self.path.stat().st_size > MAX_FILE:
            raise ValueError(f'{self.path} is too large to be a proposals file')
        data = json.loads(self.path.read_text())
        if type(data) is not dict or data.get('version') != 1 or type(data.get('videos')) is not dict \
                or type(data.get('proposals')) is not list:
            raise ValueError(f'{self.path} is not a movement proposals file')
        for item in data['proposals']:
            if type(item) is not dict or item.get('version') != _version(item) or item.get('status') not in STATUSES:
                raise ValueError(f'{self.path}: a proposal does not match its version; the file was changed by hand')
        return data

    @contextmanager
    def _edit(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = open(str(self.path) + '.lock', 'a')
        try:
            try:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX)
            except ImportError:     # no advisory locks on this platform: one writer at a time is the caller's duty
                pass
            data = self._read()
            yield data
            descriptor, name = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name)   # mode 0600
            try:
                with os.fdopen(descriptor, 'w') as out:
                    json.dump(data, out, ensure_ascii=False, indent=1)
                os.replace(name, self.path)
            except BaseException:
                Path(name).unlink(missing_ok=True)
                raise
        finally:
            lock.close()

    # ------------------------------------------------------------------ propose

    def _add(self, data, item, proposer, model=None):
        if proposer not in PROPOSERS:
            raise ValueError(f'proposer must be one of {PROPOSERS}')
        identifier(item['recipe'], 'recipe name')
        if len(data['proposals']) >= MAX_PROPOSALS:
            raise ValueError(f'at most {MAX_PROPOSALS} proposals in one file')
        item = dict(id=uuid.uuid4().hex, owner=OWNER, proposer=proposer, status='open', created_ms=self._clock(),
                    **item)
        if model is not None:
            item['model'] = str(model)[:128]
        item['version'] = _version(item)
        data['proposals'].append(item)
        return item

    def propose_intervals(self, *, recipe, text, video, candidates, duration_s=None, proposer='finder', model=None):
        """Add candidates of one video (``video_identity``); an interval already proposed for it is skipped."""
        text = movement_text(text)
        with self._edit() as data:
            key = video['sha256']
            if key not in data['videos']:
                if len(data['videos']) >= MAX_VIDEOS:
                    raise ValueError(f'at most {MAX_VIDEOS} videos in one file')
                data['videos'][key] = dict(video, duration_s=duration_s)
            known = {(item['text'], item['start_s'], item['end_s']) for item in data['proposals']
                     if item.get('video') == key}
            added, duration_s = [], data['videos'][key].get('duration_s')
            for candidate in candidates:
                start_s, end_s = checked_interval(candidate['start_s'], candidate['end_s'], duration_s)
                if (text, start_s, end_s) in known:
                    continue
                score = candidate.get('score')
                if score is not None and (type(score) not in (int, float) or not 0 <= score <= 1):
                    raise ValueError('a score is a number from 0 to 1')
                added.append(self._add(data, dict(kind='interval', recipe=recipe, text=text, video=key,
                                                  start_s=start_s, end_s=end_s, score=score), proposer, model))
            return [self._public(data, item) for item in added]

    def propose_known_video(self, *, recipe, text, video, start_s, end_s, proposer):
        """An agent's interval on a video the owner already searched, named by SHA256 or file name (never a path)."""
        videos = self._read()['videos']
        matches = [item for key, item in videos.items() if video in (key, item['name'])]
        if len(matches) != 1:
            raise ValueError('unknown video; an interval needs one of the videos already in the proposals file')
        identity = {key: matches[0][key] for key in ('path', 'name', 'bytes', 'sha256')}
        return self.propose_intervals(recipe=recipe, text=text, video=identity, proposer=proposer,
                                      candidates=[dict(start_s=start_s, end_s=end_s)])

    def propose_rule(self, *, recipe, text, rule, proposer):
        text, rule = movement_text(text), validate_rule(rule)
        with self._edit() as data:
            return self._public(data, self._add(data, dict(kind='rule', recipe=recipe, text=text, rule=rule), proposer))

    def propose_term(self, *, recipe, spellings, clips=(), proposer='finder', model=None):
        """A term as it was written for example clips (``clips``: where, and what was recognized there)."""
        spellings = checked_spellings(spellings)
        with self._edit() as data:
            item = dict(kind='term', recipe=recipe, text=movement_text(spellings[0]), spellings=spellings)
            if clips:
                item['clips'] = [dict(clip) for clip in clips][:32]
            return self._public(data, self._add(data, item, proposer, model))

    # ------------------------------------------------------------------ resolve

    def resolve(self, *, proposal_id, actor, decision, expected_version, start_s=None, end_s=None, spellings=None):
        """Confirm or reject one proposal. Only its owner may, naming the version that was looked at."""
        if actor not in OWNERS:
            raise ValueError(f'actor must be one of {OWNERS}')
        if decision not in ('confirm', 'reject'):
            raise ValueError("decision must be 'confirm' or 'reject'")
        with self._edit() as data:
            item = next((item for item in data['proposals'] if item['id'] == proposal_id), None)
            if item is None:
                raise ValueError('unknown proposal_id')
            if item['owner'] != actor:
                raise ValueError(f"proposal is owned by {item['owner']}; {actor} may only propose")
            if item['version'] != expected_version:
                raise ValueError('expected_version does not match the current proposal')
            if start_s is not None or end_s is not None:
                if item['kind'] != 'interval' or decision != 'confirm':
                    raise ValueError('only a confirmed interval can be adjusted')
                item['proposed'] = item.get('proposed') or [item['start_s'], item['end_s']]
                item['start_s'], item['end_s'] = checked_interval(start_s, end_s,
                                                           data['videos'][item['video']].get('duration_s'))
            if spellings is not None:
                if item['kind'] != 'term' or decision != 'confirm':
                    raise ValueError('only a confirmed term can be edited')
                item['proposed'] = item.get('proposed') or item['spellings']
                item['spellings'] = checked_spellings(spellings, patterns=True)
                item['text'] = movement_text(item['spellings'][0])
            item.update(status=dict(confirm='confirmed', reject='rejected')[decision], resolved_by=actor,
                        resolved_ms=self._clock())
            item['version'] = _version(item)
            return self._public(data, item)

    # ------------------------------------------------------------------ read

    def _public(self, data, item):
        """What a listing shows: the video by name and SHA256, not by path."""
        result = {key: value for key, value in item.items() if value is not None}
        if item.get('video'):
            result['video'] = dict(sha256=item['video'], name=data['videos'][item['video']]['name'])
        return result

    def list(self, *, status=None, text=None, limit=100):
        if status is not None and status not in STATUSES:
            raise ValueError(f'status must be one of {STATUSES}')
        data = self._read()
        wanted = None if text is None else movement_text(text)
        items = [item for item in data['proposals'] if status in (None, item['status'])
                 and wanted in (None, item['text'])]
        return dict(proposals=[self._public(data, item) for item in items[:limit]], total=len(items),
                    truncated=len(items) > limit)

    def selected(self, *, text=None, ids=()):
        """(proposals, videos) to enroll from: by ID, or every proposal of one text."""
        data = self._read()
        if ids:
            chosen = [item for item in data['proposals'] if item['id'] in ids]
            if len(chosen) != len(set(ids)):
                raise ValueError('unknown proposal id')
        else:
            wanted = movement_text(text)
            chosen = [item for item in data['proposals'] if item['text'] == wanted]
        return chosen, data['videos']
