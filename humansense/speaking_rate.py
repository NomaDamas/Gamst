"""Bounded orthographic rate estimates over accepted ASR capture intervals.

No acoustic syllable detection, pause removal, word alignment or new inference.
The transcript reducer calls this on its latest validated full replacement only.
"""
import unicodedata

METHOD = 'asr_text_interval_v1'
MIN_DURATION_MS = 1000
_CONNECTORS = "'-’"


def _latin_letter(char):
    return unicodedata.category(char).startswith('L') and 'LATIN' in unicodedata.name(char, '')


def _count(text, language):
    """Count only declared script; ambiguous rendered numbers are not spoken words."""
    if any(unicodedata.category(char).startswith('N') for char in text):
        return None, 'numeric_text'
    letters = (lambda char: '\uac00' <= char <= '\ud7a3') if language == 'ko' else _latin_letter
    for char in text:
        if not (letters(char) or char.isspace() or unicodedata.category(char).startswith('P')):
            return None, 'unsupported_script_or_symbol'
    if language == 'ko':
        return sum(letters(char) for char in text), None
    # Apostrophes/hyphens connect Latin runs only between letters. Other
    # punctuation/whitespace separates words; dangling punctuation is ignored.
    count, in_word = 0, False
    for index, char in enumerate(text):
        if letters(char):
            if not in_word:
                count += 1
            in_word = True
        elif char in _CONNECTORS and in_word and index + 1 < len(text) and letters(text[index + 1]):
            continue
        else:
            in_word = False
    return count, None


def estimate_rate(update, status):
    """Pure projection of one accepted update; never accumulates revisions.

    status is the reducer's current partial/final/interrupted status. Bounds are
    repeated here so direct use cannot allocate or divide on arbitrary input.
    """
    if type(update) is not dict or status not in ('partial', 'final', 'interrupted'):
        raise ValueError('accepted transcript update and reducer status required')
    text, language = update.get('text'), update.get('language')
    start, end, revision = update.get('start_ms'), update.get('end_ms'), update.get('revision')
    if (type(text) is not str or len(text) > 4096 or type(language) is not str or len(language) > 32 or
            type(start) is not int or type(end) is not int or not 0 <= start <= end or end.bit_length() > 1024 or
            end - start > 300000 or type(revision) is not int or not 0 <= revision <= 2**31-1 or
            update.get('status') not in ('partial', 'final') or
            status != 'interrupted' and update['status'] != status):
        raise ValueError('invalid bounded transcript rate input')
    duration = end - start
    result = dict(method=METHOD, unicode_version=unicodedata.unidata_version, status='unavailable', reason=None,
                  interval_ms=[start, end], duration_ms=duration, revision=revision,
                  count=None, value=None, unit=None, pause_time_included=True,
                  acoustic_rate_verified=False)
    if status == 'interrupted':
        result['reason'] = 'interrupted_utterance'
        return result
    # Accept a primary language or one region tag, not arbitrary prefix matches.
    parts = language.lower().split('-')
    primary = parts[0]
    if (primary not in ('en', 'ko') or len(parts) > 2 or
            len(parts) == 2 and not (len(parts[1]) == 2 and parts[1].isascii() and parts[1].isalpha())):
        result['reason'] = 'unsupported_language'
        return result
    result['unit'] = 'orthographic_words_per_minute' if primary == 'en' else 'hangul_blocks_per_second'
    count, error = _count(unicodedata.normalize('NFC', text), primary)
    result['count'] = count
    if error:
        result['reason'] = error
    elif count == 0:
        result['reason'] = 'no_lexical_text'
    elif duration < MIN_DURATION_MS:
        result['reason'] = 'interval_under_1000ms'
    else:
        result['status'] = 'provisional' if status == 'partial' else 'estimated'
        result['value'] = count * (60000 if primary == 'en' else 1000) / duration
    return result
