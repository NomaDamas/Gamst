"""Bounded exact JSON primitives for original aggregate evidence, not invoices."""
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import json
import re

MAX_BYTES = 4 * 1024 * 1024
MAX_DEPTH = 16
MAX_NODES = 250000
_NUMBER = re.compile(r'(-?)(0|[1-9][0-9]*)(?:\.([0-9]+))?(?:[eE]([+-]?[0-9]+))?\Z')


@dataclass(frozen=True)
class SourceNumber:
    lexeme: str
    value: Fraction

    def integer(self, *, signed=False):
        if '.' in self.lexeme or 'e' in self.lexeme.lower():
            raise ValueError('integer JSON token required')
        value = self.value.numerator
        if self.value.denominator != 1 or not (-(2**63-1) if signed else 0) <= value <= 2**63-1:
            raise ValueError('bounded source integer required')
        return value

    def usd_atoms(self):
        value = self.value * 10**9
        if value.denominator != 1 or abs(value.numerator).bit_length() > 256:
            raise ValueError('USD amount is not exactly representable at frozen scale')
        return value.numerator


def source_number(lexeme):
    if type(lexeme) is not str or len(lexeme) > 64:
        raise ValueError('bounded numeric lexeme required')
    match = _NUMBER.fullmatch(lexeme)
    if match is None:
        raise ValueError('strict source JSON number required')
    integer, fraction, exponent = match.group(2), match.group(3) or '', match.group(4) or '0'
    if len(integer) + len(fraction) > 38 or len(exponent.lstrip('+-')) > 2 or abs(int(exponent)) > 18:
        raise ValueError('source decimal digits or exponent exceed bound')
    # Decimal construction/as_tuple are exact and independent of caller precision.
    sign, digits, scale = Decimal(lexeme).as_tuple()
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    coefficient *= -1 if sign else 1
    value = Fraction(coefficient * 10**max(scale, 0), 10**max(-scale, 0))
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 256:
        raise ValueError('source rational exceeds bound')
    return SourceNumber(lexeme, value)


def _preflight(text):
    """Bound expansion before the JSON decoder allocates nested containers."""
    depth = nodes = 0
    quoted = escaped = primitive = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
            primitive = False
            nodes += 1
        elif char in '[{':
            depth += 1
            nodes += 1
            primitive = False
            if depth > MAX_DEPTH:
                raise ValueError('source JSON nesting exceeds bound')
        elif char in ']}':
            depth -= 1
            primitive = False
        elif char in ',: \t\r\n':
            primitive = False
        elif not primitive:
            nodes += 1
            primitive = True
        if nodes > MAX_NODES:
            raise ValueError('source JSON node count exceeds bound')
    # Full grammar, matching brackets and escapes are validated by json.loads.


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate original JSON key')
        result[key] = value
    return result


def _nonfinite(_):
    raise ValueError('nonfinite source number')


def decode_original(payload):
    if type(payload) is not bytes or not 0 < len(payload) <= MAX_BYTES:
        raise ValueError('bounded nonempty original JSON bytes required')
    text = payload.decode('utf-8', errors='strict')
    _preflight(text)
    return json.loads(text, object_pairs_hook=_pairs, parse_int=source_number,
                      parse_float=source_number, parse_constant=_nonfinite)
