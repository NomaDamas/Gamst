import unittest
from decimal import localcontext
from fractions import Fraction
from unittest.mock import patch

from benchmarks.billing_json import decode_original, source_number


class OriginalNumbers(unittest.TestCase):
    def test_exact_lexemes_and_ambient_decimal_context(self):
        for precision in (1, 6, 28):
            with localcontext() as context:
                context.prec = precision
                data = decode_original(b'{"amount":0.123456789,"negative":-0.001,"integer":1000,"scientific":1e-9}')
                self.assertEqual(data['amount'].lexeme, '0.123456789')
                self.assertEqual(data['amount'].usd_atoms(), 123456789)
                self.assertEqual(data['negative'].usd_atoms(), -1000000)
                self.assertEqual(data['integer'].integer(), 1000)
                self.assertEqual(data['scientific'].value, Fraction(1, 10**9))

    def test_no_implicit_rounding_or_float_counter_coercion(self):
        with self.assertRaisesRegex(ValueError, 'representable'):
            source_number('0.0000000005').usd_atoms()
        for number in ('1.0', '1e0', '-1', '9223372036854775808'):
            with self.subTest(number=number), self.assertRaises(ValueError):
                source_number(number).integer()
        self.assertEqual(source_number('-1').integer(signed=True), -1)
        self.assertIs(decode_original(b'true'), True)

    def test_numeric_bounds_before_arithmetic(self):
        self.assertEqual(source_number('1e000018').value, 10**18)
        for number in ('1e9999999', '1e19', '1e-19', '9'*39, 'NaN', 'Infinity', '01', '+1'):
            with self.subTest(number=number), self.assertRaises(ValueError):
                decode_original(number.encode())

    def test_duplicate_nonfinite_invalid_and_depth_fail(self):
        for payload in (b'{"a":1,"a":2}', b'{"a":NaN}', b'"\xff"', b'[1,]', b'['*17+b'0'+b']'*17):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                decode_original(payload)
        self.assertEqual(decode_original(b'['*16+b'0'+b']'*16)[0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0].integer(), 0)

    def test_escaped_brackets_are_not_nesting_and_node_bound_precedes_decoder(self):
        self.assertEqual(decode_original(b'{"s":"[\\\"{]"}')['s'], '["{]')
        with patch('benchmarks.billing_json.MAX_NODES', 4), patch('benchmarks.billing_json.json.loads') as decoder:
            with self.assertRaisesRegex(ValueError, 'node count'):
                decode_original(b'[1,2,3,4]')
            decoder.assert_not_called()
