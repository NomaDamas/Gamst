import json
import unittest
from fractions import Fraction
from humansense.fill_conflict_rule import parse_claim,compare,PREDICATE,prompt


def claim(interval,status='estimated'):
    return json.dumps(dict(schema_version=1,predicate=PREDICATE,status=status,interval=interval))


class Rule(unittest.TestCase):
    def test_strict_parser_bounds_before_arbitrary_precision(self):
        value,span=parse_claim(claim([.2,.3]));self.assertEqual(span,(Fraction(1,5),Fraction(3,10)))
        self.assertIsNone(parse_claim(claim(None,'unknown'))[1])
        for raw in (claim([True,.3]),claim([-.1,.2]),claim([.3,.2]),claim([0,1],'unknown'),
            claim([0,1]).replace('"schema_version": 1','"schema_version": true'),
            claim([0,1]).replace('"schema_version": 1','"schema_version": 1.0'),
            claim([0,1])[:-1]+',"status":"estimated"}',claim([0,1]).replace('[0, 1]','[0,1e999999]'),
            claim([0,1]).replace('[0, 1]','[0,0.'+'1'*40+']'),claim([0,1]).replace('[0, 1]','[[0],1]'),
            '```json\n'+claim([0,1])+'\n```',claim([0,1])+'.',claim([0,1]).replace('[0, 1]','[0,NaN]'),
            claim([0,1]).replace('}',',"confidence":1}'),claim([0,1]).replace(PREDICATE,'volume_fraction'),'{'+('"x":['*20)):
            with self.subTest(raw=raw),self.assertRaises((ValueError,json.JSONDecodeError)):parse_claim(raw)

    def test_exact_endpoint_guard_unknown_and_degenerate(self):
        fill=dict(status='estimated',height_resolution_interval=[.2,.3])
        cases=[([.25,.4],'consistent_under_rule'),([.3,.4],'indeterminate_boundary'),
            ([.35,.4],'indeterminate_guard_band'),([.350000000001,.4],'conditional_image_model_conflict'),
            ([.25,.25],'indeterminate_boundary'),([.1,.15],'indeterminate_guard_band')]
        for span,expected in cases:self.assertEqual(compare(fill,parse_claim(claim(span))[1]),expected)
        self.assertEqual(compare(fill,None),'indeterminate_claim')
        self.assertEqual(compare(dict(status='unknown',height_resolution_interval=None),None),'indeterminate_fill')

    def test_fixed_stage_prompts_no_numeric_measurement_input(self):
        prompts=[prompt(stage,width=64,height=64,roi=[8,8,56,56],container_id='cup',coordinate_frame='pixels')
            for stage in ('initial','reinspection','strong')]
        self.assertEqual(len(set(prompts)),3)
        self.assertTrue(all(len(text.encode())<4096 for text in prompts))
        self.assertNotIn('height_resolution_interval',''.join(prompts))
