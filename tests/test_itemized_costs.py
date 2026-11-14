from decimal import Decimal, localcontext
import unittest

from benchmarks.perception_comparison import compare, plan_digest
from test_perception_comparison import fixture


def rate(provider, category, value, unit='atomic-unit'):
    return dict(provider=provider,revision='frozen-v1',unit=unit,category=category,per_unit=value)


def itemized():
    plan,rows=fixture()
    plan['cost']=dict(mode='itemized',currency='USD',scope='full_pipeline',
        accounting_protocol='Synthetic all-call fixture; no real measurement',rate_card={
            'small.input':rate('small','input_tokens',.000001,'uncached-token'),
            'strong.input':rate('strong','input_tokens',.00001,'uncached-token'),
            'cache.read':rate('strong','cache_read_tokens',.0000001,'cached-token'),
            'cache.write':rate('strong','cache_write_tokens',.000002,'write-token'),
            'media':rate('vision','media',.01,'image'),
            'local':rate('local','local_compute',.001,'millisecond'),
            'text.output':rate('text','output_tokens',.000003,'generated-token'),
            'request':rate('remote','request',.02,'attempt')})
    for row in rows:
        row['plan_sha256']=plan_digest(plan)
        row['cost_quantities']={key:1 if row['system']=='candidate' else 4 for key in plan['cost']['rate_card']}
    return plan,rows


class Itemized(unittest.TestCase):
    def result(self,plan,rows):
        return compare(plan,rows,expected_plan_sha256=plan_digest(plan))

    def test_mixed_provider_units_cache_media_local_and_calls(self):
        plan,rows=itemized();result=self.result(plan,rows)
        expected=sum(Decimal(str(r['per_unit'])) for r in plan['cost']['rate_card'].values())*80
        self.assertEqual(Decimal(result['summaries']['baseline']['cost']),expected)
        self.assertEqual(result['exact_point']['cost_ratio'],'0.25')
        self.assertEqual(result['summaries']['candidate']['cost_quantities'],{k:20 for k in plan['cost']['rate_card']})
        self.assertFalse(result['meets_numeric_gates'])
        self.assertFalse(result['gates']['measured_held_out'])
        # Token usage stays diagnostic: itemized costs never charge it a second time.
        for row in rows:row['usage']['input_tokens']=10**10
        self.assertEqual(self.result(plan,rows)['summaries']['baseline']['cost'],str(expected))

    def test_missing_extra_unknown_negative_bool_fractional_quantities_fail(self):
        for value in (None,-1,True,.1,10**12+1,float('nan')):
            plan,rows=itemized();rows[0]['cost_quantities']['local']=value
            with self.assertRaises(ValueError):self.result(plan,rows)
        for change in ('missing','extra','absent'):
            plan,rows=itemized()
            if change=='missing':del rows[0]['cost_quantities']['local']
            if change=='extra':rows[0]['cost_quantities']['unpriced']=1
            if change=='absent':del rows[0]['cost_quantities']
            with self.assertRaises(ValueError):self.result(plan,rows)

    def test_bad_rate_card_scope_and_post_freeze_changes_fail(self):
        for value in (None,-1,True,float('inf'),float('nan'),1e10):
            plan,rows=itemized();plan['cost']['rate_card']['media']['per_unit']=value
            with self.assertRaises(ValueError):self.result(plan,rows)
        for card in ({}, {'x':{}}, {str(i):rate('p','other',1) for i in range(65)}, {'x':rate('p','unrecognized',1)}):
            plan,rows=itemized();plan['cost']['rate_card']=card
            with self.assertRaises(ValueError):self.result(plan,rows)
        plan,rows=itemized();plan['cost']['scope']='VLM-only'
        with self.assertRaises(ValueError):self.result(plan,rows)
        plan,rows=itemized();digest=plan_digest(plan);plan['cost']['rate_card']['media']['revision']='changed'
        with self.assertRaises(ValueError):compare(plan,rows,expected_plan_sha256=digest)

    def test_strict_decimal_boundary_includes_tiny_other_charges(self):
        plan,rows=itemized()
        plan['cost']['rate_card']={'main':rate('p','request',1),'extra':rate('p','other',1e-300)}
        for row in rows:
            row['plan_sha256']=plan_digest(plan)
            row['cost_quantities']=dict(main=80 if row['system']=='candidate' else 100,
                                       extra=1 if row['system']=='candidate' else 0)
        with localcontext() as context:
            context.prec=6
            result=self.result(plan,rows)
        self.assertEqual(result['point']['cost_ratio'],.8)
        self.assertGreater(Decimal(result['exact_point']['cost_ratio']),Decimal('.8'))
        self.assertFalse(result['gates']['cost'])
        for row in rows:row['cost_quantities']['extra']=0
        self.assertTrue(self.result(plan,rows)['gates']['cost'])
