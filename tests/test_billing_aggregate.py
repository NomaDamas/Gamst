import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.billing_aggregate import build_report


class OriginalAggregate(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='.')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.manifest=dict(schema_version=1,kind='openai_aggregate_acquisition',account_scope='fixture-account',
            origin='synthetic_fixture',acquired_at=300,queries=[])

    def page(self, name, start=0, end=100, rows=None, more=False, cursor=None):
        if rows is None: rows=[dict(object='organization.costs.result',amount=dict(value=.123456789,currency='usd'),project_id='fixture-project')]
        value=dict(object='page',data=[dict(object='bucket',start_time=start,end_time=end,results=rows)],has_more=more,next_page=cursor)
        data=json.dumps(value,separators=(',',':')).encode();(self.root/name).write_bytes(data)
        return dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),request_cursor=None)

    def query(self, qid, pages, **changes):
        q=dict(id=qid,endpoint='costs',start_time=0,end_time=100,bucket_width='1d',
            group_by=['project_id'],filters={},pages=pages)
        q.update(changes);self.manifest['queries'].append(q);return q

    def test_actual_bytes_exact_amount_and_original_pointer_no_invoice(self):
        page=self.page('original.json');self.query('q',[page])
        report,raw=build_report(self.manifest,self.root)
        view=report['views'][0]
        self.assertEqual(view['known_positive_usd_atoms'],123456789)
        self.assertEqual(view['rows'][0]['amount']['lexeme'],'0.123456789')
        self.assertIn('/data/0/results/0',view['rows'][0]['source_pointer'])
        self.assertFalse(report['eligible_for_existing_native3_money_join'])
        self.assertIsNone(report['per_attempt_cost'])
        self.assertEqual(raw[page['sha256']],(self.root/'original.json').read_bytes())
        self.assertNotIn('invoice_id',json.dumps(report))

    def test_split_bucket_pages_and_incomplete_chain(self):
        first=self.page('a.json',more=True,cursor='next')
        second=self.page('b.json',rows=[dict(object='organization.costs.result',amount=dict(value=1,currency='usd'),project_id='other')])
        second['request_cursor']='next';q=self.query('q',[first,second])
        report,_=build_report(self.manifest,self.root)
        self.assertEqual(report['views'][0]['known_positive_usd_atoms'],1123456789)
        q['pages'].pop()
        with self.assertRaisesRegex(ValueError,'incomplete original pagination'):build_report(self.manifest,self.root)

    def test_missing_and_null_selected_group_cannot_double_count(self):
        row=dict(object='organization.costs.result',amount=dict(value=1,currency='usd'))
        self.query('q',[self.page('a.json',rows=[row,dict(row,project_id=None)])])
        with self.assertRaisesRegex(ValueError,'duplicate logical aggregate group'):build_report(self.manifest,self.root)

    def test_alias_revision_and_overlap_never_added(self):
        original=self.page('a.json');self.query('q',[original]);self.query('alias',[dict(original)])
        report,_=build_report(self.manifest,self.root)
        self.assertEqual(len(report['views']),1);self.assertEqual(report['query_aliases'][0]['canonical_query_id'],'q')
        revised=self.page('b.json',rows=[dict(object='organization.costs.result',amount=dict(value=2,currency='usd'),project_id='fixture-project')])
        self.query('revision',[revised]);report,_=build_report(self.manifest,self.root)
        self.assertTrue(all(v['state']=='revision_quarantined' for v in report['views']))
        self.assertTrue(all(v['known_positive_usd_atoms'] is None for v in report['views']))
        self.assertIsNone(report['cross_query_total'])

    def test_gaps_negative_and_unknown_projection_are_retained(self):
        rows=[dict(object='organization.costs.result',amount=dict(value=-.01,currency='usd'),project_id='p',future_fee=True)]
        self.query('q',[self.page('a.json',start=20,end=70,rows=rows)])
        report,_=build_report(self.manifest,self.root);view=report['views'][0]
        self.assertFalse(view['coverage_complete']);self.assertFalse(view['projection_complete'])
        self.assertEqual(view['negative_adjustment_usd_atoms'],-10000000)
        self.assertEqual(view['known_positive_usd_atoms'],0)
        self.assertEqual(view['missing_intervals'],[dict(start_time=0,end_time=20),dict(start_time=70,end_time=100)])

    def test_usage_subsets_not_added_and_unknowns_visible(self):
        row=dict(object='organization.usage.completions.result',input_tokens=100,input_cached_tokens=40,num_model_requests=1,project_id='p')
        self.query('q',[self.page('a.json',rows=[row])],endpoint='completions_usage',bucket_width='1m')
        report,_=build_report(self.manifest,self.root);view=report['views'][0]
        self.assertEqual(view['rows'][0]['usage']['input_tokens']['value'],100)
        self.assertEqual(view['rows'][0]['usage']['input_cached_tokens']['value'],40)
        self.assertEqual(view['rows'][0]['usage']['output_tokens']['state'],'missing')
        self.assertIsNone(view['known_positive_usd_atoms'])

    def test_original_mutation_and_symlink_rejected(self):
        page=self.page('a.json');self.query('q',[page]);data=(self.root/'a.json').read_bytes()
        (self.root/'a.json').write_bytes(data.replace(b'123456789',b'923456789'))
        with self.assertRaisesRegex(ValueError,'frozen hash'):build_report(self.manifest,self.root)
        (self.root/'link.json').symlink_to(self.root/'a.json');page['path']='link.json'
        with self.assertRaises(OSError):build_report(self.manifest,self.root)

    def test_empty_bucket_inventory_is_unknown_not_zero_coverage(self):
        page=self.page('a.json');path=self.root/'a.json'
        value=json.loads(path.read_bytes());value['data']=[]
        data=json.dumps(value).encode();path.write_bytes(data)
        page.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        self.query('q',[page]);report,_=build_report(self.manifest,self.root)
        view=report['views'][0]
        self.assertFalse(view['coverage_complete'])
        self.assertEqual(view['missing_intervals'],[dict(start_time=0,end_time=100)])
        self.assertIsNone(report['per_attempt_cost'])

    def test_non_usd_and_missing_amount_remain_unknown(self):
        rows=[dict(object='organization.costs.result',amount=dict(value=7,currency='eur'),project_id='p'),
              dict(object='organization.costs.result',project_id='q')]
        self.query('q',[self.page('a.json',rows=rows)])
        report,_=build_report(self.manifest,self.root);view=report['views'][0]
        self.assertEqual(view['unknown_amount_rows'],2)
        self.assertTrue(all(row['usd_atoms'] is None for row in view['rows']))
        self.assertIsNone(report['cross_query_total'])

    def test_query_filter_mismatch_fails_and_missing_scope_stays_unverified(self):
        q=self.query('q',[self.page('a.json')],filters={'project_ids':['different']})
        with self.assertRaisesRegex(ValueError,'contradicts query filter'):build_report(self.manifest,self.root)
        q['pages']=[self.page('b.json',rows=[dict(object='organization.costs.result',amount=dict(value=1,currency='usd'))])]
        report,_=build_report(self.manifest,self.root)
        self.assertIn('unverified_filter_scope',report['views'][0]['diagnostics'])
        self.assertFalse(report['account_isolation_verified'])

    def test_page_reference_bound_counts_aliases(self):
        page=self.page('a.json');self.query('q',[dict(page) for _ in range(65)])
        with self.assertRaisesRegex(ValueError,'page reference bound'):build_report(self.manifest,self.root)

    def test_joint_rows_bound_prevents_expansion(self):
        rows=[dict(object='organization.costs.result',amount=dict(value=1,currency='usd'),project_id=str(i)) for i in range(10001)]
        self.query('q',[self.page('a.json',rows=rows)])
        with self.assertRaisesRegex(ValueError,'joint original row bound'):build_report(self.manifest,self.root)

    def test_escaped_report_bytes_are_bounded_before_publication(self):
        rows=[dict(object='organization.costs.result',amount=dict(value=1,currency='usd'),project_id=('界'*250)+str(i)) for i in range(9000)]
        # Multiple original files remain within each 4MiB original bound.
        pages=[]
        for index in range(3):
            page=self.page(str(index)+'.json',rows=rows[index*3000:(index+1)*3000],more=index<2,cursor=str(index+1) if index<2 else None)
            path=self.root/page['path']
            data=json.dumps(json.loads(path.read_bytes()),ensure_ascii=False,separators=(',',':')).encode()
            path.write_bytes(data);page.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            page['request_cursor']=str(index) if index else None
            pages.append(page)
        self.query('q',pages)
        with self.assertRaisesRegex(ValueError,'expanded aggregate report'):build_report(self.manifest,self.root)

    def test_diagnostic_examples_bounded_without_dropping_count(self):
        rows=[dict(object='organization.costs.result',project_id=str(i),future_field='untrusted') for i in range(30)]
        self.query('q',[self.page('a.json',rows=rows)])
        report,_=build_report(self.manifest,self.root)
        diagnostic=report['views'][0]['diagnostics']['unknown_projection_fields']
        self.assertEqual(diagnostic['count'],30);self.assertEqual(len(diagnostic['examples']),16)
        self.assertNotIn('untrusted',json.dumps(report))
