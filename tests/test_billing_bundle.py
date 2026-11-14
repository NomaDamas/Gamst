import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from benchmarks.billing_bundle import persist_bundle, load_acquisition
from benchmarks.billing_aggregate import encoded, sha
from test_billing_aggregate import OriginalAggregate


class BundlePersistence(OriginalAggregate):
    # Reuse fixture helpers without executing the aggregate test methods here.
    def test_bundle_preserves_original_bytes_and_reuses_verified_output(self):
        page=self.page('a.json');self.query('q',[page]);out=self.root/'bundle'
        report=persist_bundle(self.manifest,self.root,out)
        self.assertEqual((out/(page['sha256']+'.original.json')).read_bytes(),(self.root/'a.json').read_bytes())
        self.assertEqual(os.stat(out).st_mode & 0o777,0o700)
        self.assertTrue(all(os.stat(p).st_mode & 0o777 == 0o600 for p in out.iterdir()))
        self.assertEqual(report,persist_bundle(self.manifest,self.root,out))

    def test_consistently_rehashed_false_report_cannot_be_reused(self):
        self.query('q',[self.page('a.json')]);out=self.root/'bundle'
        persist_bundle(self.manifest,self.root,out)
        report=json.loads((out/'report.json').read_bytes());report['per_attempt_cost']=0
        data=encoded(report);(out/'report.json').write_bytes(data)
        completion=json.loads((out/'completion.json').read_bytes())
        completion['files']['report.json']=dict(bytes=len(data),sha256=sha(data))
        (out/'completion.json').write_bytes(encoded(completion))
        with self.assertRaises(ValueError):persist_bundle(self.manifest,self.root,out)
        self.assertEqual((out/'report.json').read_bytes(),data)

    def test_partial_write_has_no_success_marker_and_is_not_repaired(self):
        self.query('q',[self.page('a.json')]);out=self.root/'bundle'
        from benchmarks.billing_bundle import _write_file
        def fail(directory,name,data):
            if name=='report.json':raise OSError('simulated full disk')
            return _write_file(directory,name,data)
        with patch('benchmarks.billing_bundle._write_file',side_effect=fail):
            with self.assertRaises(OSError):persist_bundle(self.manifest,self.root,out)
        self.assertFalse((out/'completion.json').exists())
        with self.assertRaises(ValueError):persist_bundle(self.manifest,self.root,out)
        self.assertEqual(set(p.name for p in out.iterdir()),{'acquisition.json'})

    def test_output_symlink_and_fifo_rejected_without_blocking(self):
        self.query('q',[self.page('a.json')]);out=self.root/'bundle'
        persist_bundle(self.manifest,self.root,out)
        link=self.root/'link';link.symlink_to(out,target_is_directory=True)
        with self.assertRaises(OSError):persist_bundle(self.manifest,self.root,link)
        (out/'report.json').unlink();os.mkfifo(out/'report.json',0o600)
        with self.assertRaises(ValueError):persist_bundle(self.manifest,self.root,out)

    def test_acquisition_loader_rejects_duplicate_keys_and_noninteger_declarations(self):
        self.query('q',[self.page('a.json')]);path=self.root/'acquisition.json'
        path.write_bytes(encoded(self.manifest))
        self.assertEqual(load_acquisition(path),self.manifest)
        path.write_bytes(encoded(self.manifest).replace(b'"schema_version":1',b'"schema_version":1,"schema_version":1'))
        with self.assertRaises(ValueError):load_acquisition(path)
        path.write_bytes(encoded(self.manifest).replace(b'"schema_version":1',b'"schema_version":1.0'))
        with self.assertRaises(ValueError):load_acquisition(path)

    def test_original_revision_cannot_overwrite_existing_output(self):
        page=self.page('a.json');self.query('q',[page]);out=self.root/'bundle'
        persist_bundle(self.manifest,self.root,out);before=(out/'report.json').read_bytes()
        self.manifest['acquired_at']=301
        with self.assertRaises(ValueError):persist_bundle(self.manifest,self.root,out)
        self.assertEqual((out/'report.json').read_bytes(),before)


# Helpers inherited above are useful, but avoid reporting duplicate test coverage.
for _name in tuple(vars(OriginalAggregate)):
    if _name.startswith('test_'):setattr(BundlePersistence,_name,None)
del OriginalAggregate
