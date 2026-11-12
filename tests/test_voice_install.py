import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('local_models',Path(__file__).parents[1]/'scripts/local_models.py')
installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)

class VoiceInstallTests(unittest.TestCase):
    def test_optional_install_preserves_base_manifest_and_creates_pinned_config(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root=Path(temporary);base=root/'.local-models';base.mkdir()
            (base/'owner.json').write_text(json.dumps({'project':str(root)}))
            (base/'manifest.json').write_text(json.dumps({'base-model':{'sha256':'existing'}}))
            def download(item):
                name,url=item
                path=base/name;path.parent.mkdir(exist_ok=True);path.write_text('fixture')
                return name,{'url':url,'sha256':'fixture'}
            with patch.object(installer,'ROOT',root),patch.object(installer,'BASE',base),patch.object(installer,'download',download),patch('sys.argv',['local_models','install-voice']),patch.object(installer.tarfile,'open') as archive:
                installer.main()
                archive.assert_not_called()
            manifest=json.loads((base/'manifest.json').read_text())
            self.assertIn('base-model',manifest)
            self.assertEqual(len(manifest),5)
            config=json.loads((base/'voice-emotion/plugins.json').read_text())[0]
            self.assertEqual(config['options']['revision'],installer.VOICE)
            self.assertEqual(config['options']['window_seconds'],3)
            self.assertEqual(config['options']['hop_seconds'],1)
