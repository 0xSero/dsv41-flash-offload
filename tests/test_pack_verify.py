import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('pack_verify', Path(__file__).parents[1] / 'pack/verify.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class PackVerifyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.pack = root / 'pack'; self.pack.mkdir()
        for name in ('Mia-DeepSeek-V4.1-Flash-EXL3-3.0bpw', 'DeepSeek-V4.1-Flash-engram'): (root / name).mkdir()
        self.key = 'layers.0.attn.wo_a.weight'
        (self.pack / 'woa-layers-0.safetensors').write_bytes(b'woa')
        (self.pack / 'native-dense.safetensors').write_bytes(b'dense')
        (self.pack / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {self.key: 'woa-layers-0.safetensors'}}))
        self.want = {'final_files': {}, 'grouped_woa': {self.key: hashlib.sha256(b'woa').hexdigest()}}
        (self.pack / 'adapter-receipt.json').write_text(json.dumps({'grouped_woa': [{'key': self.key, 'sha256': self.want['grouped_woa'][self.key]}]}))
        (self.pack / 'native-dense-receipt.json').write_text(json.dumps({'sha256': hashlib.sha256(b'dense').hexdigest()}))
    def test_valid_pack(self): self.assertEqual(module.verify(self.pack, self.want), [])
    def test_missing_woa_fails_even_with_matching_receipt(self):
        (self.pack / 'woa-layers-0.safetensors').unlink()
        self.assertTrue(module.verify(self.pack, self.want))
    def test_corrupted_woa_fails_even_with_matching_receipt(self):
        (self.pack / 'woa-layers-0.safetensors').write_bytes(b'bad')
        self.assertTrue(module.verify(self.pack, self.want))
    def test_corrupted_native_dense_fails(self):
        (self.pack / 'native-dense.safetensors').write_bytes(b'bad')
        self.assertTrue(module.verify(self.pack, self.want))
    def test_missing_dense_hash_fails(self):
        (self.pack / 'native-dense-receipt.json').write_text('{}')
        self.assertTrue(module.verify(self.pack, self.want))
    def test_empty_group_receipt_fails(self):
        (self.pack / 'adapter-receipt.json').write_text('{"grouped_woa":[]}')
        self.assertTrue(module.verify(self.pack, self.want))
