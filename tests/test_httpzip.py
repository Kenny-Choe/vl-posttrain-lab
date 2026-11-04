import io
import re
import unittest
import zipfile
from unittest.mock import patch
from vl_posttrain.data.httpzip import BoundedHTTPFile


class Response:
    def __init__(self,payload,start,end,total,*,status=206,etag='"fixture-v1"'):
        self.status=status;self.headers={'Content-Range':f'bytes {start}-{end}/{total}','ETag':etag}
        self.payload=payload
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self,n):return self.payload[:n]


class HTTPZipTests(unittest.TestCase):
    def test_zip_reads_only_requested_ranges_and_crc(self):
        memory=io.BytesIO()
        with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('train/one.json','{"native":"train"}')
        raw=memory.getvalue();events=[]
        def open_range(request,timeout):
            start,end=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',request.headers['Range']).groups())
            events.append((start,end));return Response(raw[start:end+1],start,end,len(raw))
        with patch('urllib.request.urlopen',side_effect=open_range):
            stream=BoundedHTTPFile('https://example.invalid/fixture.zip',max_bytes=4096)
            with zipfile.ZipFile(stream) as z:self.assertEqual(z.read('train/one.json'),b'{"native":"train"}')
            self.assertGreater(len(events),1)
            self.assertEqual(stream.receipt()['download_payload_bytes'],sum(b-a+1 for a,b in events))

    def test_server_ignoring_range_refused_before_reading_big_body(self):
        response=Response(b'x',0,0,10**10,status=200)
        with patch.object(response,'read',side_effect=AssertionError('must not consume whole body')):
            with patch('urllib.request.urlopen',return_value=response):
                with self.assertRaisesRegex(RuntimeError,'Range_not_honored'):BoundedHTTPFile('https://example.invalid/fixture.zip')

    def test_file_version_change_and_payload_cap_stop(self):
        replies=[Response(b'a',0,0,10),Response(b'b',1,1,10,etag='"fixture-v2"')]
        with patch('urllib.request.urlopen',side_effect=replies):
            stream=BoundedHTTPFile('https://example.invalid/fixture.zip');stream.seek(1)
            with self.assertRaisesRegex(RuntimeError,'validator_changed'):stream.read(1)
        with patch('urllib.request.urlopen',return_value=Response(b'a',0,0,10)):
            stream=BoundedHTTPFile('https://example.invalid/fixture.zip',max_bytes=1)
            with self.assertRaisesRegex(RuntimeError,'payload_cap'):stream.read(2)

    def test_wrong_content_range_not_trusted(self):
        with patch('urllib.request.urlopen',return_value=Response(b'a',1,1,10)):
            with self.assertRaisesRegex(RuntimeError,'does_not_match_request'):BoundedHTTPFile('https://example.invalid/fixture.zip')
