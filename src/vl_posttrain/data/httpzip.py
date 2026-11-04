"""有限HTTP Range读取远程公开zip；拒绝服务器忽略Range或中途换版本。"""
from __future__ import annotations

import io
import re
import time
import urllib.request


class BoundedHTTPFile(io.RawIOBase):
    def __init__(self,url:str,*,max_bytes:int=96*1024**2,max_requests:int=100,wall_seconds:int=300):
        super().__init__()
        if not url.startswith('https://') or min(max_bytes,max_requests,wall_seconds)<1:
            raise ValueError('需要HTTPS和正的资源上限')
        self.url=url;self.max_bytes=max_bytes;self.max_requests=max_requests;self.wall_seconds=wall_seconds
        self.started=time.monotonic();self.position=0;self.payload_bytes=0;self.requests=0
        self.size=None;self.validator=None;self.validator_kind=None;self.events=[]
        self._range(0,0)

    def _range(self,start:int,end:int):
        if self.requests>=self.max_requests or time.monotonic()-self.started>self.wall_seconds:
            raise RuntimeError('HTTP_request_or_wall_cap')
        wanted=end-start+1
        if wanted<1 or self.payload_bytes+wanted>self.max_bytes:
            raise RuntimeError('HTTP_payload_cap')
        request=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}',
            'Accept-Encoding':'identity','User-Agent':'VL-PostTrain-Lab-CPU-Data/0.1'})
        self.requests+=1
        with urllib.request.urlopen(request,timeout=min(30,self.wall_seconds)) as response:
            if response.status!=206:raise RuntimeError('HTTP_Range_not_honored_refuse_full_download')
            content_range=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
            if not content_range:raise RuntimeError('HTTP_Content_Range_missing_or_invalid')
            a,b,total=map(int,content_range.groups())
            if (a,b)!=(start,end) or b>=total:raise RuntimeError('HTTP_Content_Range_does_not_match_request')
            if self.size is not None and self.size!=total:raise RuntimeError('remote_file_size_changed')
            self.size=total
            kind='ETag' if response.headers.get('ETag') else 'Last-Modified'
            validator=response.headers.get(kind)
            if self.validator is not None and (kind!=self.validator_kind or validator!=self.validator):
                raise RuntimeError('remote_file_validator_changed')
            if self.validator is None and validator is not None:
                self.validator=validator;self.validator_kind=kind
            data=response.read(wanted+1)
            self.payload_bytes+=len(data)
            if len(data)!=wanted:raise RuntimeError('HTTP_range_length_mismatch')
            self.events.append({'start':start,'end':end,'bytes':len(data)})
            return data

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=io.SEEK_SET):
        value=offset if whence==io.SEEK_SET else self.position+offset if whence==io.SEEK_CUR else self.size+offset if whence==io.SEEK_END else None
        if value is None or value<0:raise ValueError('invalid_HTTP_file_seek')
        self.position=value;return value
    def read(self,size=-1):
        if self.closed:raise ValueError('read_from_closed_HTTP_file')
        if self.position>=self.size:return b''
        amount=self.size-self.position if size is None or size<0 else min(size,self.size-self.position)
        if amount==0:return b''
        data=self._range(self.position,self.position+amount-1);self.position+=len(data);return data
    def receipt(self):
        return {'url':self.url,'remote_bytes':self.size,'download_payload_bytes':self.payload_bytes,
                'requests':self.requests,'validator_kind':self.validator_kind,'validator':self.validator,
                'HTTP_range_version_validator_present':self.validator is not None,
                'full_archive_downloaded':False,'wire_bytes':'unknown'}
