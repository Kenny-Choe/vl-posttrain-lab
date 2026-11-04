"""加载权重前的只读准入。不导入Torch，不把宿主机总内存当容器额度。"""
from __future__ import annotations

import os
import re
import subprocess
import csv
from pathlib import Path
from .io import read_json, file_sha256

GiB=1024**3


def linux_memory_snapshot(proc_root=Path('/proc'),cgroup_root=Path('/sys/fs/cgroup')) -> dict:
    proc_root,cgroup_root=Path(proc_root),Path(cgroup_root)
    mem={}
    p=proc_root/'meminfo'
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith(('MemTotal:','MemAvailable:')):
                name,value,*_=line.split();mem[name.rstrip(':')]=int(value)*1024
    limit_path=cgroup_root/'memory.max';usage_path=cgroup_root/'memory.current'
    limit=None;usage=None
    if limit_path.exists():
        text=limit_path.read_text().strip()
        if text!='max':limit=int(text)
    if usage_path.exists():usage=int(usage_path.read_text().strip())
    candidates=[mem['MemAvailable']] if 'MemAvailable' in mem else []
    if limit is not None and usage is not None:candidates.append(max(0,limit-usage))
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
    quota=None
    cpu=cgroup_root/'cpu.max'
    if cpu.exists():
        amount,period=cpu.read_text().split()
        if amount!='max':quota=int(amount)/int(period)
    return {'host_total_bytes':mem.get('MemTotal'),'host_available_bytes':mem.get('MemAvailable'),
            'cgroup_limit_bytes':limit,'cgroup_current_bytes':usage,
            'effective_available_bytes':min(candidates) if candidates else None,
            'affinity_cpus':affinity,'cpu_quota_cores':quota,
            'effective_cpu_cores':min(affinity,quota) if affinity is not None and quota is not None else affinity}


def memory_snapshot() -> dict:
    if os.name!='nt':return linux_memory_snapshot()
    import ctypes
    class Status(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('total',ctypes.c_ulonglong),
                  ('available',ctypes.c_ulonglong),('total_page',ctypes.c_ulonglong),('available_page',ctypes.c_ulonglong),
                  ('total_virtual',ctypes.c_ulonglong),('available_virtual',ctypes.c_ulonglong),('extended',ctypes.c_ulonglong)]
    status=Status();status.length=ctypes.sizeof(Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('无法查询实际Windows可用内存')
    return {'host_total_bytes':status.total,'host_available_bytes':status.available,
            'effective_available_bytes':status.available,'cgroup_limit_bytes':None,'cgroup_current_bytes':None,
            'effective_cpu_cores':os.cpu_count()}


def weight_inventory(model: str|Path) -> dict:
    root=Path(model).resolve()
    config=root/'config.json';index=root/'model.safetensors.index.json'
    if not config.is_file() or not index.is_file():
        raise ValueError('需要已有本地config与分片safetensors索引；不自动下载/猜测权重规模')
    conf=read_json(config);data=read_json(index)
    total=data.get('metadata',{}).get('total_size')
    if isinstance(total,bool) or not isinstance(total,int) or total<=0:
        raise ValueError('索引缺少可靠total_size')
    if conf.get('quantization_config'):
        raise ValueError('量化模型的CPU展开内存和新后端兼容性需单独验证')
    shards=sorted(set(data['weight_map'].values()))
    for name in shards:
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('权重分片路径越界或缺失')
    # Actual Qwen3-VL config stores dtype under text_config.dtype. Keep the
    # location in the receipt; a root-only lookup rejects usable metadata.
    candidates=[('dtype',conf.get('dtype')),('torch_dtype',conf.get('torch_dtype')),
                ('text_config.dtype',conf.get('text_config',{}).get('dtype')),
                ('text_config.torch_dtype',conf.get('text_config',{}).get('torch_dtype'))]
    location,dtype=next(((key,value) for key,value in candidates if value is not None),('missing','unknown'))
    factor={'bfloat16':2.,'float16':2.,'float32':1.,'float64':.5}.get(str(dtype))
    return {'stored_tensor_bytes':total,'shards':len(shards),'stored_dtype':dtype,'dtype_metadata_location':location,
            'cpu_float32_expansion':factor,'index_sha256':file_sha256(index),'config_sha256':file_sha256(config),
            'shard_presence_checked':True,'tensor_values_loaded':False}


def gpu_snapshot() -> list[dict]:
    try:
        process=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free',
                                '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    except (FileNotFoundError,subprocess.TimeoutExpired):return []
    if process.returncode:return []
    result=[]
    for fields in csv.reader(process.stdout.splitlines()):
        if len(fields)!=4:continue
        index,uuid,total,free=[x.strip() for x in fields]
        if index.isdigit() and total.isdigit() and free.isdigit():
            result.append({'physical_index':int(index),'uuid':uuid,'total_bytes':int(total)*1024**2,'free_bytes':int(free)*1024**2})
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        selected=[]
        for token in visible.split(','):
            token=token.strip()
            if not token or token=='-1':return []
            matches=[r for r in result if str(r['physical_index'])==token or r['uuid'].startswith(token)]
            if len(matches)!=1:return []
            selected.append(matches[0])
    else:selected=sorted(result,key=lambda r:r['physical_index'])
    return [{**r,'index':index} for index,r in enumerate(selected)]


def model_admission(model: str|Path,device: str='cpu',memory: dict|None=None,gpus: list[dict]|None=None) -> dict:
    inventory=weight_inventory(model);memory=memory if memory is not None else memory_snapshot()
    reasons=[];available=memory.get('effective_available_bytes');required=None
    if device=='cpu':
        factor=inventory['cpu_float32_expansion']
        if factor is None:reasons.append('stored_dtype_unknown_cannot_estimate_float32_expansion')
        else:required=int(inventory['stored_tensor_bytes']*factor)+2*GiB
        if available is None:reasons.append('effective_memory_unknown')
        elif required is not None and available<required:reasons.append('effective_CPU_memory_below_estimated_weights_plus_headroom')
        gpu=None
    elif re.fullmatch(r'cuda(?::\d+)?',device):
        wanted=int(device.split(':')[1]) if ':' in device else 0
        entries=gpus if gpus is not None else gpu_snapshot()
        gpu=next((g for g in entries if g['index']==wanted),None)
        # GPU加载仍需CPU处理分片。2GiB是工程下限，并非保证峰值。
        if available is None or available<2*GiB:reasons.append('CPU_staging_headroom_below_2GiB_or_unknown')
        required=inventory['stored_tensor_bytes']+2*GiB
        if gpu is None:reasons.append('requested_GPU_not_visible')
        elif gpu['free_bytes']<required:reasons.append('GPU_free_memory_below_estimated_weights_plus_headroom')
    else:raise ValueError('只支持明确cpu/cuda:N设备')
    return {'admitted':not reasons,'reasons':reasons,'device':device,'memory':memory,'GPU':gpu,
            'estimated_required_device_bytes':required,'weight_inventory':inventory,
            'estimate_scope':'weights_and_2GiB_headroom_not_peak_memory_guarantee',
            'model_weights_loaded':False,'training':False,'downloads':0}
