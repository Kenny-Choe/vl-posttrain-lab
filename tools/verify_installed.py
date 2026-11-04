"""在项目outputs中构建并安装wheel，离开源码目录验证CLI；不训练或联网。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vl_posttrain.io import file_sha256,identity,write_json
from package_release import release_files,check_files


def check_install(output:Path) -> dict:
    output=output.resolve()
    if not output.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('安装验证只能写入当前仓库outputs')
    if output.exists():
        raise ValueError('验证目录已存在，先核对原收据；新验证使用新attempt目录')
    output.mkdir(parents=True)
    source=output/'source';installed=output/'installed';wheels=output/'wheels';empty=output/'empty_cwd'
    for path in [source,installed,wheels,empty,output/'temp']:
        path.mkdir()
    files=release_files();hygiene=check_files(files)
    hashes={p.relative_to(ROOT).as_posix():file_sha256(p) for p in files}
    for path in files:
        copied=source/path.relative_to(ROOT);copied.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,copied)
    env=dict(os.environ)
    env.update(PYTHONPATH='',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',
               PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1',
               PIP_CACHE_DIR=str(output/'pip-cache'),TEMP=str(output/'temp'),TMP=str(output/'temp'),
               SOURCE_DATE_EPOCH=str(int(datetime(2026,11,4,tzinfo=timezone.utc).timestamp())))
    logs=[];started=time.perf_counter()
    def run(label,command,cwd,environment):
        begin=time.perf_counter()
        result=subprocess.run(command,cwd=cwd,env=environment,capture_output=True,text=True,encoding='utf-8',timeout=120)
        (output/(label+'.stdout.log')).write_text(result.stdout,encoding='utf-8')
        (output/(label+'.stderr.log')).write_text(result.stderr,encoding='utf-8')
        logs.append({'step':label,'exit_code':result.returncode,'wall_seconds':time.perf_counter()-begin})
        if result.returncode:
            raise RuntimeError(label+'失败，请查看该attempt日志')
        return result
    receipt={'status':'started','scope':'offline_wheel_install_and_CPU_CLI_only','source_sha256':identity(hashes),
             'source_files':len(files),'network_disabled':True,'model_weights_loaded':False,'optimizer_steps':0,
             'model_generation_calls':0,'GPU_jobs_started':0,'source_files_changed':False}
    write_json(output/'launch_manifest.json',receipt)
    try:
        run('build',[sys.executable,'-m','pip','wheel','--no-index','--no-deps','--no-build-isolation',
                     '--wheel-dir',str(wheels),str(source)],source,env)
        candidates=list(wheels.glob('vl_posttrain_lab-*.whl'))
        if len(candidates)!=1:raise ValueError('应且仅应生成一个本项目wheel')
        wheel=candidates[0]
        import zipfile
        with zipfile.ZipFile(wheel) as archive:
            if any(name.startswith('vl_posttrain/evidence/') for name in archive.namelist()):
                raise ValueError('wheel包含个人历史导入模块')
        run('install',[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-compile',
                       '--target',str(installed),str(wheel)],empty,env)
        child_env={**env,'PYTHONPATH':str(installed)}
        prove="import vl_posttrain,json;from pathlib import Path;p=Path(vl_posttrain.__file__).resolve();assert p.is_relative_to(Path("+repr(str(installed))+"));print(json.dumps({'installed_package_verified':True,'module':str(p),'version':vl_posttrain.__version__}))"
        run('import_identity',[sys.executable,'-c',prove],empty,child_env)
        outputs={}
        for label,arguments in [
            ('doctor',['doctor']),('gradient',['gradient-probe']),
            ('plan',['plan','--config',str(source/'configs/route.target.json')]),
            ('dataset',['dataset-demo','--qa-count','96','--output',str(output/'fixture96')])]:
            result=run(label,[sys.executable,'-m','vl_posttrain',*arguments],empty,child_env)
            outputs[label]=json.loads(result.stdout)
        if outputs['doctor']['models_loaded'] or outputs['plan']['training_started']:
            raise ValueError('CPU安装验证越过了无模型/无训练范围')
        if outputs['dataset']['target_12k_dataset_counts_met'] or outputs['dataset']['reviewed_real_data_release_passed']:
            raise ValueError('fixture范围错误')
        course=run('curriculum',[sys.executable,'-m','vl_posttrain','curriculum-fixture',
            '--domain-release',str(output/'fixture96/release'),'--image-root',str(output/'fixture96'),
            '--output',str(output/'course72'),'--localization','32','--search-stop','32','--replay-count','8'],empty,child_env)
        outputs['curriculum']=json.loads(course.stdout)
        if outputs['curriculum']['trajectories']!=72 or outputs['curriculum']['ready_for_training']:
            raise ValueError('已安装课程入口范围错误')
        if not all(file_sha256(ROOT/relative)==digest for relative,digest in hashes.items()):
            raise ValueError('验证期间原源码发生变化')
        receipt.update(status='passed',wheel=wheel.name,wheel_sha256=file_sha256(wheel),wheel_bytes=wheel.stat().st_size,
                       installed_package_verified=True,executed_outside_source_directory=True,
                       commands=list(outputs),dataset_fixture_QA=96,release_hygiene=hygiene)
    except Exception as exc:
        receipt.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        receipt.update(wall_seconds=time.perf_counter()-started,steps=logs)
        write_json(output/'receipt.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/installed_check_v1')
    args=parser.parse_args()
    print(json.dumps(check_install(args.output),ensure_ascii=False,indent=2))
