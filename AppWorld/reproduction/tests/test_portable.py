import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

HERE = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE/f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def task(root, *, output=None, finished=True):
    p = root/'sample_1'
    (p/'logs').mkdir(parents=True)
    (p/'misc').mkdir()
    if finished:
        (p/'misc/finished').touch()
    value = {'input': {'model':'test-model', 'temperature':1, 'seed':100, 'timeout':14400,
                       'tools':[{'type':'function'}]},
             'output':output or {'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[
                 {'function':{'name':'app__read','arguments':'{"ids":[1,2]}'}}]}}]}}
    (p/'logs/lm_calls.jsonl').write_text(json.dumps(value)+'\n')
    (p/'logs/api_calls.jsonl').write_text('{}\n')
    return p


def test_incomplete_task_cannot_be_scored(tmp_path):
    task(tmp_path, finished=False)
    with pytest.raises(RuntimeError, match='Incomplete task'):
        module('run').audit(tmp_path, ['sample_1'], 'test-model', 'baseline', 'native')


def test_missing_task_cannot_shrink_denominator(tmp_path):
    task(tmp_path)
    with pytest.raises(RuntimeError, match='task set'):
        module('run').audit(tmp_path, ['sample_1', 'sample_2'], 'test-model', 'baseline', 'native')


def test_transport_errors_and_leaked_calls_remain_visible(tmp_path):
    task(tmp_path, output={'error':'HTTP 400: context too long','choices':[{'message':{
        'content':'[{"name":"app__read","arguments":{}}]'}}]})
    result=module('run').audit(tmp_path,['sample_1'],'test-model','baseline','native')
    assert len(result['model_response_errors'])==1
    assert result['format_or_execution_issues'][0]['issue']=='unparsed native call array'


def test_baseline_cannot_silently_enable_harness(tmp_path):
    p=task(tmp_path)
    (p/'logs/harness_events.jsonl').write_text('{"layer":"H2"}\n')
    with pytest.raises(RuntimeError, match='Baseline unexpectedly'):
        module('run').audit(tmp_path,['sample_1'],'test-model','baseline','native')


def test_timeout_patch_applies_once_and_is_idempotent(tmp_path):
    # The patch must work on the pinned original, without requiring a GPU/model.
    root=tmp_path/'appworld';root.mkdir()
    subprocess.run(['git','init','-q',str(root)],check=True)
    patch=(HERE/'patches/appworld-http-timeout.patch').read_text()
    # Reconstruct the exact affected original blocks from unified diff hunks.
    blocks=[]
    for hunk in patch.split('@@')[2::2]:
        lines=[]
        for line in hunk.splitlines():
            if line.startswith(' ') or line.startswith('-'):
                lines.append(line[1:])
        blocks.append('\n'.join(lines))
    target=root/'experiments/code/simplified/language_model.py'
    target.parent.mkdir(parents=True)
    target.write_text('\n\n'.join(blocks)+'\n')
    setup=module('setup')
    setup.apply_patch(root)
    once=target.read_bytes()
    setup.apply_patch(root)
    assert target.read_bytes()==once
    assert b'kwargs["timeout"] = timeout' in once
