"""Run the original Life-Harness runner against an existing OpenAI-compatible service."""
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
ADAPTER = HERE.parent


def interrupted(*_):
    raise KeyboardInterrupt


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def read_ids(path):
    ids = [s.strip() for s in path.read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate task IDs: {path}")
    return ids


def audit(root, ids, model, condition, thinking):
    if {p.name for p in root.iterdir()} != set(ids):
        raise RuntimeError('Actual task set differs from requested denominator')
    counts, layers, errors, format_errors = collections.Counter(), collections.Counter(), [], []
    for task_id in ids:
        task = root/task_id
        if not (task/'misc/finished').exists():
            raise RuntimeError(f'Incomplete task: {task_id}; no aggregate score produced')
        rows = [json.loads(s) for s in (task/'logs/lm_calls.jsonl').read_text().splitlines()]
        for row in rows:
            request, response = row['input'], row['output']
            assert request['model'] == model and request['temperature'] == 1 and request['seed'] == 100
            main = request.get('tools') is not None
            assert request.get('timeout') == 14400
            if main:
                assert request.get('max_tokens') is None and request.get('max_completion_tokens') is None
            else:
                assert request.get('max_completion_tokens') == 3000
            override = request.get('extra_body', {}).get('chat_template_kwargs', {})
            assert override == ({'enable_thinking': False} if thinking == 'off' else {})
            counts['main_calls' if main else 'predictor_calls'] += 1
            if response.get('error'):
                errors.append({'task_id': task_id, 'error': response['error']})
            for choice in response.get('choices', []):
                counts['finish_reason_'+str(choice.get('finish_reason'))] += 1
                msg = choice.get('message', {})
                calls = msg.get('tool_calls') or []
                counts['tool_calls'] += len(calls)
                for call in calls:
                    try:
                        assert call['function']['name']
                        assert isinstance(json.loads(call['function']['arguments']), dict)
                    except (ValueError, KeyError, TypeError, AssertionError):
                        format_errors.append({'task_id': task_id, 'issue': 'invalid parsed arguments'})
                if main and not calls:
                    content = msg.get('content') or ''
                    try:
                        value = json.loads(content)
                    except (ValueError, TypeError):
                        value = None
                    if isinstance(value, list) and any(isinstance(v, dict) and 'name' in v and 'arguments' in v for v in value):
                        format_errors.append({'task_id': task_id, 'issue': 'unparsed native call array'})
        event_path = task/'logs/harness_events.jsonl'
        if event_path.exists():
            layers.update(json.loads(s)['layer'] for s in event_path.read_text().splitlines())
        api_path = task/'logs/api_calls.jsonl'
        if not api_path.exists() or not api_path.stat().st_size:
            format_errors.append({'task_id': task_id, 'issue': 'no AppWorld API executions'})
    if condition == 'baseline' and layers:
        raise RuntimeError('Baseline unexpectedly emitted harness events')
    return {'finished_tasks': len(ids), 'calls': dict(counts), 'model_response_errors': errors,
            'format_or_execution_issues': format_errors, 'harness_events_by_layer': dict(layers)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--appworld-root', type=Path, required=True)
    p.add_argument('--server-url', required=True, help='Base URL, with or without trailing /v1')
    p.add_argument('--model', required=True, help='Exact id advertised by /v1/models')
    p.add_argument('--run-name', required=True, help='New identifier; existing outputs are never overwritten')
    p.add_argument('--conditions', nargs='+', choices=['baseline', 'harness'], default=['baseline', 'harness'])
    p.add_argument('--num-processes', type=int, default=16)
    p.add_argument('--pilot', action='store_true', help='5 fixed training tasks per condition; no dev score')
    p.add_argument('--thinking', choices=['native', 'off'], default='native')
    p.add_argument('--max-steps', type=int, default=50, help='Changes protocol if not 50; useful for infrastructure smoke')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', a.run_name):
        p.error('run-name must contain only letters, digits, _, - and .')
    if a.num_processes < 1 or a.max_steps < 1 or len(set(a.conditions)) != len(a.conditions):
        p.error('Use positive worker/step counts and distinct conditions')
    root = a.appworld_root.resolve()
    sys.path.insert(0, str(root/'experiments'))
    sys.path.insert(0, str(ADAPTER/'src'))
    from appworld import update_root, load_task_ids
    from appworld.cli import extract_runner_config
    from appworld.evaluator import evaluate_tasks
    from appworld_agents.code.simplified.language_model import non_cached_lm_call
    import inspect
    if 'timeout' not in inspect.signature(non_cached_lm_call).parameters:
        raise RuntimeError('Run reproduction/setup.py first: HTTP timeout patch is missing')
    update_root(str(root))
    url = a.server_url.rstrip('/')
    if url.endswith('/v1'):
        url = url[:-3]
    key = os.environ.get('NO_API_KEY', os.environ.get('OPENAI_API_KEY', 'EMPTY'))
    request = urllib.request.Request(url+'/v1/models', headers={'Authorization': 'Bearer '+key})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=15) as response:
        models = json.load(response)
    if a.model not in {m['id'] for m in models['data']}:
        raise RuntimeError('Requested model does not match /v1/models')
    ids = read_ids(HERE/('pilot_task_ids.txt' if a.pilot else 'dev_task_ids.txt'))
    if not a.pilot and set(ids) != set(load_task_ids('dev')):
        raise RuntimeError('Pinned dev57 list differs from installed AppWorld data')
    if not set(ids).issubset(load_task_ids('train' if a.pilot else 'dev')):
        raise RuntimeError('Task IDs missing from installed dataset')
    dataset = 'train' if a.pilot else 'dev'
    prefix = 'reproduction/portable/'+a.run_name
    out = root/'experiments/outputs'/prefix
    if out.exists() and not a.check_only:
        raise FileExistsError(f'Use a new run-name; outputs already exist: {out}')
    config = (HERE/'experiment.jsonnet').read_text().replace('"MODEL_NAME"', json.dumps(a.model))
    config = config.replace('"dataset": "dev"', f'"dataset": "{dataset}"').replace('"max_steps": 50', f'"max_steps": {a.max_steps}')
    config_name = 'simplified_function_calling_agent/local/life_harness_repro/'+a.run_name
    config_path = root/'experiments/configs'/f'{config_name}.jsonnet'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and config_path.read_text() != config:
        raise RuntimeError('Run-name already has different configuration')
    config_path.write_text(config)
    env = dict(os.environ)
    for k in ['BASELINE', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
        env.pop(k, None)
    env.update(APPWORLD_ROOT=str(root), MODEL_SERVER_URL=url, NO_API_KEY=key, OPENAI_API_KEY=key,
               PYTHONPATH=str(ADAPTER/'src')+os.pathsep+str(root/'experiments'), NO_PROXY='*', no_proxy='*')
    # extract_runner_config resolves environment placeholders inside model_config.
    os.environ.update(APPWORLD_ROOT=str(root), MODEL_SERVER_URL=url, NO_API_KEY=key)
    extract_runner_config(config_name)
    if a.check_only:
        print(f'Preflight OK: model={a.model}, tasks={len(ids)}, config={config_name}')
        return
    out.mkdir(parents=True)
    (out/'task_ids.txt').write_text('\n'.join(ids)+'\n')
    manifest = {'model': a.model, 'server_url': url, 'dataset': dataset, 'task_count': len(ids),
                'pilot': a.pilot, 'max_steps': a.max_steps, 'thinking': a.thinking, 'temperature': 1,
                'request_seed': 100, 'timeout': 14400, 'predictor_max_completion_tokens': 3000,
                'main_max_completion_tokens': None, 'workers': min(a.num_processes, len(ids)),
                'policy': 'v009_r4', 'config_sha256': hashlib.sha256(config.encode()).hexdigest(),
                'appworld_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}
    save(out/'manifest.json', manifest)
    results = {}
    try:
        for condition in a.conditions:
            save(out/'status.json', {'state':'running', 'condition':condition})
            logs = out/'run_logs'/condition
            logs.mkdir(parents=True)
            workers, handles = [], []
            try:
                for index in range(manifest['workers']):
                    cmd = [sys.executable, '-m', 'life_harness_appworld.runner', '--appworld-root', str(root),
                           '--policy-directory', str(ADAPTER/'policies/v009_r4'), '--base-experiment', config_name,
                           '--model-name', a.model, '--temperature', '1.0', '--api-predictor-temperature', '1.0',
                           '--task-ids-file', str(out/'task_ids.txt'), '--experiment-name', prefix+'/'+condition,
                           '--num-processes', str(manifest['workers']), '--process-index', str(index)]
                    if condition == 'baseline':
                        cmd.append('--baseline')
                    if a.thinking == 'off':
                        cmd.append('--qwen-disable-thinking')
                    log = (logs/f'process_{index}.log').open('w')
                    handles.append(log)
                    workers.append(subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                    start_new_session=True))
                codes = [worker.wait() for worker in workers]
                if any(codes):
                    raise RuntimeError(f'Worker exit codes: {codes}; inspect {logs}')
            finally:
                for worker in workers:
                    if worker.poll() is None:
                        os.killpg(worker.pid, signal.SIGTERM)
                        try:
                            worker.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(worker.pid, signal.SIGKILL)
                            worker.wait()
                for log in handles:
                    log.close()
            audited = audit(out/condition/'tasks', ids, a.model, condition, a.thinking)
            save(out/f'{condition}_audit.json', audited)
            if a.pilot:
                if audited['model_response_errors'] or audited['format_or_execution_issues']:
                    raise RuntimeError(f'{condition} pilot compatibility checks failed; inspect its audit')
                results[condition] = {'pilot_compatibility_checks_passed': True, 'tasks': len(ids)}
            else:
                value = evaluate_tasks(ids, experiment_name=prefix+'/'+condition, suppress_errors=True)
                assert set(value['individual']) == set(ids)
                save(out/f'{condition}_metrics.json', value)
                results[condition] = {'passed': sum(r['success'] for r in value['individual'].values()),
                                      'total': len(ids), **value['aggregate'],
                                      'model_response_errors':len(audited['model_response_errors'])}
            save(out/'summary.json', results)
        save(out/'status.json', {'state':'complete', 'finished_at':datetime.now(timezone.utc).isoformat()})
        print(json.dumps(results, indent=2))
        print(f'Outputs: {out}\nSGLang service is left running.')
    except BaseException as exc:
        save(out/'status.json', {'state':'failed', 'error':str(exc)})
        raise


if __name__ == '__main__':
    main()
