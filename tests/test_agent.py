"""Offline SDK boundary tests; no model calls or credentials are needed."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from temporal_memory_os import MemoryStore, cli
from temporal_memory_os.agent import MEMORY_TOOLS, ask


@pytest.fixture
def memory(tmp_path):
    with MemoryStore(tmp_path) as store:
        store.add('Acme', 'status', 'pending', valid_from='2026-01-01')
    return tmp_path


def result(sdk, **overrides):
    fields = dict(subtype='success', duration_ms=1, duration_api_ms=1, is_error=False,
                  num_turns=2, session_id='test-session', total_cost_usd=0.01,
                  result='Pending [fact-id, synthetic source]')
    fields.update(overrides)
    return sdk.ResultMessage(**fields)


def test_sdk_owns_loop_and_only_memory_tools_are_enabled(memory, monkeypatch):
    sdk = pytest.importorskip('claude_agent_sdk')
    seen = {}

    async def query(*, prompt, options):
        seen.update(prompt=json.loads(prompt), options=options)
        assert Path(options.cwd).is_dir()
        yield result(sdk)

    monkeypatch.setattr(sdk, 'query', query)
    answer = asyncio.run(ask(memory, 'Acme status?', as_of='2026-01-02',
                             known_at='2026-10-29', model='test-model'))
    opts = seen['options']
    assert opts.tools == []
    assert opts.allowed_tools == MEMORY_TOOLS
    assert opts.permission_mode == 'dontAsk'
    assert opts.setting_sources == [] and opts.strict_mcp_config
    assert opts.skills == []
    assert opts.extra_args == {'no-session-persistence': None}
    assert opts.max_turns == 8 and opts.max_budget_usd == 0.5
    assert opts.model == 'test-model'
    assert opts.stderr('private-provider-diagnostic') is None
    assert not Path(opts.cwd).exists()
    server = opts.mcp_servers['memory']
    assert server['command'] == sys.executable
    assert server['args'] == ['-m', 'temporal_memory_os', '--root', str(memory), 'mcp']
    assert Path(server['env']['PYTHONPATH'], 'temporal_memory_os').is_dir()
    assert seen['prompt']['question'] == 'Acme status?'
    assert answer['as_of'] == '2026-01-02T00:00:00.000000Z'
    assert answer['known_at'] == seen['prompt']['known_at']
    assert answer['num_turns'] == 2 and answer['total_cost_usd'] == 0.01


def test_default_dates_are_frozen_together(memory, monkeypatch):
    sdk = pytest.importorskip('claude_agent_sdk')

    async def query(**kwargs):
        yield result(sdk)

    monkeypatch.setattr(sdk, 'query', query)
    answer = asyncio.run(ask(memory, 'status'))
    assert answer['as_of'] == answer['known_at']


@pytest.mark.parametrize('kwargs', [
    {'question': ''}, {'question': '  '}, {'question': '\ud800'},
    {'max_turns': 0}, {'max_turns': True}, {'max_turns': 1.5},
    {'max_budget_usd': float('nan')}, {'max_budget_usd': -1},
    {'max_budget_usd': 10 ** 1000}, {'timeout': float('inf')}, {'timeout': False},
    {'timeout': '3'}, {'model': ''}, {'model': 1}, {'as_of': 'wrong'},
])
def test_invalid_inputs_fail_before_sdk(memory, kwargs):
    values = {'question': 'status', **kwargs}
    with pytest.raises(ValueError):
        asyncio.run(ask(memory, **values))


def test_missing_store_does_not_start_sdk(tmp_path):
    with pytest.raises(ValueError, match='init or demo'):
        asyncio.run(ask(tmp_path, 'status'))
    assert list(tmp_path.iterdir()) == []


def test_missing_extra_gives_install_hint(memory, monkeypatch):
    monkeypatch.setitem(sys.modules, 'claude_agent_sdk', None)
    with pytest.raises(RuntimeError, match=r"pip install -e '.\[agent\]'"):
        asyncio.run(ask(memory, 'status'))


@pytest.mark.parametrize('overrides', [
    {'is_error': True}, {'subtype': 'error_max_turns'}, {'subtype': 'error_max_budget_usd'},
    {'result': None}, {'result': '  '},
])
def test_unsuccessful_sdk_result_is_not_an_answer(memory, monkeypatch, overrides):
    sdk = pytest.importorskip('claude_agent_sdk')

    async def query(**kwargs):
        yield result(sdk, **overrides)

    monkeypatch.setattr(sdk, 'query', query)
    with pytest.raises(RuntimeError):
        asyncio.run(ask(memory, 'status'))


def test_empty_stream_fails(memory, monkeypatch):
    sdk = pytest.importorskip('claude_agent_sdk')

    async def query(**kwargs):
        if False:
            yield

    monkeypatch.setattr(sdk, 'query', query)
    with pytest.raises(RuntimeError, match='without a final result'):
        asyncio.run(ask(memory, 'status'))


def test_timeout_cancels_and_cleans_up(memory, monkeypatch):
    sdk = pytest.importorskip('claude_agent_sdk')
    seen = {}

    async def query(**kwargs):
        seen['cwd'] = kwargs['options'].cwd
        try:
            await asyncio.sleep(100)
            yield result(sdk)
        finally:
            seen['closed'] = True

    monkeypatch.setattr(sdk, 'query', query)
    with pytest.raises(RuntimeError, match='timeout'):
        asyncio.run(ask(memory, 'status', timeout=0.01))
    assert seen['closed'] and not Path(seen['cwd']).exists()


@pytest.mark.parametrize('error_type', [ValueError, RuntimeError])
def test_sdk_exception_does_not_expose_provider_body(memory, monkeypatch, error_type):
    sdk = pytest.importorskip('claude_agent_sdk')

    async def query(**kwargs):
        raise error_type('private-provider-response')
        yield

    monkeypatch.setattr(sdk, 'query', query)
    with pytest.raises(RuntimeError) as error:
        asyncio.run(ask(memory, 'status'))
    assert 'private-provider-response' not in str(error.value)


def test_cli_passes_agent_options_and_outputs_json(memory, monkeypatch, capsys):
    from temporal_memory_os import agent

    async def fake_ask(root, question, **kwargs):
        assert root == memory and question == 'status'
        assert kwargs == dict(as_of='2026-01-01', known_at=None, model='test-model',
                              max_turns=3, max_budget_usd=0.1, timeout=20)
        return {'answer': 'pending'}

    monkeypatch.setattr(agent, 'ask', fake_ask)
    assert cli.main(['--root', str(memory), 'agent', 'status', '--as-of', '2026-01-01',
                     '--model', 'test-model', '--max-turns', '3', '--max-budget-usd', '0.1',
                     '--timeout', '20']) == 0
    assert json.loads(capsys.readouterr().out) == {'answer': 'pending'}
