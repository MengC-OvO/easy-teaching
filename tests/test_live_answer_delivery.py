import asyncio
import json
from datetime import date
from datetime import datetime
from zoneinfo import ZoneInfo
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api.execution import _invoke_graph_with_progress, _node_progress
from app.api.routes.drafts import _draft_frames
from app.main import create_app
from app.schemas import MainDecision, TaskType
from app.tools import ToolExecutionContext, ToolRegistry
from app.tools.controlled_tools.daily_context import build_get_daily_context_tool
from app.tools.controlled_tools.check_activity_safety import build_check_activity_safety_tool
from tests.test_async_api import FakeRuntime
from tests.test_daily_context_tool import StubResponse
from tests.test_main_react_graph import _graph, _invoke


def test_weather_cache_reuses_concurrent_calls_expires_and_isolates_location(tmp_path):
    now = [0]
    class Store:
        async def get_centre_location(self, **scope):
            return dict(centre_id=scope['class_id'], suburb='Sydney', state='NSW',
                        latitude=-33, longitude=151 if scope['class_id'] == 'a' else 152)
    class Client:
        calls = 0
        async def get(self, *args, **kwargs):
            self.calls += 1
            await asyncio.sleep(0)
            return StubResponse({'daily': {'temperature_2m_max': [20], 'weather_code': [3]}})
    calendar = tmp_path / 'calendar.json'
    calendar.write_text('{"state":"NSW","holidays":{}}')
    client = Client()
    registry = ToolRegistry()
    registry.register(build_get_daily_context_tool(Store(), client=client,
        calendar_path=calendar, clock=lambda: now[0]))
    async def run(class_id='a', day='2026-09-10'):
        return await registry.execute_async('get_daily_context', {'target_date': day},
            execution_context=ToolExecutionContext(teacher_id='t', class_id=class_id))
    async def scenario():
        first, second = await asyncio.gather(run(), run())
        assert client.calls == 1
        assert sorted([first.data['cache_hit'], second.data['cache_hit']]) == [False, True]
        assert first.data['fetched_at'] == second.data['fetched_at']
        await run('b')
        await run(day='2026-09-11')
        assert client.calls == 3
        now[0] = 601
        assert not (await run()).data['cache_hit']
        assert client.calls == 4
    asyncio.run(scenario())


def test_final_candidate_is_checked_and_published_without_another_model_turn():
    registry = ToolRegistry()
    registry.register(build_check_activity_safety_tool())
    candidate = 'Sort large paper shapes by colour.\nEXACT FINAL TEXT'
    # Only one model decision is available; a second call would fail.
    state = _invoke(_graph([MainDecision(task_type=TaskType.ACTIVITY_PLAN,
        requires_activity_safety=True, artifact_title='Shape activity',
        reason='Complete candidate', final_answer=candidate)], registry=registry))
    assert state.draft.content == candidate
    assert state.draft.title == 'Shape activity'
    assert state.tool_attempt_counts['check_activity_safety'] == 1
    assert state.checked_final_candidate is None


def test_rejected_candidate_returns_to_model_for_revision():
    registry = ToolRegistry()
    registry.register(build_check_activity_safety_tool())
    decisions = [MainDecision(task_type=TaskType.ACTIVITY_PLAN, requires_activity_safety=True,
        reason='Candidate', final_answer=text) for text in
        ['Play with water.', 'Sort large paper shapes at the table.']]
    state = _invoke(_graph(decisions, registry=registry))
    assert state.draft.content == 'Sort large paper shapes at the table.'
    assert state.tool_attempt_counts['check_activity_safety'] == 2


def test_answer_chunks_resume_unicode_exactly_and_do_not_include_full_text_in_metadata():
    text = '中文😀\n' * 20
    payload = {'draft': {'content': text, 'title': 'Title'}, 'citations': []}
    class Request:
        async def is_disconnected(self): return False
    async def collect(offset):
        return [frame async for frame in _draft_frames(payload, Request(), offset)]
    frames = asyncio.run(collect(24))
    data = [json.loads(frame.split('data: ', 1)[1].strip()) for frame in frames]
    assert data[0]['draft']['content'] == ''
    assert ''.join(item['text'] for item in data if 'text' in item) == text[24:]
    assert data[-2]['offset'] == len(text)


def test_draft_stream_reuses_session_scope_and_ready_checks():
    runtime = FakeRuntime()
    with TestClient(create_app(runtime_factory=lambda: runtime)) as client:
        sid = client.post('/sessions', json={'teacher_id':'teacher-1','class_id':'kangaroo-room'}).json()['session_id']
        client.post(f'/sessions/{sid}/messages', json={'message':'Synthetic draft','request_id':'stream-test'})
        url = f'/sessions/{sid}/drafts/stream-test/stream'
        response = client.get(url)
        assert response.status_code == 200
        assert 'event: answer_delta' in response.text
        assert 'event: answer_done' in response.text
        other = client.post('/sessions', json={'teacher_id':'teacher-1','class_id':'kangaroo-room'}).json()['session_id']
        assert client.get(f'/sessions/{other}/drafts/stream-test/stream').status_code == 404
        runtime.store.results.clear()
        runtime.store.runs['stream-test']['status'] = 'running'
        assert client.get(url).status_code == 409


def test_inline_progress_is_persisted_before_next_node_runs():
    events = []
    class Store:
        async def append_conversation_event(self, **event): events.append(event)
    class Graph:
        async def astream(self, *args, **kwargs):
            yield {'initialize': {}}
            assert events, 'First event must be available before graph completion'
            yield {'single_tool': {'private_model_output': 'secret'}}
        async def aget_state(self, config): return SimpleNamespace(values={'request_id':'r'})
    asyncio.run(_invoke_graph_with_progress(runtime=SimpleNamespace(graph=Graph(), store=Store()),
        graph_input={}, config={}, request_id='r', session_id='s'))
    assert len(events) == 2
    assert 'secret' not in str(events)


def test_tool_progress_reports_cache_without_exposing_tool_arguments():
    data = _node_progress('single_tool', {'observations': {'weather': {
        'capability_name':'get_daily_context', 'status':'completed',
        'data': {'cache_hit':True, 'private_text':'secret'}}}})
    assert '10分钟' in data['message']
    assert 'secret' not in str(data)


def test_today_is_resolved_by_backend_and_failed_weather_is_not_cached(tmp_path):
    class Store:
        async def get_centre_location(self, **scope):
            return dict(centre_id='c', suburb='Sydney', state='NSW', latitude=-33,
                        longitude=151, timezone='Australia/Sydney')
    class Client:
        calls = 0
        async def get(self, url, *, params, timeout):
            self.calls += 1
            assert params['start_date'] == datetime.now(ZoneInfo('Australia/Sydney')).date().isoformat()
            if self.calls == 1:
                raise RuntimeError('temporary network failure')
            return StubResponse({'daily': {'temperature_2m_max': [20]}})
    calendar = tmp_path / 'calendar.json'
    calendar.write_text('{"state":"NSW","holidays":{}}')
    client = Client()
    registry = ToolRegistry()
    registry.register(build_get_daily_context_tool(Store(), client=client, calendar_path=calendar))
    async def scenario():
        scope = ToolExecutionContext(teacher_id='t', class_id='c')
        first = await registry.execute_async('get_daily_context', {}, execution_context=scope)
        assert not first.success
        second = await registry.execute_async('get_daily_context', {}, execution_context=scope)
        assert second.success and not second.data['cache_hit']
        third = await registry.execute_async('get_daily_context', {}, execution_context=scope)
        assert third.data['cache_hit']
        assert client.calls == 2
    asyncio.run(scenario())
