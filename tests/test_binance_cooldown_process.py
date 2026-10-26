"""Real spawned processes share only temporary cooldown files; no network."""
from pathlib import Path
from datetime import datetime, timezone
from email.utils import format_datetime
import json
import multiprocessing
import sys
import time

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import binance_terminal_client as bc
from process_lock import exclusive_process_lock

BASE_URL = 'https://example.invalid'


def _seed(path, until):
    path.write_text(json.dumps({'base_url': BASE_URL, 'retry_at_ms': until,
                                'reason': 'offline fixture'}) + '\n', encoding='utf-8')


def _writer(path, until, results, read_ready=None, release_read=None, wait_read=None,
            attempting=None):
    """The read hook exposes a lost-update race without replacing real locks."""
    import binance_terminal_client as child_bc
    child_bc.COOLDOWN_PATH = Path(path)
    network_calls = []

    def no_network(*args, **kwargs):
        network_calls.append(True)
        raise AssertionError('Network must remain disabled in cooldown tests')

    child_bc.requests.sessions.Session.request = no_network
    client = child_bc.BinanceTerminalClient()
    client.base_url = BASE_URL
    client._public_diagnostics = False
    client._curl_public_get = no_network
    original_read = client.cooldown_until_ms
    if read_ready is not None:
        def pause_after_read():
            current = original_read()
            read_ready.set()
            if not release_read.wait(10):
                raise TimeoutError('Parent did not release the test read hook')
            return current
        client.cooldown_until_ms = pause_after_read
    if wait_read is not None and not wait_read.wait(10):
        results.put({'error_type': 'test_peer_not_ready'})
        return
    if attempting is not None:
        attempting.set()
    began = time.monotonic()
    result = {}
    try:
        result['returned_until'] = client.record_cooldown(f'banned until {until}')
    except Exception as exc:
        result['error_type'] = type(exc).__name__
    result['record_seconds'] = time.monotonic() - began
    try:
        client.public_get('/fapi/v1/time')
        result['public_get'] = 'unexpected_success'
    except child_bc.BinanceApiError as exc:
        result['public_get'] = 'paused'
        result['public_retry_at_ms'] = exc.retry_at_ms
    except Exception as exc:
        result['public_get'] = type(exc).__name__
    result['network_calls'] = len(network_calls)
    results.put(result)
    client.session.close()


def _finish(processes):
    for process in processes:
        process.join(3)
        if process.is_alive():
            process.terminate()
            process.join(3)


def test_spawned_short_writer_cannot_overwrite_a_concurrent_longer_cooldown(tmp_path):
    context = multiprocessing.get_context('spawn')
    path = tmp_path / 'cooldown.json'
    now = int(time.time() * 1000)
    seed, short, longest = now + 180_000, now + 300_000, now + 600_000
    _seed(path, seed)
    results = context.Queue()
    read_ready, release_read, attempting = context.Event(), context.Event(), context.Event()
    small = context.Process(target=_writer, args=(str(path), short, results, read_ready, release_read))
    large = context.Process(target=_writer, args=(str(path), longest, results),
                            kwargs={'wait_read': read_ready, 'attempting': attempting})
    processes = [small, large]
    observed = [seed]
    try:
        for process in processes:
            process.start()
        assert read_ready.wait(10), 'Short writer must enter its real read/merge/write transaction'
        assert attempting.wait(10), 'Long writer must start while short writer is inside that transaction'
        # A proper process lock keeps the large write pending. Without one,
        # large commits now and the paused short writer then rolls it back.
        time.sleep(.15)
        release_read.set()
        deadline = time.monotonic() + 10
        while any(process.is_alive() for process in processes):
            assert time.monotonic() < deadline, 'Bounded write transaction did not complete'
            payload = json.loads(path.read_text(encoding='utf-8'))
            assert payload['base_url'] == BASE_URL
            observed.append(payload['retry_at_ms'])
            time.sleep(.005)
        replies = [results.get(timeout=3) for _ in processes]
        assert all('error_type' not in result for result in replies), replies
        assert all(result['network_calls'] == 0 and result['public_get'] == 'paused' for result in replies)
        final = json.loads(path.read_text(encoding='utf-8'))
        assert final['retry_at_ms'] == longest + 5000
        observed.append(final['retry_at_ms'])
        assert observed == sorted(observed), 'A published cooldown must never move backward'
        assert all(process.exitcode == 0 for process in processes)
        assert not list(tmp_path.glob('cooldown.json.*.tmp'))
    finally:
        release_read.set()
        _finish(processes)
        results.close()


def test_busy_process_lock_has_bounded_failure_and_existing_ban_still_blocks_network(tmp_path):
    context = multiprocessing.get_context('spawn')
    path = tmp_path / 'cooldown.json'
    now = int(time.time() * 1000)
    longest = now + 600_000
    _seed(path, longest)
    before = path.read_bytes()
    results = context.Queue()
    process = context.Process(target=_writer, args=(str(path), now + 120_000, results))
    try:
        with exclusive_process_lock(path.with_name(path.name + '.lock')):
            process.start()
            result = results.get(timeout=12)
            process.join(3)
            assert 'error_type' in result, 'A busy interprocess lock must not be silently bypassed'
            assert result['record_seconds'] < 2, 'Lock contention must have a finite approximately-one-second budget'
            assert result['public_get'] == 'paused' and result['public_retry_at_ms'] == longest
            assert result['network_calls'] == 0
            assert path.read_bytes() == before
        assert process.exitcode == 0
        assert not list(tmp_path.glob('cooldown.json.*.tmp'))
    finally:
        _finish([process])
        results.close()


def test_failed_atomic_replace_propagates_and_leaves_existing_cooldown_bytes_unchanged(tmp_path, monkeypatch):
    path = tmp_path / 'cooldown.json'
    now = int(time.time() * 1000)
    _seed(path, now + 300_000)
    before = path.read_bytes()
    monkeypatch.setattr(bc, 'COOLDOWN_PATH', path)
    client = bc.BinanceTerminalClient()
    client.base_url = BASE_URL
    client._public_diagnostics = False
    original_replace = Path.replace

    def fail_cooldown_replace(self, destination):
        if Path(destination) == path:
            raise OSError('Offline atomic replace failure')
        return original_replace(self, destination)

    monkeypatch.setattr(Path, 'replace', fail_cooldown_replace)
    try:
        with pytest.raises(OSError, match='Offline atomic replace failure'):
            client.record_cooldown(f'banned until {now + 600_000}')
        assert path.read_bytes() == before
        assert not list(tmp_path.glob('cooldown.json.*.tmp'))
    finally:
        client.session.close()


@pytest.mark.parametrize('header_offset_seconds', [-60, 600, 7200])
def test_http_date_retry_after_merges_with_existing_longer_ban_without_shortening(tmp_path, monkeypatch, header_offset_seconds):
    path = tmp_path / 'cooldown.json'
    now = int(time.time()) * 1000
    existing = now + 3600_000
    _seed(path, existing)
    monkeypatch.setattr(bc, 'COOLDOWN_PATH', path)
    client = bc.BinanceTerminalClient()
    client.base_url = BASE_URL
    client._public_diagnostics = False
    header_until = now + header_offset_seconds * 1000
    response = bc.requests.Response()
    response.status_code = 429
    response._content = b'{"code":-1003,"msg":"Too many requests"}'
    response.headers['Retry-After'] = format_datetime(
        datetime.fromtimestamp(header_until / 1000, timezone.utc), usegmt=True)
    try:
        with pytest.raises(bc.BinanceApiError) as error:
            client._decode_response(response)
        expected = max(existing, header_until + 5000)
        assert error.value.retry_at_ms == expected
        assert json.loads(path.read_text(encoding='utf-8'))['retry_at_ms'] == expected
        assert not list(tmp_path.glob('cooldown.json.*.tmp'))
    finally:
        response.close()
        client.session.close()
