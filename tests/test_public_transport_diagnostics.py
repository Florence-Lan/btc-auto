"""Safe optional public diagnostics; no response bodies, routes or network I/O."""
import copy
import json
from pathlib import Path
import sys
import traceback
from decimal import Decimal

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from public_transport import (
    INVALID_PUBLIC_RESPONSE_REASONS, MAX_DIAGNOSTIC_ELAPSED_MS,
    InvalidPublicResponse, PublicCurlError, PublicTransportError, safe_transport_error,
)

SECRET = 'https://proxy-user:private-password@private.invalid/?api_key=private-key'


class HostileValue:
    def __str__(self):
        raise AssertionError('An untrusted diagnostic must never be rendered')

    def __repr__(self):
        raise AssertionError('An untrusted diagnostic must never be rendered')

    def __eq__(self, other):
        raise AssertionError('An untrusted diagnostic must never be compared')


class HostileString(str):
    def __str__(self):
        raise AssertionError('String subclasses are not trusted enum values')


def failure(**changes):
    return {'transport': 'requests', 'error_type': 'InvalidPublicResponse',
            'error_chain': ['InvalidPublicResponse'], 'category': 'invalid_response', **changes}


@pytest.mark.parametrize('reason', sorted(INVALID_PUBLIC_RESPONSE_REASONS))
def test_known_reason_keeps_compatible_category_and_safe_numeric_elapsed(reason):
    error = InvalidPublicResponse(SECRET, reason_code=reason, elapsed_ms=5123.25)
    diagnostic = safe_transport_error(error)
    assert diagnostic == {'error_type': 'InvalidPublicResponse', 'error_chain': ['InvalidPublicResponse'],
                          'category': 'invalid_response', 'reason_code': reason, 'elapsed_ms': 5123.25}
    aggregate = PublicTransportError('Both public transports failed', [{'transport': 'requests', **diagnostic}])
    assert f'reason={reason}' in str(aggregate)
    assert 'elapsed_ms=5123.25' in str(aggregate)
    assert SECRET not in str(aggregate) + json.dumps(aggregate.failures)


@pytest.mark.parametrize('reason', [None, '', 'not_declared', SECRET, 'clock_timing_invalid\n' + SECRET,
                                    {'reason': SECRET}, [SECRET], 1, True, HostileValue(), HostileString('response_null')])
def test_untrusted_reason_ignored_at_exception_and_aggregate_boundaries(reason):
    error = InvalidPublicResponse(SECRET, reason_code=reason)
    assert 'reason_code' not in safe_transport_error(error)
    aggregate = PublicTransportError('Both public transports failed', [failure(reason_code=reason)])
    assert 'reason_code' not in aggregate.failures[0]
    assert '(reason=' not in str(aggregate)
    assert 'private-password' not in str(aggregate) + json.dumps(aggregate.failures)


@pytest.mark.parametrize('elapsed', [-1, float('nan'), float('inf'), -float('inf'), True, False,
                                     '100', Decimal('100'), {'elapsed': SECRET}, HostileValue(),
                                     MAX_DIAGNOSTIC_ELAPSED_MS + 1, 10**1000])
def test_invalid_elapsed_not_logged_or_stringified(elapsed):
    error = InvalidPublicResponse(SECRET, elapsed_ms=elapsed)
    diagnostic = safe_transport_error(error)
    assert 'elapsed_ms' not in diagnostic
    aggregate = PublicTransportError('Both public transports failed', [failure(elapsed_ms=elapsed)])
    assert 'elapsed_ms' not in aggregate.failures[0]
    assert 'elapsed_ms=' not in str(aggregate)
    json.dumps(diagnostic, allow_nan=False)
    json.dumps(aggregate.failures, allow_nan=False)


@pytest.mark.parametrize('elapsed', [0, 0.0, .125, 5000, MAX_DIAGNOSTIC_ELAPSED_MS])
def test_finite_bounded_elapsed_is_retained_including_zero(elapsed):
    diagnostic = safe_transport_error(InvalidPublicResponse('invalid', elapsed_ms=elapsed))
    assert diagnostic['elapsed_ms'] == elapsed
    aggregate = PublicTransportError('failed', [failure(elapsed_ms=elapsed)])
    assert aggregate.failures[0]['elapsed_ms'] == elapsed
    assert 'elapsed_ms=' in str(aggregate)


def test_attributes_are_revalidated_even_if_modified_after_construction():
    error = InvalidPublicResponse(SECRET, reason_code='response_null', elapsed_ms=3)
    error.reason_code, error.elapsed_ms = HostileValue(), float('nan')
    diagnostic = safe_transport_error(error)
    assert 'reason_code' not in diagnostic and 'elapsed_ms' not in diagnostic


def test_nested_exception_secrets_remain_private_and_outer_reason_is_preserved():
    cause = OSError(10054, SECRET)
    proxy = requests.exceptions.ProxyError(cause)
    error = InvalidPublicResponse(SECRET, reason_code='clock_roundtrip_exceeded', elapsed_ms=5001)
    error.__cause__ = proxy
    diagnostic = safe_transport_error(error)
    assert diagnostic['reason_code'] == 'clock_roundtrip_exceeded'
    assert diagnostic['errno'] == [10054]
    assert diagnostic['error_chain'] == ['InvalidPublicResponse', 'ProxyError', type(cause).__name__]
    aggregate = PublicTransportError('Both public transports failed', [{'transport': 'requests', **diagnostic}])
    rendered = str(aggregate) + json.dumps(aggregate.failures) + ''.join(traceback.format_exception(aggregate))
    for secret in ('proxy-user', 'private-password', 'private.invalid', 'private-key'):
        assert secret not in rendered


def test_aggregate_drops_unknown_and_unsafe_failure_fields_without_mutating_caller():
    given = failure(reason_code='api_code_invalid', elapsed_ms=1.5, message=SECRET, url=SECRET,
                    raw_exception={'proxy': SECRET}, extra=HostileValue())
    expected = {k: v for k, v in given.items() if k not in {'message', 'url', 'raw_exception', 'extra'}}
    aggregate = PublicTransportError('failed', [given])
    assert aggregate.failures == [expected]
    assert given['message'] == SECRET and 'extra' in given
    assert 'private-password' not in json.dumps(aggregate.failures) + str(aggregate)


def test_malicious_existing_failure_values_do_not_bypass_new_safe_boundary():
    given = {'transport': SECRET, 'error_type': SECRET, 'category': SECRET,
             'error_chain': [SECRET, HostileValue()], 'errno': [SECRET, True, HostileValue()],
             'winerror': [SECRET], 'curl_exit_code': SECRET,
             'reason_code': HostileValue(), 'elapsed_ms': HostileValue()}
    aggregate = PublicTransportError('failed', [given])
    assert aggregate.failures == [{'transport': 'unknown', 'error_type': 'UnknownError',
                                   'category': 'request', 'error_chain': []}]
    assert 'private-password' not in str(aggregate) + json.dumps(aggregate.failures)


def test_old_exception_shape_and_aggregate_string_are_unchanged_without_new_fields():
    cause = OSError(10054, SECRET)
    error = requests.exceptions.ConnectionError(cause)
    diagnostic = safe_transport_error(error)
    assert diagnostic == {'error_type': 'ConnectionError', 'error_chain': ['ConnectionError', type(cause).__name__],
                          'category': 'connection', 'errno': [10054]}
    failures = [{'transport': 'requests', **diagnostic},
                {'transport': 'curl', **safe_transport_error(PublicCurlError(SECRET, curl_exit_code=28))}]
    before = copy.deepcopy(failures)
    aggregate = PublicTransportError('Both public transports failed', failures)
    assert aggregate.failures == before and failures == before
    assert str(aggregate) == ('Both public transports failed '
                             '[requests=ConnectionError:connection(errno 10054); '
                             'curl=PublicCurlError:timeout(curl 28)]')
    assert safe_transport_error(InvalidPublicResponse('old call')) == {
        'error_type': 'InvalidPublicResponse', 'error_chain': ['InvalidPublicResponse'], 'category': 'invalid_response'}


def test_request_exception_response_keyword_remains_supported():
    response = requests.Response()
    error = InvalidPublicResponse('invalid', response=response, reason_code='response_not_json')
    assert error.response is response
    assert safe_transport_error(error)['reason_code'] == 'response_not_json'


@pytest.mark.parametrize('scope', ['network', 'caller_wait'])
def test_elapsed_scope_is_explicit_and_does_not_change_failure_category(scope):
    error = InvalidPublicResponse(SECRET, reason_code='clock_roundtrip_exceeded',
                                  elapsed_ms=5000.0001, elapsed_scope=scope)
    diagnostic = safe_transport_error(error)
    assert diagnostic['elapsed_scope'] == scope
    assert diagnostic['elapsed_ms'] == 5000.0001
    aggregate = PublicTransportError('failed', [{'transport': 'requests', **diagnostic}])
    assert f'(elapsed_ms=5000.0001,elapsed_scope={scope})' in str(aggregate)
    assert aggregate.failures[0]['category'] == 'invalid_response'


@pytest.mark.parametrize('scope', [None, '', 'roundtrip', SECRET, HostileValue(),
                                   HostileString('network'), {'scope': SECRET}, True])
def test_unknown_elapsed_scope_is_ignored_without_rendering_it(scope):
    error = InvalidPublicResponse(SECRET, elapsed_ms=2, elapsed_scope=scope)
    assert 'elapsed_scope' not in safe_transport_error(error)
    aggregate = PublicTransportError('failed', [failure(elapsed_ms=2, elapsed_scope=scope)])
    assert 'elapsed_scope' not in aggregate.failures[0]
    assert 'private-password' not in str(aggregate) + json.dumps(aggregate.failures)


def test_scope_without_a_valid_elapsed_value_is_not_reported():
    error = InvalidPublicResponse(SECRET, elapsed_ms=float('nan'), elapsed_scope='network')
    diagnostic = safe_transport_error(error)
    assert 'elapsed_ms' not in diagnostic and 'elapsed_scope' not in diagnostic
    aggregate = PublicTransportError('failed', [failure(elapsed_ms=True, elapsed_scope='caller_wait')])
    assert 'elapsed_ms' not in aggregate.failures[0] and 'elapsed_scope' not in aggregate.failures[0]


def test_dynamic_exception_class_name_cannot_expose_url_or_credentials():
    untrusted_type = type('InvalidClock:' + SECRET, (requests.RequestException,), {})
    error = untrusted_type(SECRET)
    diagnostic = safe_transport_error(error)
    assert diagnostic['error_type'] == 'UnknownError'
    assert diagnostic['error_chain'] == ['UnknownError']
    aggregate = PublicTransportError('failed', [{'transport': 'requests', **diagnostic}])
    rendered = json.dumps(diagnostic) + json.dumps(aggregate.failures) + str(aggregate)
    for secret in ('https://', 'proxy-user', 'private-password', 'private.invalid', 'private-key'):
        assert secret not in rendered
