"""Unit tests for app.rate_limit.get_client_ip's trusted-proxy handling.

Regression coverage for a bug where FORWARDED_ALLOW_IPS="0.0.0.0/0" (set in
production to trust Cloud Run's front end) never actually matched, because
the old check did exact string membership instead of CIDR-aware matching.
Every request's peer IP fell back to Cloud Run's internal proxy address,
which is identical for all end users — collapsing every real client into a
single auth-failure rate-limit bucket and locking out the whole site on a
handful of unrelated expired-token failures.
"""

from unittest.mock import patch

from app.rate_limit import get_client_ip


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    def __init__(self, peer_ip, headers=None):
        self.client = _FakeClient(peer_ip)
        self.headers = headers or {}


def test_cidr_trusted_proxy_is_matched():
    request = _FakeRequest("169.254.1.1", {"x-forwarded-for": "203.0.113.7"})
    with patch("app.rate_limit.settings.FORWARDED_ALLOW_IPS", "0.0.0.0/0"):
        assert get_client_ip(request) == "203.0.113.7"


def test_untrusted_peer_falls_back_to_peer_ip():
    request = _FakeRequest("8.8.8.8", {"x-forwarded-for": "203.0.113.7"})
    with patch("app.rate_limit.settings.FORWARDED_ALLOW_IPS", "10.0.0.0/8"):
        assert get_client_ip(request) == "8.8.8.8"


def test_exact_ip_fallback_when_not_parseable_as_network():
    request = _FakeRequest("169.254.1.1", {"x-forwarded-for": "203.0.113.7"})
    with patch("app.rate_limit.settings.FORWARDED_ALLOW_IPS", "169.254.1.1"):
        assert get_client_ip(request) == "203.0.113.7"


def test_rightmost_forwarded_entry_is_used_to_resist_spoofing():
    # The trusted proxy appends what it saw to the end of any existing
    # header, so a client can set a fake leftmost entry but not the one
    # the proxy itself appends.
    request = _FakeRequest(
        "169.254.1.1",
        {"x-forwarded-for": "1.2.3.4, 203.0.113.7"},
    )
    with patch("app.rate_limit.settings.FORWARDED_ALLOW_IPS", "0.0.0.0/0"):
        assert get_client_ip(request) == "203.0.113.7"


def test_no_forwarded_header_falls_back_to_peer_ip():
    request = _FakeRequest("169.254.1.1")
    with patch("app.rate_limit.settings.FORWARDED_ALLOW_IPS", "0.0.0.0/0"):
        assert get_client_ip(request) == "169.254.1.1"
