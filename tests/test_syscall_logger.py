import json

from masina_invata.logger.syscall_logger import (
    SYSCALL_TO_ID,
    SyscallEvent,
    SyscallLogger,
    TraceeParser,
)


# ---------------------------------------------------------------------------
# TraceeParser
# ---------------------------------------------------------------------------

def test_tracee_parser_basic():
    parser = TraceeParser(hostname="h1")
    line = json.dumps({
        "eventName": "security_socket_connect",
        "timestamp": 1234.5,
        "processId": 4242,
        "threadId": 4242,
        "parentProcessId": 1,
        "userId": 1000,
        "mountNamespace": 4026531840,
        "processName": "python3",
        "returnValue": 0,
        "args": [{"name": "remote_addr", "type": "struct", "value": "127.0.0.1:6379"}],
    })
    ev = parser.parse_line(line)
    assert ev is not None
    assert ev.eventName == "security_socket_connect"
    assert ev.eventId == SYSCALL_TO_ID["security_socket_connect"]
    assert ev.processId == 4242
    assert ev.userId == 1000
    assert ev.processName == "python3"
    assert ev.argsNum == 1


def test_tracee_parser_ignores_blank_and_garbage():
    parser = TraceeParser(hostname="h1")
    assert parser.parse_line("") is None
    assert parser.parse_line("   ") is None
    assert parser.parse_line("not json") is None


# ---------------------------------------------------------------------------
# Filtering in _process_event (redis-forwarding path)
# ---------------------------------------------------------------------------

def _logger_forwarding():
    """A logger wired as if Redis-forwarding is on, but with no real connection."""
    lg = SyscallLogger(source="tracee", output_path=None)
    lg._redis_publisher = object()  # truthy sentinel — _process_event forwards
    lg._redis_batch = []
    return lg


def _event(**kw):
    base = dict(
        timestamp=1.0, processId=500, parentProcessId=1, userId=1000,
        mountNamespace=0, processName="bash", eventId=59, eventName="execve",
    )
    base.update(kw)
    return SyscallEvent(**base)


def test_redis_backchannel_drops_loopback_redis_connect():
    lg = _logger_forwarding()
    lg._redis_port = 6379
    lg._redis_addrs = {"localhost", "127.0.0.1"}
    ev = _event(
        processName="python3",
        eventId=SYSCALL_TO_ID["security_socket_connect"],
        eventName="security_socket_connect",
        args=json.dumps([{"name": "remote_addr", "value": "127.0.0.1:6379"}]),
        argsNum=1,
    )
    assert lg._is_redis_backchannel(ev) is True


def test_redis_backchannel_keeps_unrelated_remote_connect():
    lg = _logger_forwarding()
    lg._redis_port = 6379
    lg._redis_addrs = {"10.0.0.5"}
    ev = _event(
        processName="curl",
        eventId=SYSCALL_TO_ID["security_socket_connect"],
        eventName="security_socket_connect",
        args=json.dumps([{"name": "remote_addr", "value": "93.184.216.34:443"}]),
        argsNum=1,
    )
    assert lg._is_redis_backchannel(ev) is False


def test_redis_backchannel_drops_loopback_when_agent_uses_lan_ip():
    # All-in-one box: agent started with the LAN IP, but the server's uvicorn
    # connects to 127.0.0.1:6379 — the connect args show loopback, not the LAN IP.
    lg = _logger_forwarding()
    lg._redis_port = 6379
    lg._redis_addrs = {"10.0.0.5"}
    ev = _event(
        userId=0, processName="python3",
        eventId=SYSCALL_TO_ID["security_socket_connect"],
        eventName="security_socket_connect",
        args=json.dumps([{"name": "remote_addr",
                          "value": "{'sa_family': 'AF_INET', 'sin_port': '6379', 'sin_addr': '127.0.0.1'}"}]),
        argsNum=1,
    )
    assert lg._is_redis_backchannel(ev) is True


def test_redis_backchannel_ignores_non_connect_events():
    lg = _logger_forwarding()
    lg._redis_port = 6379
    lg._redis_addrs = {"127.0.0.1"}
    ev = _event(args=json.dumps([{"name": "x", "value": "6379"}]), argsNum=1)
    assert lg._is_redis_backchannel(ev) is False


def test_ssh_filter_drops_uid_without_session():
    lg = _logger_forwarding()
    lg._ssh_uids = {1000}
    lg._process_event(_event(userId=1234), None)
    assert lg._redis_batch == []


def test_ssh_filter_keeps_uid_with_session():
    lg = _logger_forwarding()
    lg._ssh_uids = {1000}
    lg._process_event(_event(userId=1000), None)
    assert len(lg._redis_batch) == 1


def test_ssh_filter_fail_closed_when_no_sessions():
    lg = _logger_forwarding()
    lg._ssh_uids = set()  # no SSH sessions -> fail-closed, forward nothing
    lg._process_event(_event(userId=1234), None)
    assert lg._redis_batch == []


def test_self_pid_excluded():
    lg = _logger_forwarding()
    lg._ssh_uids = set()
    lg._process_event(_event(processId=lg._own_pid, processName="python3"), None)
    assert lg._redis_batch == []
