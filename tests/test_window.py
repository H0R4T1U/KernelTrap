from central_server.window_v3 import SlidingWindowTracker


def _make_tracker(**overrides):
    defaults = dict(pivot_threshold=3, window_seconds=60, min_sev2_rate=0.30)
    defaults.update(overrides)
    return SlidingWindowTracker(**defaults)


def test_no_pivot_below_threshold():
    tracker = _make_tracker()
    for _ in range(2):
        tracker.feed("host-a", user_id=1500, severity=2)
    assert tracker.drain_pivots() == []


def test_pivot_fires_when_count_and_rate_satisfied():
    tracker = _make_tracker()
    for _ in range(3):
        tracker.feed("host-a", user_id=1500, severity=2)

    pivots = tracker.drain_pivots()
    assert len(pivots) == 1

    hostname, user_id, trigger = pivots[0]
    assert hostname == "host-a"
    assert user_id == 1500
    assert "sev2=3/3" in trigger


def test_rate_gate_blocks_pivot():
    tracker = _make_tracker()
    for _ in range(20):
        tracker.feed("host-a", user_id=1500, severity=0)
    for _ in range(3):
        tracker.feed("host-a", user_id=1500, severity=2)

    assert tracker.drain_pivots() == []


def test_whitelist_uid_ignored():
    tracker = _make_tracker(whitelist_uids={0})
    for _ in range(5):
        tracker.feed("host-a", user_id=0, severity=2)

    assert tracker.drain_pivots() == []


def test_post_pivot_suppression():
    tracker = _make_tracker()
    for _ in range(3):
        tracker.feed("host-a", user_id=1500, severity=2)

    assert len(tracker.drain_pivots()) == 1

    for _ in range(5):
        tracker.feed("host-a", user_id=1500, severity=2)

    assert tracker.drain_pivots() == []


def test_re_queue_pivots_round_trips():
    tracker = _make_tracker()
    tracker.re_queue_pivots([("host-a", 1500, "manual")])
    assert tracker.drain_pivots() == [("host-a", 1500, "manual")]
    assert tracker.drain_pivots() == []  # drain clears the queue


def test_stats_reports_counts():
    tracker = _make_tracker(pivot_threshold=3)
    for _ in range(3):
        tracker.feed("host-a", user_id=1500, severity=2)
    s = tracker.stats()
    assert s["pivot_threshold"] == 3
    assert s["pivoted"] == 1
    assert s["tracked_entities"] >= 1


def test_user_states_reports_window():
    tracker = _make_tracker()
    tracker.feed("host-a", user_id=1500, severity=2)
    tracker.feed("host-a", user_id=1500, severity=0)
    states = tracker.user_states()
    assert len(states) == 1
    st = states[0]
    assert st["hostname"] == "host-a"
    assert st["user_id"] == 1500
    assert st["window"]["total_count"] == 2
    assert st["window"]["severity2_count"] == 1


def test_events_cleared_after_pivot():
    tracker = _make_tracker(pivot_threshold=3)
    for _ in range(3):
        tracker.feed("host-a", user_id=1500, severity=2)
    tracker.drain_pivots()
    # Pivoted window keeps its entry (for visibility) but frees the deque.
    st = {(s["hostname"], s["user_id"]): s for s in tracker.user_states()}
    entry = st[("host-a", 1500)]
    assert entry["pivoted"] is True
    assert entry["window"]["total_count"] == 0
