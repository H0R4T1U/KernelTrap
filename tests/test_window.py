from central_server.window_v3 import SlidingWindowTracker


def _make_tracker(**overrides):
    # pivot_threshold = VOLUME floor; conc_min_count = CONCENTRATION floor.
    defaults = dict(pivot_threshold=10, window_seconds=60,
                    min_sev2_rate=0.30, conc_min_count=5)
    defaults.update(overrides)
    return SlidingWindowTracker(**defaults)


def test_no_pivot_below_both_floors():
    # 4 sev-2 at 100% rate: below the volume floor (10) AND below the
    # concentration floor (5) -> no pivot.
    tracker = _make_tracker()
    for _ in range(4):
        tracker.feed("host-a", user_id=1500, severity=2)
    assert tracker.drain_pivots() == []


def test_hair_trigger_guard_single_event():
    # A lone sev-2 event is 1/1 = 100% rate but must NOT pivot: the
    # concentration floor blocks single-event windows (the OR hazard).
    tracker = _make_tracker()
    tracker.feed("host-a", user_id=1500, severity=2)
    assert tracker.drain_pivots() == []


def test_volume_branch_fires_on_count_despite_low_rate():
    # Loud recon (linpeas): 10 sev-2 diluted in 100 total = 10% rate, far
    # below MIN_SEV2_RATE, but the VOLUME branch fires on absolute count.
    tracker = _make_tracker()  # volume floor 10
    for _ in range(90):
        tracker.feed("host-a", user_id=1500, severity=0)
    for _ in range(10):
        tracker.feed("host-a", user_id=1500, severity=2)
    pivots = tracker.drain_pivots()
    assert len(pivots) == 1
    _, _, trigger = pivots[0]
    assert "volume" in trigger


def test_concentration_branch_fires_below_volume_floor():
    # Stealthy: 5 sev-2 out of 7 (71%) — below the volume floor (10) but at
    # the concentration floor (5) and above the rate (30%) -> pivot.
    tracker = _make_tracker()
    for _ in range(2):
        tracker.feed("host-a", user_id=1500, severity=0)
    for _ in range(5):
        tracker.feed("host-a", user_id=1500, severity=2)
    pivots = tracker.drain_pivots()
    assert len(pivots) == 1
    _, _, trigger = pivots[0]
    assert "concentration" in trigger


def test_concentration_rate_gate_blocks():
    # 5 sev-2 meets the concentration floor but 5/25 = 20% < 30% rate, and
    # 5 < 10 volume floor -> neither branch fires.
    tracker = _make_tracker()
    for _ in range(20):
        tracker.feed("host-a", user_id=1500, severity=0)
    for _ in range(5):
        tracker.feed("host-a", user_id=1500, severity=2)
    assert tracker.drain_pivots() == []


def test_whitelist_uid_ignored():
    tracker = _make_tracker(whitelist_uids={0})
    for _ in range(15):
        tracker.feed("host-a", user_id=0, severity=2)
    assert tracker.drain_pivots() == []


def test_post_pivot_suppression():
    tracker = _make_tracker()
    for _ in range(10):
        tracker.feed("host-a", user_id=1500, severity=2)
    assert len(tracker.drain_pivots()) == 1

    for _ in range(10):
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
    assert s["conc_min_count"] == 5
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
    assert st["window"]["conc_min_count"] == 5


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
