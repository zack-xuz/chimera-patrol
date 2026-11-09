"""CooldownTable 单元测试（需求 3.9）。"""

import pytest

from chimera_patrol.rules.cooldown import CooldownTable


def test_first_trigger_is_accepted():
    table = CooldownTable(cooldown_s=5.0)
    assert table.try_accept("no_helmet", "track:1", 1_000) is True


def test_trigger_within_cooldown_is_rejected():
    table = CooldownTable(cooldown_s=5.0)
    assert table.try_accept("no_helmet", "track:1", 0) is True
    assert table.try_accept("no_helmet", "track:1", 4_999) is False


def test_trigger_exactly_at_cooldown_is_accepted():
    table = CooldownTable(cooldown_s=5.0)
    assert table.try_accept("no_helmet", "track:1", 0) is True
    assert table.try_accept("no_helmet", "track:1", 5_000) is True


def test_rejected_trigger_does_not_update_record():
    table = CooldownTable(cooldown_s=5.0)
    assert table.try_accept("no_helmet", "track:1", 0) is True
    assert table.try_accept("no_helmet", "track:1", 3_000) is False
    # 冷却基准仍为 0 ms 的接受时间，而不是被丢弃的 3000 ms。
    assert table.try_accept("no_helmet", "track:1", 5_000) is True


def test_keys_are_independent():
    table = CooldownTable(cooldown_s=5.0)
    assert table.try_accept("no_helmet", "track:1", 0) is True
    assert table.try_accept("no_helmet", "track:2", 100) is True
    assert table.try_accept("intrusion", "track:1", 100) is True


def test_zero_cooldown_accepts_every_trigger():
    table = CooldownTable(cooldown_s=0.0)
    assert table.try_accept("e", "k", 0) is True
    assert table.try_accept("e", "k", 0) is True


def test_earlier_timestamp_raises():
    table = CooldownTable(cooldown_s=5.0)
    assert table.try_accept("e", "k", 10_000) is True
    with pytest.raises(ValueError):
        table.try_accept("e", "k", 9_999)


def test_negative_cooldown_raises():
    with pytest.raises(ValueError):
        CooldownTable(cooldown_s=-1.0)


def test_reset_clears_records():
    table = CooldownTable(cooldown_s=5.0)
    assert table.try_accept("e", "k", 0) is True
    table.reset()
    assert table.try_accept("e", "k", 1) is True


def test_spot_first_trigger_is_accepted():
    table = CooldownTable(cooldown_s=10.0)
    assert table.try_accept_spot("smoke_fire", (0, 0, 10, 10), 0, 0.3) is True


def test_spot_overlapping_box_within_cooldown_is_rejected():
    table = CooldownTable(cooldown_s=10.0)
    assert table.try_accept_spot("smoke_fire", (0, 0, 10, 10), 0, 0.3) is True
    # IoU = 81 / 100 = 0.81 ≥ 0.3，视为同一点位。
    assert table.try_accept_spot("smoke_fire", (1, 1, 10, 10), 5_000, 0.3) is False


def test_spot_overlapping_box_at_cooldown_is_accepted():
    table = CooldownTable(cooldown_s=10.0)
    assert table.try_accept_spot("smoke_fire", (0, 0, 10, 10), 0, 0.3) is True
    assert table.try_accept_spot("smoke_fire", (1, 1, 10, 10), 10_000, 0.3) is True


def test_spot_below_iou_threshold_is_a_new_spot():
    table = CooldownTable(cooldown_s=10.0)
    assert table.try_accept_spot("smoke_fire", (0, 0, 10, 10), 0, 0.3) is True
    # 与第一个框无重叠，属于新点位，不受冷却限制。
    assert table.try_accept_spot("smoke_fire", (100, 100, 110, 110), 1, 0.3) is True
    # 仍与第一个点位重叠，但在冷却内，因此被丢弃。
    assert table.try_accept_spot("smoke_fire", (0, 0, 10, 10), 2, 0.3) is False


def test_spot_rejected_trigger_does_not_replace_stored_box():
    table = CooldownTable(cooldown_s=10.0)
    assert table.try_accept_spot("smoke_fire", (0, 0, 10, 10), 0, 0.5) is True
    # IoU with stored box = 0.81，命中点位但在冷却内，丢弃。
    assert table.try_accept_spot("smoke_fire", (1, 1, 10, 10), 1_000, 0.5) is False
    # C 与存储框 (0,0,10,10) 的 IoU = 0.6，与被丢弃的框 (1,1,10,10) 的 IoU ≈ 0.47。
    # 若被丢弃的框覆盖了存储框，C 会被当作新点位而接受；正确行为是命中原点位并丢弃。
    assert table.try_accept_spot("smoke_fire", (0, 0, 6, 10), 5_000, 0.5) is False


def test_spot_records_are_per_event_type():
    table = CooldownTable(cooldown_s=10.0)
    assert table.try_accept_spot("smoke_fire", (0, 0, 10, 10), 0, 0.3) is True
    assert table.try_accept_spot("other", (0, 0, 10, 10), 1, 0.3) is True


def test_spot_reset_clears_records():
    table = CooldownTable(cooldown_s=10.0)
    assert table.try_accept_spot("smoke_fire", (0, 0, 10, 10), 0, 0.3) is True
    table.reset()
    assert table.try_accept_spot("smoke_fire", (0, 0, 10, 10), 1, 0.3) is True


def test_spot_idle_over_ten_minutes_is_evicted_and_reaccepted():
    table = CooldownTable(cooldown_s=1.0)
    assert table.try_accept_spot('smoke_fire', (0, 0, 10, 10), 0, 0.3) is True
    # 11 分钟后：旧点位已被淘汰，同一框按新点位接受。
    assert table.try_accept_spot('smoke_fire', (0, 0, 10, 10), 11 * 60_000, 0.3) is True
    assert len(table._spots['smoke_fire']) == 1


def test_spot_within_ten_minutes_still_matches():
    table = CooldownTable(cooldown_s=1.0)
    assert table.try_accept_spot('smoke_fire', (0, 0, 10, 10), 0, 0.3) is True
    assert table.try_accept_spot('smoke_fire', (0, 0, 10, 10), 5 * 60_000, 0.3) is True
    assert len(table._spots['smoke_fire']) == 1


def test_spot_not_evicted_while_cooldown_longer_than_ten_minutes():
    table = CooldownTable(cooldown_s=3600.0)
    assert table.try_accept_spot('smoke_fire', (0, 0, 10, 10), 0, 0.3) is True
    # 11 分钟后仍在 1 小时冷却内，必须丢弃而不是被淘汰后放行。
    assert table.try_accept_spot('smoke_fire', (0, 0, 10, 10), 11 * 60_000, 0.3) is False


def test_keyed_entry_not_evicted_early_when_cooldown_is_long():
    table = CooldownTable(cooldown_s=3600.0)
    assert table.try_accept('no_helmet', 'track:1', 0) is True
    assert table.prune(11 * 60_000) == 0
    assert table.try_accept('no_helmet', 'track:1', 11 * 60_000) is False


def test_keyed_entry_evicted_after_horizon_when_cooldown_short():
    table = CooldownTable(cooldown_s=1.0)
    assert table.try_accept('no_helmet', 'track:1', 0) is True
    assert table.prune(11 * 60_000) == 1
