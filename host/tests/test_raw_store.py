import time

import pyarrow.parquet as pq

from g1rec.raw_store import RawEpisodeStore, flatten_payload, recover
from g1rec.sync import Sample


def test_flatten_payload_keeps_layout():
    values, layout = flatten_payload({"pressure": [[1, 2], [3, 4]], "lost": [0], "flag": True, "name": "x"})
    assert values == [1.0, 2.0, 3.0, 4.0, 0.0, 1.0]
    assert layout == "pressure:4,lost:1,flag:1"


def test_raw_rows_keep_all_timestamps_and_native_rate(tmp_path):
    store = RawEpisodeStore(tmp_path, 1, flush_s=0.05)
    for i in range(500):                                     # 500 tactile messages: none is dropped
        store.on_sample(Sample("tactile_left", {"pressure": [[float(i)] * 12]}, recv_ns=i * 2_000_000,
                               source_ns=i, seq=i))
    store.on_sample(Sample("neck", {"present_deg": [1.0, 2.0]}, recv_ns=5, source_ns=7, seq=1, capture_ns=6))
    store.on_sample(Sample("image", object(), recv_ns=9, seq=3, capture_ns=8))
    store.close()
    ep = tmp_path / "episode_000001"
    tactile = pq.read_table(ep / "tactile.parquet").to_pylist()
    assert len(tactile) == 500 and tactile[10]["seq"] == 10 and tactile[10]["recv_ns"] == 20_000_000
    assert tactile[3]["values"] == [3.0] * 12
    assert pq.read_table(ep / "camera_state.parquet").to_pylist()[0]["capture_ns"] == 6
    stamps = pq.read_table(ep / "timestamps.parquet").to_pylist()
    assert stamps[0]["topic"] == "image" and stamps[0]["values"] == []     # image pixels never go to raw
    assert not list(ep.glob("*.part-*"))


def test_crashed_episode_is_recoverable(tmp_path):
    store = RawEpisodeStore(tmp_path, 2, flush_s=0.05)
    for i in range(50):
        store.on_sample(Sample("arm_state", {"x": [1.0]}, recv_ns=i, seq=i))
    time.sleep(0.5)                                          # periodic flush wrote part files
    ep = tmp_path / "episode_000002"
    assert list(ep.glob("robot_state.part-*.parquet"))
    # simulate a crash: the process dies, close() is never called
    counts = recover(ep)
    assert counts["robot_state"] == 50 and (ep / "robot_state.parquet").is_file()
