"""Bounded growth of the router's append-only telemetry files."""

import server


def test_secure_append_rotates_one_generation(monkeypatch, tmp_path):
    path = tmp_path / "decisions.log"
    monkeypatch.setattr(server, "_LOG_MAX_BYTES", 200)

    for index in range(12):
        server._secure_append(path, {"index": index, "pad": "x" * 40})

    rotated = tmp_path / "decisions.log.1"
    assert rotated.exists()  # previous generation preserved, one generation only
    assert path.exists()
    current = path.read_text().splitlines()
    archived = rotated.read_text().splitlines()
    # The archived generation is strictly older than the live file's content.
    assert archived[0].startswith('{"index": 4')
    assert current[-1].startswith('{"index": 11')

    # Further appends keep overwriting the single archived generation.
    for index in range(12, 30):
        server._secure_append(path, {"index": index, "pad": "x" * 40})
    assert len(list(tmp_path.glob("decisions.log*"))) == 2
    assert '{"index": 12' not in path.read_text().splitlines()[0]

    assert path.stat().st_mode & 0o777 == 0o600
    assert rotated.stat().st_mode & 0o777 == 0o600


def test_secure_append_creates_fresh_file(tmp_path):
    path = tmp_path / "outcomes.jsonl"
    server._secure_append(path, {"first": 1})
    assert path.read_text() == '{"first": 1}\n'
