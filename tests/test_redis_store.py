from __future__ import annotations

import serve


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.members = set()
        self.ttls = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.values[key] = value
        self.ttls[key] = ttl

    def sadd(self, _key, value):
        self.members.add(value)

    def expire(self, *_args):
        return True

    def scard(self, _key):
        return len(self.members)

    def smembers(self, _key):
        return set(self.members)

    def delete(self, key):
        self.values.pop(key, None)

    def srem(self, _key, value):
        self.members.discard(value)

    class _Lock:
        def acquire(self, **_kwargs):
            return True

        def release(self):
            return None

    def lock(self, *_args, **_kwargs):
        return self._Lock()


def test_redis_run_state_roundtrip(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(serve, "RUN_STORE", "redis")
    monkeypatch.setattr(serve, "_redis_client", fake)
    run = serve.TrinityRun("a" * 32, [{"role": "user", "content": "hi"}], [])
    run.response_metadata = {"citations": ["https://example.test"]}
    run.request_events = {"req-1": {"type": "final", "text": "cached"}}
    serve._register_run(run)
    loaded = serve.get_run(run.run_id)
    assert loaded.run_id == run.run_id
    assert loaded.response_metadata == run.response_metadata
    assert loaded.request_events == run.request_events
    assert serve.delete_run(run.run_id)
    monkeypatch.setattr(serve, "_redis_client", None)


def test_redis_sweep_removes_expired_state(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(serve, "RUN_STORE", "redis")
    monkeypatch.setattr(serve, "_redis_client", fake)
    run = serve.TrinityRun("b" * 32, [{"role": "user", "content": "hi"}], [])
    run.last_active = 0
    serve._register_run(run)
    serve._sweep_runs()
    assert run.run_id not in fake.members
    monkeypatch.setattr(serve, "_redis_client", None)


def test_redis_sweep_preserves_in_flight_run_and_extends_ttl(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(serve, "RUN_STORE", "redis")
    monkeypatch.setattr(serve, "_redis_client", fake)
    monkeypatch.setattr(serve, "RUN_TTL", 10)
    monkeypatch.setattr(serve, "REDIS_LOCK_TIMEOUT", 300)
    run = serve.TrinityRun("c" * 32, [{"role": "user", "content": "hi"}], [])
    run.last_active = 0
    run.in_flight = 1

    serve._register_run(run)
    serve._sweep_runs()

    assert run.run_id in fake.members
    assert fake.ttls[serve._redis_key(run.run_id)] == 300
    monkeypatch.setattr(serve, "_redis_client", None)
