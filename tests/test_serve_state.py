import threading

from cairn.serve.state import FAILED, LOADING, READY, STARTING, WARMING, Service


def test_a_new_service_has_not_started_and_is_not_ready():
    service = Service(lambda: "engine")
    assert service.status == STARTING
    assert not service.ready
    assert service.engine is None


def test_loading_succeeds_and_the_engine_is_kept():
    service = Service(lambda: "engine")
    service.start()
    assert service.wait(5)
    assert service.status == READY
    assert service.ready
    assert service.engine == "engine"
    assert service.error is None
    assert service.load_seconds >= 0


def test_the_service_is_not_ready_while_the_loader_runs():
    release = threading.Event()
    entered = threading.Event()

    def loader():
        entered.set()
        release.wait(5)
        return "engine"

    service = Service(loader)
    service.start()
    assert entered.wait(5)
    assert service.status == LOADING
    assert not service.ready
    assert service.engine is None
    assert service.wait(0.05) is False
    release.set()
    assert service.wait(5)
    assert service.ready


def test_a_failing_loader_makes_the_service_failed_with_the_error_text():
    def loader():
        raise FileNotFoundError("data/chunks.jsonl not found")

    service = Service(loader)
    service.start()
    assert service.wait(5)
    assert service.status == FAILED
    assert not service.ready
    assert service.engine is None
    assert "FileNotFoundError" in service.error
    assert "chunks.jsonl" in service.error


def test_warm_up_runs_once_on_the_engine_before_the_service_is_ready():
    seen = []
    states = []

    def warm_up(engine):
        seen.append(engine)
        states.append(service.status)

    service = Service(lambda: "engine", warm_up=warm_up)
    service.start()
    assert service.wait(5)
    assert seen == ["engine"]
    assert states == [WARMING]
    assert service.ready
    assert service.warm_seconds >= 0


def test_a_failing_warm_up_makes_the_service_failed_and_hides_the_engine():
    def warm_up(engine):
        raise RuntimeError("model would not load")

    service = Service(lambda: "engine", warm_up=warm_up)
    service.start()
    assert service.wait(5)
    assert service.status == FAILED
    assert service.engine is None
    assert "RuntimeError" in service.error


def test_start_twice_runs_the_loader_once():
    calls = []

    def loader():
        calls.append(1)
        return "engine"

    service = Service(loader)
    service.start()
    service.start()
    assert service.wait(5)
    service.start()
    assert calls == [1]


def test_describe_is_what_health_shows():
    service = Service(lambda: "engine")
    service.start()
    service.wait(5)
    info = service.describe()
    assert info["status"] == READY
    assert info["ready"] is True
    assert info["error"] is None
    assert set(info) == {"status", "ready", "error", "load_seconds", "warm_seconds"}
