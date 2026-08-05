from investigation_rag.gui.scan_panel import ScanLogSink


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_scan_log_sink_coalesces_routine_per_file_updates():
    emitted: list[str] = []
    clock = FakeClock()
    sink = ScanLogSink(emitted.append, update_interval_seconds=0.5, clock=clock)

    sink("Processing 1/100: first.txt")
    clock.now = 0.1
    sink("Unchanged, using 2 existing chunk(s): first.txt")
    clock.now = 0.2
    sink("Progress: files=10/100")
    clock.now = 0.6
    sink("Processing 11/100: next.txt")

    assert emitted == [
        "Processing 1/100: first.txt",
        "Progress: files=10/100 [coalesced 2 routine update(s)]",
    ]


def test_scan_log_sink_preserves_important_messages_and_flushes_at_end():
    emitted: list[str] = []
    clock = FakeClock()
    sink = ScanLogSink(emitted.append, update_interval_seconds=1.0, clock=clock)

    sink("Processing 1/2: first.txt")
    clock.now = 0.1
    sink("Processing 2/2: second.txt")
    sink("Failed: second.txt (permission denied)")
    clock.now = 0.2
    sink("Progress: files=2/2")
    sink.flush()

    assert emitted == [
        "Processing 1/2: first.txt",
        "Processing 2/2: second.txt",
        "Failed: second.txt (permission denied)",
        "Progress: files=2/2",
    ]
