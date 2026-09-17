"""Concurrent first calls must not race on GenZ's scratch directory.

budsim evaluates deployment candidates on several threads. On the vGPU test cluster the first plan after a pod start
logged "Failed to estimate CPU prefill performance: [Errno 17] File exists: '/tmp/genz/data/model'":
save_layers checked for the directory and then created it, so the thread that lost the race raised
and its candidate was planned with no latency metrics at all.
"""
import threading

from llm_memory_calculator.genz.Models import get_language_model as glm


def test_concurrent_save_layers_into_a_fresh_directory_all_succeed(tmp_path):
    data_path = str(tmp_path / "fresh")  # does not exist yet: every thread takes the create path
    barrier = threading.Barrier(16)
    errors, names = [], []

    def worker():
        try:
            barrier.wait()
            names.append(glm.save_layers(layers=[["QKV", 1, 1, 1, 1, 1, 0, 0]], data_path=data_path, name="m"))
        except Exception as exc:  # pragma: no cover - the failure being guarded against
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(set(names)) == 16
