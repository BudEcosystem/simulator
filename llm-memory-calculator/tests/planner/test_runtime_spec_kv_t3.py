"""FRD-023 Phase 4 on the runtime-spec path (FRD-025): the ``kv.tiers`` module renders T3, the Mooncake store
connector, exactly as budcluster's legacy render does (``deployment/engine_processors.py``):

* the store alone, or behind T1 in a ``MultiConnector``, with ``kv_load_failure_policy: recompute`` and the plan's KV
  namespace as ``cache_prefix``;
* the store client's JSON config in ``MOONCAKE_CONFIG_JSON``, written to ``MOONCAKE_CONFIG_PATH`` by the engine's
  command before it execs the server; the TCP limits; ``PYTHONHASHSEED``; the reproducible prefix hash;
* the pod's host memory grows by the store client's buffers, and it carries ``bud-ai-foundry/kv-namespace``;
* nothing of it without ``mooncake_store`` on the engine record, with prefix caching off, or for RDMA.
"""

import copy
import json

from llm_memory_calculator.runtime.modules import LEGACY
from llm_memory_calculator.runtime.modules import legacy as lg
from llm_memory_calculator.runtime.spec import flatten_args

from .test_runtime_spec import golden_spec

NAMESPACE = "kvns-3b1d0f6a2c9e8d71"
MASTER = "mooncake-master.bud-kv.svc:50051"
FEATURES = ["kv_events", "cache_salt", "cpu_offload_shm", "mooncake_store"]
T3 = {
    "tier": "T3",
    "backend": "mooncake",
    "size_gib": 16.0,
    "storage_class": None,
    "transport": "tcp",
    "params": {
        "block_size": 16,
        "load_failure_policy": "recompute",
        "cache_prefix": NAMESPACE,
        "master_server_address": MASTER,
        "protocol": "tcp",
        "local_buffer_size_mib": 512,
    },
}
T1 = {"tier": "T1", "backend": "cpu", "size_gib": 2.0, "params": {"host_memory": "register", "block_size": 256}}


def _plan(tiers=None, **over):
    plan = {
        "version": 1,
        "namespace": NAMESPACE,
        "profile": "mooncake",
        "hash_algo": "sha256_cbor",
        "kv_cache_dtype": "auto",
        "tiers": copy.deepcopy([T3] if tiers is None else tiers),
    }
    plan.update(over)
    return plan


def _spec(plan, features=FEATURES):
    spec, _ = golden_spec("cuda_dedicated_tp1", versions=LEGACY, kv_features=features, kv_plan=plan)
    return spec


def _args(spec):
    return flatten_args(spec.engine_container().args, style=spec.engine.arg_style)


def _arg(spec, name):
    values = [a.split("=", 1)[1] for a in _args(spec) if a.startswith(f"--{name}=")]
    return values[0] if values else None


def test_the_store_alone_renders_the_mooncake_connector():
    spec = _spec(_plan())
    assert json.loads(_arg(spec, "kv-transfer-config")) == {
        "kv_connector": "MooncakeStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"load_async": True, "cache_prefix": NAMESPACE},
        "kv_load_failure_policy": "recompute",
    }
    assert _arg(spec, "prefix-caching-hash-algo") == "sha256_cbor"


def test_t1_and_the_store_render_a_multi_connector():
    spec = _spec(_plan(tiers=[T1, T3], profile="native_mooncake"))
    config = json.loads(_arg(spec, "kv-transfer-config"))
    assert config["kv_connector"] == "MultiConnector" and config["kv_load_failure_policy"] == "recompute"
    first, second = config["kv_connector_extra_config"]["connectors"]
    assert first["kv_connector"] == "OffloadingConnector" and first["kv_connector_extra_config"]["cpu_bytes_to_use"] == 2 * 2**30
    assert second == {
        "kv_connector": "MooncakeStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"load_async": True, "cache_prefix": NAMESPACE},
    }
    command = spec.engine_container().command
    assert command[2].startswith("rm -f /dev/shm/vllm_offload_*.mmap; ")


def test_the_store_clients_config_env_and_command():
    spec = _spec(_plan())
    eng = spec.engine_container()
    env = dict(eng.env)
    assert env["MOONCAKE_CONFIG_PATH"] == lg.T3_CONFIG_PATH
    assert json.loads(env["MOONCAKE_CONFIG_JSON"]) == {
        "mode": "standalone-store",
        "metadata_server": "P2PHANDSHAKE",
        "master_server_address": MASTER,
        "global_segment_size": 0,
        "local_buffer_size": "512MB",
        "protocol": "tcp",
        "device_name": "",
        "enable_offload": False,
    }
    assert env["MC_TCP_LANES_PER_PEER"] == "16" and env["MC_TCP_ADMISSION_TIMEOUT_MS"] == "30000"
    assert env["PYTHONHASHSEED"] == "0"
    assert eng.command[:2] == ["/bin/sh", "-c"]
    assert 'printf \'%s\' "$MOONCAKE_CONFIG_JSON" > "$MOONCAKE_CONFIG_PATH" && ' in eng.command[2]
    assert eng.command[2].endswith('exec python3 -m vllm.entrypoints.openai.api_server "$@"')
    assert spec.pod.labels.get("bud-ai-foundry/kv-namespace") == NAMESPACE


def test_the_store_clients_buffers_are_budgeted_as_host_memory():
    """T3 alone requests no host memory (its ~1 GiB per rank fits budsim's per-GPU allowance for pods without a
    request); with T1, the pod's request carries T3's buffers too (memory_plan.compute_budget)."""
    group = {"type": "cuda", "engine_type": "vllm", "tp_size": 2, "pp_size": 1, "kv_features": FEATURES}
    assert lg.plan_t3(group, _plan())["bytes"] == 2 * lg.T3_HOST_MIB_PER_RANK * lg.MIB
    assert _spec(_plan()).engine_container().resources.limits.get("memory_mib") is None
    both = _spec(_plan(tiers=[T1, T3], profile="native_mooncake")).engine_container().resources.limits["memory_mib"]
    t1 = _spec(_plan(tiers=[T1], profile="native")).engine_container().resources.limits["memory_mib"]
    assert both - t1 == lg.T3_HOST_MIB_PER_RANK


def test_nothing_renders_without_the_feature_or_for_rdma():
    for spec in (
        _spec(_plan(), features=["kv_events", "cache_salt", "cpu_offload_shm"]),
        _spec(_plan(tiers=[{**T3, "params": {**T3["params"], "protocol": "rdma"}, "transport": "rdma"}])),
    ):
        assert "MooncakeStore" not in (_arg(spec, "kv-transfer-config") or "")
        assert not any(k.startswith(("MOONCAKE_", "MC_TCP_")) for k in dict(spec.engine_container().env))


def test_the_budget_mirrors_budclusters_gates():
    """legacy.plan_t3 mirrors memory_plan.plan_t3: the same gates, the same bytes."""
    base = {"type": "cuda", "engine_type": "vllm", "tp_size": 1, "pp_size": 1, "kv_features": FEATURES}
    assert lg.plan_t3(base, _plan())["master"] == MASTER
    assert lg.plan_t3({**base, "pp_size": 2}, _plan()) is None
    assert lg.plan_t3({**base, "type": "cpu_high"}, _plan()) is None
    assert lg.plan_t3({**base, "is_embedding": True}, _plan()) is None
    assert lg.plan_t3(base, _plan(namespace="../../etc")) is None
    bad = {**T3, "params": {**T3["params"], "master_server_address": "evil;rm -rf /:1"}}
    assert lg.plan_t3(base, _plan(tiers=[bad])) is None
    wrong = {**T3, "params": {**T3["params"], "cache_prefix": "kvns-0000000000000000"}}
    assert lg.plan_t3(base, _plan(tiers=[wrong])) is None
