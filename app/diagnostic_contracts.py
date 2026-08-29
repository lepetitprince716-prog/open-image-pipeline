"""Versioned response contracts for the fixed remote diagnostic probes."""
from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class _StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class TransportObservation(_StrictContract):
    schema_: Literal["loramake.remote.transport.v1"] = Field(alias="schema")
    healthy: bool
    computer_name: str
    powershell_version: str
    powershell_edition: str
    remote_time_utc: str
    encoding_sentinel: str
    apostrophe_sentinel: str
    dollar_sentinel: str
    json_literal_ok: bool
    multiline_ok: bool


class ServiceObservation(_StrictContract):
    name: str
    status: str
    present: bool
    running: bool


class OverviewObservation(_StrictContract):
    schema_: Literal["loramake.remote.overview.v1"] = Field(alias="schema")
    healthy: bool
    computer_name: str
    powershell_version: str
    os_caption: str
    os_version: str
    os_build: str
    last_boot_utc: str
    uptime_seconds: int | float
    memory_total_mb: int | float
    memory_free_mb: int | float
    cpu_name: str
    logical_processors: int
    services: list[ServiceObservation]


class NetworkAdapterObservation(_StrictContract):
    alias: str
    description: str | None = None
    ipv4: list[str]
    gateways: list[str]
    dns: list[str]


class NetworkObservation(_StrictContract):
    schema_: Literal["loramake.remote.network.v1"] = Field(alias="schema")
    healthy: bool
    computer_name: str
    adapters: list[NetworkAdapterObservation]
    ssh_port_22: bool
    comfyui_port: int
    comfyui_listening: bool
    collection_error: str | None


class ComfyDeviceObservation(_StrictContract):
    name: str
    type: str
    vram_total_mb: int | float
    vram_free_mb: int | float


class ComfyUIObservation(_StrictContract):
    schema_: Literal["loramake.remote.comfyui.v1"] = Field(alias="schema")
    healthy: bool
    reachable: bool
    base_uri: str
    version: str | None
    expected_version: str | None
    version_matches: bool
    queue_running: int | None
    queue_pending: int | None
    devices: list[ComfyDeviceObservation]
    error: str | None


class ProcessLineageObservation(_StrictContract):
    depth: int
    pid: int
    name: str | None
    executable: str | None
    parent_pid: int | None
    found: bool
    query_error: str | None


class ListenerProcessObservation(_StrictContract):
    pid: int
    name: str | None
    executable: str | None
    parent_pid: int | None
    service_bound: bool | None
    canonical_python_in_lineage: bool | None
    lineage: list[ProcessLineageObservation]


class ListenerServiceObservation(_StrictContract):
    name: str
    status: str
    present: bool
    process_id: int


class ListenerObservation(_StrictContract):
    schema_: Literal["loramake.remote.listener.v1"] = Field(alias="schema")
    healthy: bool
    port: int
    listening: bool
    verdict: Literal["listener-present", "missing", "canonical", "rogue"]
    service_pid: int
    all_owners_service_bound: bool
    all_owners_canonical_python: bool
    processes: list[ListenerProcessObservation]
    expected_service: str | None
    canonical_python: str | None
    service: ListenerServiceObservation | None
    collection_error: str | None
    service_collection_error: str | None


class GpuDeviceObservation(_StrictContract):
    index: int
    name: str
    driver_version: str | None
    memory_total_mb: int | None
    memory_used_mb: int | None
    memory_free_mb: int | None
    utilization_gpu_pct: float | None
    utilization_memory_pct: float | None
    temperature_c: float | None
    power_draw_w: float | None
    power_limit_w: float | None
    pstate: str | None


class GpuProcessObservation(_StrictContract):
    pid: int
    process_name: str
    used_memory_mb: int | None


class GpuObservation(_StrictContract):
    schema_: Literal["loramake.remote.gpu.v1"] = Field(alias="schema")
    healthy: bool
    available: bool
    collected_at_utc: str
    devices: list[GpuDeviceObservation]
    compute_processes: list[GpuProcessObservation]
    warning: str | None
    error: str | None


class SnapshotGpuObservation(_StrictContract):
    name: str
    vendor: str | None
    driver: str | None
    type: str | None


class SnapshotDiskObservation(_StrictContract):
    filesystem: str | None
    total_mb: int | None
    used_mb: int | None
    available_mb: int | None


class SnapshotObservation(_StrictContract):
    schema_: Literal["loramake.remote.snapshot.v1"] = Field(alias="schema")
    healthy: bool
    available: bool
    collected_at_utc: str
    collector: Literal["fastfetch", "powershell"]
    fastfetch_available: bool
    fastfetch_version: str | None
    os_name: str | None
    os_version: str | None
    os_build: str | None
    kernel_name: str | None
    kernel_release: str | None
    kernel_architecture: str | None
    cpu_name: str | None
    cpu_vendor: str | None
    physical_cores: int | None
    logical_processors: int | None
    cpu_max_frequency_mhz: int | None
    gpus: list[SnapshotGpuObservation]
    memory_total_mb: int | None
    memory_used_mb: int | None
    disks: list[SnapshotDiskObservation]
    packages_total: int | None
    warning: str | None
    error: str | None


Observation: TypeAlias = (
    TransportObservation
    | OverviewObservation
    | NetworkObservation
    | ComfyUIObservation
    | ListenerObservation
    | GpuObservation
    | SnapshotObservation
)

OBSERVATION_MODELS = {
    "transport": TransportObservation,
    "overview": OverviewObservation,
    "network": NetworkObservation,
    "comfyui": ComfyUIObservation,
    "listener": ListenerObservation,
    "gpu": GpuObservation,
    "snapshot": SnapshotObservation,
}
