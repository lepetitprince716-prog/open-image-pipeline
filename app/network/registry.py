"""Declarative registry of the network-facing services the console manages."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceSpec:
    id: str
    kind: str  # "box" | "provider" | "service"
    label: str
    actions: tuple[str, ...] = ("probe",)
    # brief provider name this service gates ("civitai", "novelai"); None for non-providers
    provider_id: str | None = None


# gpt_image deliberately has no entry: out of scope for the network dashboard
# (user decision 2026-08-26). novelai is registered ahead of its provider
# adapter so the dashboard and gates are ready; without a token it reports
# "unconfigured".
SERVICES: tuple[ServiceSpec, ...] = (
    ServiceSpec(id="z3", kind="box", label="Z3 ComfyUI", actions=("probe", "free_vram")),
    ServiceSpec(id="zzz", kind="box", label="ZZZ ComfyUI", actions=("probe", "free_vram")),
    ServiceSpec(id="civitai", kind="provider", label="Civitai", provider_id="civitai"),
    ServiceSpec(id="novelai", kind="provider", label="NovelAI", provider_id="novelai"),
    ServiceSpec(id="lora_manager", kind="service", label="LoRA Manager",
                actions=("probe", "start", "stop")),
)

BY_ID: dict[str, ServiceSpec] = {spec.id: spec for spec in SERVICES}
