"""Strict HTTP boundary for fixed remote diagnostics."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from ..diagnostic_contracts import Observation


class DiagnosticRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: Literal["z3", "zzz"]
    probe: Literal["transport", "overview", "network", "comfyui", "listener", "gpu", "snapshot"]


TargetId = Literal["z3", "zzz"]
ProbeId = Literal["transport", "overview", "network", "comfyui", "listener", "gpu", "snapshot"]


class TargetCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: TargetId
    label: str


class ProbeCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: ProbeId
    label: str
    description: str
    timeout_s: int


class DiagnosticCapabilitiesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_: Literal["loramake.remote-diagnostics.capabilities.v1"] = Field(alias="schema")
    runner_ready: bool
    transport: str
    arbitrary_script: Literal[False]
    targets: list[TargetCapability]
    probes: list[ProbeCapability]


class DiagnosticResultResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_: Literal["loramake.remote-diagnostics.result.v1"] = Field(alias="schema")
    request_id: str
    target: TargetId
    probe: ProbeId
    ok: bool
    healthy: bool | None
    summary: str
    error_class: str | None
    message: str | None
    duration_ms: int
    process_exit_code: int | None
    rdbg_exit_code: int | None
    observation: Observation | None
    stdout: str | None
    stderr: str | None
    stdout_truncated: bool
    stderr_truncated: bool


class DiagnosticAuditItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_: Literal["loramake.remote-diagnostics.audit.v1"] = Field(alias="schema")
    timestamp: str
    request_id: str
    target: TargetId
    probe: ProbeId
    ok: bool
    healthy: bool | None
    error_class: str | None
    duration_ms: int
    rdbg_exit_code: int | None
    process_exit_code: int | None


class DiagnosticAuditResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[DiagnosticAuditItem]


def create_router(state: Any) -> APIRouter:
    router = APIRouter()

    @router.get(
        "/api/diagnostics/capabilities",
        response_model=DiagnosticCapabilitiesResponse,
    )
    def capabilities() -> DiagnosticCapabilitiesResponse:
        return state.remote_diagnostics.capabilities()

    @router.post("/api/diagnostics", response_model=DiagnosticResultResponse)
    def diagnose(body: DiagnosticRequest) -> DiagnosticResultResponse:
        try:
            return state.remote_diagnostics.run(body.target, body.probe)
        except state.remote_diagnostics.DiagnosticsBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except state.remote_diagnostics.DiagnosticsBackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    @router.get("/api/diagnostics/audit", response_model=DiagnosticAuditResponse)
    def audit(limit: int = Query(default=20, ge=1, le=100)) -> DiagnosticAuditResponse:
        return {"items": state.remote_diagnostics.read_audit(limit)}

    return router
