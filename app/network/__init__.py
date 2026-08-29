"""Network subsystem: one managed entry for every outbound-network concern.

registry — declarative table of the services the console manages
state    — persisted per-service enable/disable (app/data/network_state.json)
probes   — reachability/credential probes for non-box services
status   — ProbeCache (single-flight, per-service intervals) + status composition

Enforcement (making a disabled service actually refuse traffic) lands in a
follow-up; this package only observes, persists intent, and reports.
"""
