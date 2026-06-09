"""Per-call runtime configuration assembled from DB + DialRequest.

Spec: cross-cut — pulls together campaign / role_config / prompt_version /
filler_set / filler_phrase rows + DialRequest fields into a single value
object the run loop consumes. Loading lives behind ``load_runtime_config``
so tests can construct ``RuntimeConfig`` in-memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from isales_common.enums import RoleKind
from isales_common.models import (
    Campaign,
    Lead,
    PromptVersion,
    RoleConfig,
)
from isales_common.schemas.messages.dial import DialRequest
from isales_common.schemas.pipeline import (
    ExtractorSpec,
    MainSpec,
    PersonaSpec,
    RefereeSpec,
    RestructureSpec,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from isales_engine.pipeline.prompt_builder import (
    LeadInfo,
    PipelineConfig,
)
from isales_engine.realtime.interruption_detector import InterruptionConfig
from isales_engine.realtime.interruption_rules import build_rule, default_rule
from isales_engine.realtime.silence_detector import SilenceConfig
from isales_engine.transfer.manager import TransferConfig
from isales_engine.wrapup.manager import WrapUpConfig

logger = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    pipeline: PipelineConfig
    filler_phrases: list[str]
    transfer: TransferConfig
    wrap_up: WrapUpConfig
    interruption: InterruptionConfig
    silence: SilenceConfig
    voice_id: str
    fixed_greeting: str | None
    max_no_progress_seconds: int | None
    # ASR EOS endpoint in seconds (pipeline-latency-tail § A). Derived from
    # campaign.asr_eos_silence_ms (ms); NULL → default 0.4s. main.py passes
    # this into build_asr so the per-campaign threshold replaces the
    # hardwired ASR-provider constant.
    asr_partial_stable_s: float = 0.4
    # filler opt-in (pipeline-stream-and-referee). Default off — the streaming
    # main link reaches first audio in ~500ms so filler usually just adds delay.
    filler_enabled: bool = False
    # filler time-gate in ms (tts-cache-and-gated-filler § B). Derived from
    # campaign.filler_delay_ms; NULL → 600. Only play a filler when the main
    # reply's first audio hasn't started within this window.
    filler_delay_ms: int = 600
    # Continuous-interruption protection (ai-pipeline spec delta). Read by
    # run_loop._decide_protection. Stored at RuntimeConfig level (not on
    # InterruptionConfig) because the strategy is a campaign-level policy,
    # not a detector parameter.
    _max_continuous_interruptions: int = 3
    _continuous_interruption_strategy: str = "short_reply"


async def load_runtime_config(
    db: AsyncSession,
    request: DialRequest,
    *,
    pipeline_default_timeout_ms: int = 8000,
) -> RuntimeConfig:
    """Read all per-call configuration from PostgreSQL."""

    campaign = await db.get(Campaign, request.lead.campaign_id)
    if campaign is None:
        raise RuntimeError(f"campaign {request.lead.campaign_id} not found")
    lead = await db.get(Lead, request.lead.lead_id)

    role_configs = (
        (
            await db.execute(
                select(RoleConfig).where(RoleConfig.campaign_id == campaign.id)
            )
        )
        .scalars()
        .all()
    )

    pv_ids = [rc.current_prompt_version_id for rc in role_configs if rc.current_prompt_version_id]
    pv_rows = (
        (
            await db.execute(
                select(PromptVersion).where(PromptVersion.id.in_(pv_ids))
            )
        )
        .scalars()
        .all()
    ) if pv_ids else []
    pv_by_id = {pv.id: pv for pv in pv_rows}

    def _spec_for(rc: RoleConfig) -> tuple[str, str, int]:
        pv_id = rc.current_prompt_version_id
        pv = pv_by_id.get(pv_id) if pv_id else None
        return (
            rc.model or "mock",
            pv.content if pv else "",
            pv_id or 0,
        )

    def _first(kind: RoleKind) -> RoleConfig | None:
        for rc in role_configs:
            if rc.kind == kind.value:
                return rc
        return None

    def _all_enabled(kind: RoleKind) -> list[RoleConfig]:
        return [rc for rc in role_configs if rc.kind == kind.value and rc.enabled]

    def _main_spec(rc: RoleConfig | None) -> MainSpec:
        if rc is None:
            # No main slot configured → empty prompt; main stream produces a
            # default_reply, greeting falls back to "您好。".
            return MainSpec(role_config_id=0, prompt_version_id=0, system_prompt="")
        model, prompt, pv_id = _spec_for(rc)
        return MainSpec(
            role_config_id=rc.id,
            prompt_version_id=pv_id,
            system_prompt=prompt,
            model=model,
            temperature=rc.temperature or 1.0,
            top_p=rc.top_p or 1.0,
        )

    def _referee_spec(rc: RoleConfig) -> RefereeSpec:
        model, prompt, pv_id = _spec_for(rc)
        return RefereeSpec(
            role_config_id=rc.id,
            prompt_version_id=pv_id,
            system_prompt=prompt,
            model=model,
            temperature=rc.temperature or 1.0,
            top_p=rc.top_p or 1.0,
            # label binds routing rules; api enforces non-empty for referees.
            label=rc.label or f"referee_{rc.id}",
        )

    def _persona_spec(rc: RoleConfig) -> PersonaSpec:
        model, prompt, pv_id = _spec_for(rc)
        return PersonaSpec(
            role_config_id=rc.id,
            prompt_version_id=pv_id,
            system_prompt=prompt,
            model=model,
            temperature=rc.temperature or 1.0,
            top_p=rc.top_p or 1.0,
            # label binds routing rules ({type: route, to: <label>}); api
            # enforces non-empty for personas, namespace-isolated from referees.
            label=rc.label or f"persona_{rc.id}",
        )

    def _restructure_spec(rc: RoleConfig | None) -> RestructureSpec | None:
        if rc is None:
            return None
        model, prompt, pv_id = _spec_for(rc)
        return RestructureSpec(
            role_config_id=rc.id,
            prompt_version_id=pv_id,
            system_prompt=prompt,
            model=model,
            temperature=rc.temperature or 1.0,
            top_p=rc.top_p or 1.0,
            label=rc.label or f"restructure_{rc.id}",
        )

    def _extractor_spec(rc: RoleConfig | None) -> ExtractorSpec:
        if rc is None:
            return ExtractorSpec(
                role_config_id=0, prompt_version_id=0, system_prompt=""
            )
        model, prompt, pv_id = _spec_for(rc)
        return ExtractorSpec(
            role_config_id=rc.id,
            prompt_version_id=pv_id,
            system_prompt=prompt,
            model=model,
            temperature=rc.temperature or 1.0,
            top_p=rc.top_p or 1.0,
        )

    pipeline = PipelineConfig(
        main=_main_spec(_first(RoleKind.MAIN)),
        referees=[_referee_spec(rc) for rc in _all_enabled(RoleKind.REFEREE)],
        restructure=_restructure_spec(_first(RoleKind.RESTRUCTURE)),
        routing_rules=[dict(r) for r in (campaign.routing_rules or [])],
        max_continuous_restructure=campaign.max_continuous_restructure,
        extractor=_extractor_spec(_first(RoleKind.EXTRACTOR)),
        default_replies=[str(r) for r in (campaign.default_replies or [])],
        lead=LeadInfo(
            name=lead.name if lead else request.lead.name,
            phone=request.lead.phone,
            custom_data=lead.custom_data if lead else dict(request.lead.custom_data),
        ),
        last_call_summary=(request.history[0].summary if request.history else None),
        follow_up_count=len(request.history),
        short_reply_active=False,
        # gating + multi-persona (engine-tools-multidialogue-gating). Personas are
        # the enabled kind=persona role_configs (eager speculative dialogue
        # routes); tools/persona_fanout_cap/referee gating scalars come straight
        # off the campaign (common 0.8 columns). persona_fanout_cap is clamped to
        # [1,3] at fan-out time in run_loop, not here.
        personas=[_persona_spec(rc) for rc in _all_enabled(RoleKind.PERSONA)],
        tools={str(k): dict(v) for k, v in (campaign.tools or {}).items()},
        persona_fanout_cap=campaign.persona_fanout_cap,
        referee_timeout_ms=campaign.referee_timeout_ms,
        referee_fail_open_route=campaign.referee_fail_open_route,
    )

    # Filler phrases — a flat per-campaign pool of plain strings
    # (filler-campaign-column), same shape as silence_phrases.
    filler_phrases = [str(p) for p in (campaign.filler_phrases or [])]

    transfer = TransferConfig(
        keyword_enabled=campaign.transfer_keyword_enabled,
        keywords=tuple(str(k) for k in (campaign.transfer_keywords or [])),
        intent_enabled=campaign.transfer_intent_enabled,
        intent_threshold=campaign.transfer_intent_threshold,
        intent_system_prompt="判定用户是否要转人工。",
        round_enabled=campaign.transfer_round_enabled,
        round_threshold=campaign.transfer_round_threshold,
        llm_enabled=campaign.transfer_llm_enabled,
        llm_system_prompt="独立判定是否转人工。",
        phrases=tuple(str(p) for p in (campaign.transfer_phrases or [])) or (
            "请稍候，专员稍后联系您。",
        ),
    )

    wrap_up = WrapUpConfig(
        max_rounds=campaign.wrap_up_max_rounds,
        max_seconds=campaign.wrap_up_max_seconds,
        closing_phrases=tuple(
            str(p) for p in (campaign.wrap_up_closing_phrases or [])
        ),
    )

    _whitelist = [str(w) for w in (campaign.interruption_whitelist or [])]
    # Barge-in rule tree: explicit campaign tree if configured, else a backward-
    # compat default synthesized from the legacy whitelist + min_duration columns
    # (engine-interruption-rule-tree design D4 — equivalent to the historical
    # whitelist→length≥2→duration sequence).
    if campaign.interruption_rules is not None:
        _interruption_rule = build_rule(campaign.interruption_rules)
    else:
        _interruption_rule = default_rule(
            whitelist=_whitelist,
            min_text_length=2,
            min_duration_ms=campaign.interruption_min_duration_ms,
        )
    interruption = InterruptionConfig(
        whitelist=tuple(_whitelist),
        rule=_interruption_rule,
    )

    silence = SilenceConfig(
        threshold_ms=campaign.silence_threshold_ms,
        max_activations=campaign.max_silence_activations,
        phrases=tuple(str(p) for p in (campaign.silence_phrases or [])),
        # Preserve an empty phrase so silence-max can直接挂断不播话术 (§11);
        # only coerce a missing (NULL) value to empty, never to a default phrase.
        hangup_phrase=campaign.silence_hangup_phrase or "",
    )

    _ = pipeline_default_timeout_ms  # carried in CallSession.run() args

    # Fixed-template greeting (ai-pipeline § "开场白不走管线"). Campaign-level
    # PG column; NULL falls back to the LLM-generated greeting path (the
    # ``generate_greeting(fixed_template=None)`` branch).
    fixed_greeting: str | None = campaign.greeting

    # ``campaign.voice_id`` now holds the vendor speaker string directly
    # (campaign-greeting-tts-preview § 4C — admin types it in the form), so the
    # engine passes it straight to the TTS provider. NULL / empty → "default",
    # which the provider maps to its own default speaker. Matches the web 试听.
    voice_speaker = campaign.voice_id or "default"

    # ASR EOS endpoint (pipeline-latency-tail § A). campaign.asr_eos_silence_ms
    # is ms; NULL → engine default 400ms. Convert to seconds for the ASR
    # provider's partial_stable_s.
    asr_eos_ms = campaign.asr_eos_silence_ms
    asr_partial_stable_s = (asr_eos_ms if asr_eos_ms is not None else 400) / 1000.0

    # filler time-gate (tts-cache-and-gated-filler § B); NULL → 600ms.
    filler_delay_ms = (
        campaign.filler_delay_ms if campaign.filler_delay_ms is not None else 600
    )

    strategy_value = (
        campaign.continuous_interruption_strategy
        if isinstance(campaign.continuous_interruption_strategy, str)
        else campaign.continuous_interruption_strategy.value
    )

    return RuntimeConfig(
        pipeline=pipeline,
        filler_phrases=filler_phrases,
        transfer=transfer,
        wrap_up=wrap_up,
        interruption=interruption,
        silence=silence,
        voice_id=voice_speaker,
        fixed_greeting=fixed_greeting,
        max_no_progress_seconds=campaign.max_no_progress_seconds,
        asr_partial_stable_s=asr_partial_stable_s,
        filler_enabled=campaign.filler_enabled,
        filler_delay_ms=filler_delay_ms,
        _max_continuous_interruptions=campaign.max_continuous_interruptions,
        _continuous_interruption_strategy=strategy_value,
    )
