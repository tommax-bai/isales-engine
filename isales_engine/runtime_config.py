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
    FillerPhrase,
    FillerSet,
    Lead,
    PromptVersion,
    RoleConfig,
)
from isales_common.schemas.messages.dial import DialRequest
from isales_common.schemas.pipeline import ExtractorSpec, MainSpec, RefereeSpec
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from isales_engine.pipeline.prompt_builder import (
    LeadInfo,
    PipelineConfig,
)
from isales_engine.realtime.filler_manager import FillerPhraseSpec, FillerSetSpec
from isales_engine.realtime.interruption_detector import InterruptionConfig
from isales_engine.realtime.silence_detector import SilenceConfig
from isales_engine.transfer.manager import TransferConfig
from isales_engine.wrapup.manager import WrapUpConfig

logger = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    pipeline: PipelineConfig
    fillers: list[FillerSetSpec]
    transfer: TransferConfig
    wrap_up: WrapUpConfig
    interruption: InterruptionConfig
    silence: SilenceConfig
    voice_id: str
    fixed_greeting: str | None
    max_no_progress_seconds: int | None
    # filler opt-in (pipeline-stream-and-referee). Default off — the streaming
    # main link reaches first audio in ~500ms so filler usually just adds delay.
    filler_enabled: bool = False
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

    def _referee_spec(rc: RoleConfig | None) -> RefereeSpec:
        if rc is None:
            # No referee slot → empty prompt; run_referee fail-opens to continue.
            return RefereeSpec(
                role_config_id=0, prompt_version_id=0, system_prompt=""
            )
        model, prompt, pv_id = _spec_for(rc)
        return RefereeSpec(
            role_config_id=rc.id,
            prompt_version_id=pv_id,
            system_prompt=prompt,
            model=model,
            temperature=rc.temperature or 1.0,
            top_p=rc.top_p or 1.0,
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
        referee=_referee_spec(_first(RoleKind.REFEREE)),
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
    )

    # Filler sets (sorted by sort_order in FillerManager).
    filler_sets = (
        (
            await db.execute(
                select(FillerSet).where(FillerSet.campaign_id == campaign.id)
            )
        )
        .scalars()
        .all()
    )
    fillers: list[FillerSetSpec] = []
    if filler_sets:
        fs_ids = [fs.id for fs in filler_sets]
        phrase_rows = (
            (
                await db.execute(
                    select(FillerPhrase).where(FillerPhrase.filler_set_id.in_(fs_ids))
                )
            )
            .scalars()
            .all()
        )
        phrases_by_set: dict[int, list[FillerPhrase]] = {fs.id: [] for fs in filler_sets}
        for p in phrase_rows:
            phrases_by_set[p.filler_set_id].append(p)
        for fs in filler_sets:
            fillers.append(
                FillerSetSpec(
                    id=fs.id,
                    sort_order=fs.sort_order,
                    phrases=[
                        FillerPhraseSpec(
                            id=p.id,
                            text=p.phrase,
                            audio_url=p.audio_url,
                            generation_status=(
                                p.generation_status
                                if isinstance(p.generation_status, str)
                                else p.generation_status.value
                            ),
                        )
                        for p in phrases_by_set[fs.id]
                    ],
                )
            )

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

    interruption = InterruptionConfig(
        whitelist=tuple(str(w) for w in (campaign.interruption_whitelist or [])),
        min_duration_ms=campaign.interruption_min_duration_ms,
    )

    silence = SilenceConfig(
        threshold_ms=campaign.silence_threshold_ms,
        max_activations=campaign.max_silence_activations,
        phrases=tuple(str(p) for p in (campaign.silence_phrases or [])),
        hangup_phrase=campaign.silence_hangup_phrase or "稍后联系，再见。",
    )

    _ = pipeline_default_timeout_ms  # carried in CallSession.run() args

    # Fixed-template greeting (ai-pipeline § "开场白不走管线"). Campaign-level
    # PG column; NULL falls back to the LLM-generated greeting path (the
    # ``generate_greeting(fixed_template=None)`` branch).
    fixed_greeting: str | None = campaign.greeting

    strategy_value = (
        campaign.continuous_interruption_strategy
        if isinstance(campaign.continuous_interruption_strategy, str)
        else campaign.continuous_interruption_strategy.value
    )

    return RuntimeConfig(
        pipeline=pipeline,
        fillers=fillers,
        transfer=transfer,
        wrap_up=wrap_up,
        interruption=interruption,
        silence=silence,
        voice_id="default",
        fixed_greeting=fixed_greeting,
        max_no_progress_seconds=campaign.max_no_progress_seconds,
        filler_enabled=campaign.filler_enabled,
        _max_continuous_interruptions=campaign.max_continuous_interruptions,
        _continuous_interruption_strategy=strategy_value,
    )
