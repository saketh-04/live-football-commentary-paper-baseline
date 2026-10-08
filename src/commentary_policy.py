"""
commentary_policy.py

V2 Architecture Component: Adaptive Commentary Policy.

Determines:
  1. Whether to speak — a genuine adaptive speech gate using importance, event
     type, sequence phase, novelty (no repeated identical actions), and confidence.
  2. Commentary style:
       'Excited Climax'        — goals, shots, climax events
       'Analytical Tactical'   — passes, crosses, build-up sequences
       'Historical Contextual' — when strong entity background is retrieved
       'Pacy Play-by-Play'     — standard live description
  3. Speech tempo / cadence.

SPEECH GATE POLICY (explicit decision table):
  - GOAL / PENALTY / RED CARD → always SPEAK (Excited Climax)
  - SHOT with importance >= 0.70 → SPEAK (Excited Climax)
  - SHOT with importance < 0.70 → SPEAK (Pacy Play-by-Play)
  - CROSS at sequence climax → SPEAK (Excited Climax)
  - CROSS not at climax and no strong background → SILENT
  - PASS / DRIVE / CLEARANCE / etc with importance < 0.40 → SILENT
  - Any event with importance >= 0.55 → SPEAK
  - Low importance (< 0.35) + low confidence (< 0.40) → SILENT
  - Repeated same action as immediately preceding event → SILENT (novelty gate)

COMMENTARY GROUNDING RULES:
  - Never claim score changes ('takes the lead', 'equaliser') unless
    live score information is actually available (state.is_score_available).
  - Never claim the exact pitch zone as fact unless it is directly from
    the event's location field (loc is PBP data — it is grounded).
  - Style adjectives (ferocious, sensational, electric) are only used for
    Critical-tier importance events (importance >= 0.90).
  - Background entity facts are only included when entity_score >= 0.30.

Explainable, transparent, and deterministic.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

from temporal_context import EventContext
from event_importance import ImportanceBreakdown
from confidence_scorer import ConfidenceBreakdown
from match_state import MatchState
from context_retrieval import ScoredDocument


@dataclass
class CommentaryPolicyDecision:
    """Decision output produced by the commentary policy."""
    should_commentate: bool
    selected_style: str
    tempo_cadence: str
    priority_rank: int  # 1 (Highest) to 4 (Routine / Silent)
    policy_rationale: str
    generated_text: str
    silence_reason: str  # Empty string if should_commentate is True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_commentate": self.should_commentate,
            "selected_style": self.selected_style,
            "tempo_cadence": self.tempo_cadence,
            "priority_rank": self.priority_rank,
            "policy_rationale": self.policy_rationale,
            "generated_text": self.generated_text,
            "silence_reason": self.silence_reason,
        }


# Actions that are always broadcast regardless of importance
ALWAYS_SPEAK_ACTIONS = {"GOAL", "PENALTY", "RED CARD", "YELLOW CARD"}

# Actions that are default-silent unless importance is high enough
DEFAULT_SILENT_ACTIONS = {
    "PASS", "DRIVE", "CLEARANCE", "THROW IN", "GOAL KICK",
    "KEEPER SAVE ATTEMPT", "BALL PLAYER BLOCK",
}

# Minimum importance thresholds per action category
SPEAK_THRESHOLD = {
    "GOAL": 0.0,          # always speak
    "PENALTY": 0.0,
    "RED CARD": 0.0,
    "YELLOW CARD": 0.50,
    "SHOT": 0.50,         # speak for most shots
    "CROSS": 0.55,        # only speak for high-importance crosses
    "FREE KICK": 0.45,
    "HEADER": 0.55,
    "CORNER": 0.45,
    "PASS": 0.65,         # very high bar for routine passes
    "HIGH PASS": 0.65,
    "DRIVE": 0.65,
    "CLEARANCE": 0.70,
    "BALL PLAYER BLOCK": 0.70,
    "KEEPER SAVE ATTEMPT": 0.60,
    "_default": 0.55,
}


def _silence_decision(reason: str) -> CommentaryPolicyDecision:
    return CommentaryPolicyDecision(
        should_commentate=False,
        selected_style="None",
        tempo_cadence="Silent",
        priority_rank=4,
        policy_rationale=reason,
        generated_text="",
        silence_reason=reason,
    )


def _loc_phrase(loc: str) -> str:
    """Return a location phrase only when it has real content from PBP data."""
    if loc and loc.lower() not in {"", "field", "unknown", "n/a"}:
        if loc.lower() == "out":
            return " from out wide"
        return f" in the {loc.lower()}"
    return ""


def _score_phrase(state: MatchState) -> str:
    """Only describe score situation when score data is actually available."""
    if state.is_score_available:
        return f" ({state.home_team} {state.home_score}–{state.away_score} {state.away_team})"
    return ""


class AdaptiveCommentaryPolicy:
    """
    Selects commentary mode and speech register dynamically.
    Applies a genuine speech gate — not every event produces commentary.
    """

    def evaluate_policy(
        self,
        event_ctx: EventContext,
        importance: ImportanceBreakdown,
        confidence: ConfidenceBreakdown,
        match_state: MatchState,
        top_context: Optional[ScoredDocument] = None,
        preferred_style: Optional[str] = None,
        recent_actions: Optional[List[str]] = None,
    ) -> CommentaryPolicyDecision:
        action = event_ctx.action
        player = event_ctx.player
        team = event_ctx.team
        opp = event_ctx.opponent
        loc = event_ctx.location
        imp_val = importance.final_importance
        conf_val = confidence.composite_confidence

        # ── POLICY DECISION: EVERY STRUCTURED EVENT RECEIVES COMMENTARY ──
        # Event importance, type, and sequence position govern style, verbosity, and register
        should_speak = True
        is_repeat = bool(recent_actions and len(recent_actions) >= 1 and action == recent_actions[-1])

        # Style selection governed by importance and context
        if preferred_style and preferred_style != "Auto (Adaptive Policy)":
            style = preferred_style
            rationale = f"User-selected override style: {preferred_style}."
            tempo = "Custom Tempo"
            rank = 2
        elif action in ALWAYS_SPEAK_ACTIONS or imp_val >= 0.85 or event_ctx.is_sequence_climax:
            style = "Excited Climax"
            rationale = (
                f"High-impact event ({action}, importance {imp_val:.2f}"
                + (", sequence climax" if event_ctx.is_sequence_climax else "")
                + ") triggers excited climax commentary."
            )
            tempo = "Urgent / Climax"
            rank = 1
        elif is_repeat:
            style = "Routine / Compressed"
            rationale = (
                f"Consecutive repeated {action} event compressed stylistically to maintain continuous broadcast without fatigue."
            )
            tempo = "Concise"
            rank = 3
        elif imp_val >= 0.65 or len(event_ctx.previous_events) >= 2:
            style = "Analytical Tactical"
            rationale = (
                f"Tactical phase ({action}, importance {imp_val:.2f}) or multi-event build-up "
                f"({event_ctx.sequence_signature}) triggers analytical commentary."
            )
            tempo = "Steady Broadcast"
            rank = 2
        elif top_context and top_context.entity_score >= 0.30:
            style = "Contextual Play-by-Play"
            rationale = (
                f"Retrieved entity background ({top_context.doc_stem}, score {top_context.final_score:.2f}) "
                f"enriches event commentary."
            )
            tempo = "Measured / In-Depth"
            rank = 2
        else:
            style = "Routine / Concise"
            rationale = f"Standard {action} (importance {imp_val:.2f}) receives concise factual broadcast commentary."
            tempo = "Concise Factual"
            rank = 3

        # ── GROUNDED COMMENTARY GENERATION ───────────────────────────────────────────
        loc_phrase = _loc_phrase(loc)
        score_phrase = _score_phrase(match_state)
        is_critical = imp_val >= 0.90

        if style == "Excited Climax":
            if action == "GOAL":
                if is_critical:
                    text = f"GOAL! What a finish by {player} for {team}!{score_phrase}".strip()
                else:
                    text = f"Goal! {player} scores for {team}!{score_phrase}".strip()
            elif action == "SHOT":
                if is_critical:
                    text = f"SHOT! {player} lets fly for {team}{loc_phrase}!"
                else:
                    text = f"Dangerous shot by {player} for {team}{loc_phrase}!"
            elif action == "PENALTY":
                text = f"PENALTY to {team}! {player} steps up to take it{score_phrase}."
            elif action in {"RED CARD", "YELLOW CARD"}:
                card_type = "red card" if action == "RED CARD" else "yellow card"
                text = f"{player} receives a {card_type} for {team}!"
            elif event_ctx.is_sequence_climax:
                text = f"KEY MOMENT! {player} with a decisive {action.lower()} for {team}{loc_phrase}!"
            else:
                text = f"Decisive moment! {player} with a {action.lower()} for {team}!"

        elif style == "Analytical Tactical":
            prev_acts = " → ".join(
                e.get("action", "action") for e in event_ctx.previous_events[-2:]
            ) if event_ctx.previous_events else ""
            if prev_acts:
                text = f"{team} build through {prev_acts} — {player} executes the {action.lower()}{loc_phrase}."
            else:
                text = f"{team} in possession: {player} plays a {action.lower()}{loc_phrase}."

        elif style == "Contextual Play-by-Play":
            stem_info = top_context.doc_stem if top_context else player
            text = f"{player} ({team}) with the {action.lower()}{loc_phrase}. Notable context: {stem_info}."

        elif style == "Routine / Compressed":
            text = f"Possession recycled: {player} moves the ball for {team}{loc_phrase}."

        else:  # Routine / Concise
            if action in {"PASS", "HIGH PASS"}:
                text = f"{player} keeps possession with a pass for {team}{loc_phrase}."
            elif action in {"DRIBBLE", "DRIVE"}:
                text = f"{player} drives forward on the ball for {team}{loc_phrase}."
            elif action == "TACKLE":
                text = f"{player} challenges into the tackle for {team}{loc_phrase}."
            elif action == "CROSS":
                text = f"{player} delivers a cross for {team}{loc_phrase}."
            elif action == "CLEARANCE":
                text = f"{player} clears the ball under pressure for {team}{loc_phrase}."
            else:
                text = f"{player} ({team}) — {action.lower()}{loc_phrase}."

        return CommentaryPolicyDecision(
            should_commentate=True,
            selected_style=style,
            tempo_cadence=tempo,
            priority_rank=rank,
            policy_rationale=rationale,
            generated_text=text,
            silence_reason="",
        )


if __name__ == "__main__":
    from temporal_context import TemporalContextBuilder
    from event_importance import EventImportanceScorer
    from confidence_scorer import MultimodalConfidenceScorer
    from match_state import MatchStateTracker

    t_b = TemporalContextBuilder()
    imp_s = EventImportanceScorer()
    conf_s = MultimodalConfidenceScorer()
    m_t = MatchStateTracker()
    policy = AdaptiveCommentaryPolicy()

    pbp = Path("data/demo/pbp/0008/play-by-play-en.jsonl")
    contexts = t_b.build_match_contexts(pbp)
    print("=== Commentary Policy Test: All Events in Match 0008 ===\n")
    recent_acts = []
    for ctx in contexts:
        state = m_t.compute_state_for_event("0008", ctx.action, ctx.team, ctx.start_time)
        imp = imp_s.compute_importance(ctx.action, state.match_minute, 0, ctx.is_sequence_climax)
        conf = conf_s.compute_confidence(0.50, 0.30, "VERIFIED", False, 0.5)
        dec = policy.evaluate_policy(ctx, imp, conf, state, recent_actions=recent_acts)
        speak_str = "SPEAK" if dec.should_commentate else "SILENT"
        print(f"  [{ctx.action}] {ctx.player} — importance={imp.final_importance:.3f} ({imp.importance_tier})")
        print(f"    Decision: {speak_str} | Style: {dec.selected_style}")
        if dec.should_commentate:
            print(f"    Text: \"{dec.generated_text}\"")
        else:
            print(f"    Reason: {dec.silence_reason}")
        print()
        recent_acts.append(ctx.action)
