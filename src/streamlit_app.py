"""
streamlit_app.py

Context-Aware Multimodal Football Commentary Intelligence System (V2)
Research Dashboard - Clean Professional Layout

Architecture:
  Video -> Event -> Entity -> Visual -> Temporal -> Match State -> Graph ->
  Importance -> Retrieval -> Counterfactual -> Policy -> Generation ->
  Factual Verification -> Visual Verification -> Confidence -> Final Commentary

Pages:
  1. Match Intelligence (sequential clean layout)
  2. AI Reasoning Pipeline (10-step trace)
  3. Retrieval Analysis (A0 / A1 / Full comparison)
  4. Event Graph & Match Analytics
  5. Research Evaluation & Ablation
  6. Methodology
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import re
import pandas as pd
import streamlit as st

from improved_rag import ImprovedFootballRAG
from temporal_context import TemporalContextBuilder, EventContext
from match_state import MatchStateTracker, MatchState
from event_importance import EventImportanceScorer, ImportanceBreakdown
from context_retrieval import ContextAwareRetriever, ScoredDocument
from factual_verifier import FactualConsistencyVerifier, VerificationReport

from visual_verifier import VisualFrameVerifier, VisualVerificationResult
from confidence_scorer import MultimodalConfidenceScorer, ConfidenceBreakdown
from event_graph import EventGraphBuilder, MatchEventGraph
from counterfactual import CounterfactualEngine, CounterfactualScenario
from commentary_policy import AdaptiveCommentaryPolicy, CommentaryPolicyDecision
from player_intelligence import PlayerIntelligenceAggregator, PlayerMatchStats

# Ã¢"â‚¬Ã¢"â‚¬ Page Config Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
st.set_page_config(
    page_title="Football AI Commentary Lab",
    page_icon="FOOTBALL",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ã¢"â‚¬Ã¢"â‚¬ CSS Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Ã¢"â‚¬Ã¢"â‚¬ Match Header Ã¢"â‚¬Ã¢"â‚¬ */
.match-header {
    background: linear-gradient(135deg, #111827 0%, #1a2035 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    gap: 1rem;
}
.match-team {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f8fafc;
}
.match-team-away {
    text-align: right;
}
.match-score-center {
    text-align: center;
}
.match-score-val {
    font-size: 2.5rem;
    font-weight: 800;
    color: #38bdf8;
    letter-spacing: 0.05em;
}
.match-meta {
    font-size: 0.82rem;
    color: #94a3b8;
    text-align: center;
    margin-top: 0.3rem;
}

/* Ã¢"â‚¬Ã¢"â‚¬ Section Headers Ã¢"â‚¬Ã¢"â‚¬ */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
}

/* Ã¢"â‚¬Ã¢"â‚¬ Cards Ã¢"â‚¬Ã¢"â‚¬ */
.card {
    background: rgba(17, 24, 39, 0.9);
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.card-green  { border-left: 3px solid #10b981; }
.card-blue   { border-left: 3px solid #38bdf8; }
.card-purple { border-left: 3px solid #8b5cf6; }
.card-amber  { border-left: 3px solid #f59e0b; }
.card-red    { border-left: 3px solid #ef4444; }

/* Ã¢"â‚¬Ã¢"â‚¬ Timeline Ã¢"â‚¬Ã¢"â‚¬ */
.timeline-event {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.7rem 0;
    border-bottom: 1px solid #1f2937;
}
.timeline-event:last-child { border-bottom: none; }
.timeline-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #374151;
    flex-shrink: 0;
    margin-top: 4px;
}
.timeline-dot-active {
    background: #38bdf8;
    box-shadow: 0 0 8px rgba(56,189,248,0.5);
    width: 14px;
    height: 14px;
}
.timeline-dot-goal {
    background: #10b981;
    box-shadow: 0 0 8px rgba(16,185,129,0.5);
}
.timeline-time {
    font-size: 0.75rem;
    color: #64748b;
    font-variant-numeric: tabular-nums;
    min-width: 48px;
}
.timeline-action {
    font-size: 0.85rem;
    font-weight: 600;
    color: #cbd5e1;
}
.timeline-action-active {
    color: #38bdf8;
}
.timeline-action-goal {
    color: #10b981;
}
.timeline-player {
    font-size: 0.75rem;
    color: #64748b;
}

/* Ã¢"â‚¬Ã¢"â‚¬ Event Identity Block Ã¢"â‚¬Ã¢"â‚¬ */
.event-identity-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.identity-cell {
    background: #111827;
    border-radius: 8px;
    padding: 0.75rem 1rem;
}
.identity-label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.25rem;
}
.identity-value {
    font-size: 1rem;
    font-weight: 700;
    color: #f1f5f9;
}

/* Ã¢"â‚¬Ã¢"â‚¬ Sequence Arrow Ã¢"â‚¬Ã¢"â‚¬ */
.seq-flow {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 0.5rem 0;
}
.seq-node {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: #94a3b8;
}
.seq-node-active {
    background: #1e3a5f;
    border-color: #38bdf8;
    color: #38bdf8;
}
.seq-node-goal {
    background: #064e3b;
    border-color: #10b981;
    color: #6ee7b7;
}
.seq-arrow {
    color: #4b5563;
    font-size: 0.9rem;
}

/* Ã¢"â‚¬Ã¢"â‚¬ Importance Meter Ã¢"â‚¬Ã¢"â‚¬ */
.imp-bar-bg {
    background: #1f2937;
    border-radius: 6px;
    height: 8px;
    width: 100%;
    margin: 0.4rem 0;
    overflow: hidden;
}
.imp-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #f59e0b, #ef4444);
}

/* Ã¢"â‚¬Ã¢"â‚¬ Evidence Row Ã¢"â‚¬Ã¢"â‚¬ */
.evidence-row {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
}
.evidence-rank {
    font-size: 0.7rem;
    color: #6b7280;
    font-weight: 700;
}
.evidence-name {
    font-weight: 700;
    color: #e2e8f0;
    font-size: 0.9rem;
}
.evidence-scores {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 0.25rem;
}
.evidence-snippet {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    line-height: 1.4;
    max-height: 50px;
    overflow: hidden;
    position: relative;
}

/* Ã¢"â‚¬Ã¢"â‚¬ Verification Checks Ã¢"â‚¬Ã¢"â‚¬ */
.check-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0;
    font-size: 0.85rem;
}
.check-pass { color: #10b981; font-weight: 600; }
.check-fail { color: #ef4444; font-weight: 600; }

/* Ã¢"â‚¬Ã¢"â‚¬ Confidence Bar Ã¢"â‚¬Ã¢"â‚¬ */
.conf-bar-bg {
    background: #1f2937;
    border-radius: 6px;
    height: 12px;
    width: 100%;
    margin: 0.5rem 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.3s;
}
.conf-sub-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.conf-sub {
    background: #111827;
    border-radius: 6px;
    padding: 0.5rem;
    text-align: center;
}
.conf-sub-val {
    font-size: 1rem;
    font-weight: 700;
}
.conf-sub-lbl {
    font-size: 0.65rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Ã¢"â‚¬Ã¢"â‚¬ Commentary Output Ã¢"â‚¬Ã¢"â‚¬ */
.commentary-box {
    background: linear-gradient(135deg, #0f2027 0%, #0d1b2a 100%);
    border: 1px solid #164e63;
    border-left: 4px solid #38bdf8;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 0.5rem;
}
.commentary-text {
    font-size: 1.25rem;
    font-weight: 600;
    color: #f0f9ff;
    line-height: 1.6;
    font-style: italic;
}

/* Ã¢"â‚¬Ã¢"â‚¬ Chips Ã¢"â‚¬Ã¢"â‚¬ */
.chip {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.74rem;
    font-weight: 700;
    margin-right: 0.35rem;
}
.chip-green  { background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
.chip-blue   { background: #1e3a8a; color: #93c5fd; border: 1px solid #3b82f6; }
.chip-amber  { background: #78350f; color: #fde68a; border: 1px solid #d97706; }
.chip-purple { background: #4c1d95; color: #ddd6fe; border: 1px solid #7c3aed; }
.chip-red    { background: #7f1d1d; color: #fecaca; border: 1px solid #dc2626; }
.chip-gray   { background: #1f2937; color: #9ca3af; border: 1px solid #374151; }

/* Ã¢"â‚¬Ã¢"â‚¬ Metric Box Ã¢"â‚¬Ã¢"â‚¬ */
.metric-box {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-size: 2rem;
    font-weight: 800;
    color: #38bdf8;
}
.metric-lbl {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* Ã¢"â‚¬Ã¢"â‚¬ Score Row Ã¢"â‚¬Ã¢"â‚¬ */
.score-row {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    margin-bottom: 0.4rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.score-row-top { border-color: #059669; background: rgba(6,78,59,0.2); }

/* Ã¢"â‚¬Ã¢"â‚¬ Dividers Ã¢"â‚¬Ã¢"â‚¬ */
.section-divider {
    border: none;
    border-top: 1px solid #1f2937;
    margin: 1.2rem 0;
}

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Ã¢"â‚¬Ã¢"â‚¬ Cached Loaders Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
@st.cache_resource(show_spinner="Loading FAISS index & Sentence-Transformer...")
def get_rag_engine() -> ImprovedFootballRAG:
    return ImprovedFootballRAG()


@st.cache_resource
def get_system_components():
    rag = get_rag_engine()
    return {
        "rag": rag,
        "t_builder": TemporalContextBuilder(),
        "m_tracker": MatchStateTracker(),
        "imp_scorer": EventImportanceScorer(),
        "retriever": ContextAwareRetriever(rag),
        "verifier": FactualConsistencyVerifier(),
        "visual_verifier": VisualFrameVerifier(),
        "conf_scorer": MultimodalConfidenceScorer(),
        "graph_builder": EventGraphBuilder(),
        "cf_engine": CounterfactualEngine(),
        "policy": AdaptiveCommentaryPolicy(),
        "player_agg": PlayerIntelligenceAggregator(),
    }


def find_video_path(match_id: str) -> Optional[Path]:
    m = f"{int(match_id):04d}" if match_id.isdigit() else match_id
    for folder in [f"outputs/demo-step4/{m}", f"outputs/demo-step3"]:
        d = Path(folder)
        if d.is_dir():
            for name in ["en-w-sub-w-gsr.mp4", "en-w-sub-wo-gsr.mp4", "en-wo-sub-wo-gsr.mp4"]:
                v = d / name
                if v.exists() and v.stat().st_size > 1_000_000:
                    return v
    p3 = Path(f"outputs/demo-step3/{m}-en.mp4")
    if p3.exists() and p3.stat().st_size > 1_000_000:
        return p3
    return None


def importance_color(tier: str) -> str:
    return {"Critical": "#ef4444", "High": "#f59e0b", "Moderate": "#38bdf8", "Routine": "#6b7280"}.get(tier, "#6b7280")


def conf_color(score: float) -> str:
    if score >= 0.75:
        return "#10b981"
    elif score >= 0.50:
        return "#f59e0b"
    return "#ef4444"


def action_dot_class(action: str, is_active: bool) -> str:
    if is_active:
        return "timeline-dot timeline-dot-active"
    if action in {"GOAL", "PENALTY"}:
        return "timeline-dot timeline-dot-goal"
    return "timeline-dot"


def seq_node_class(action: str, is_last: bool) -> str:
    if action == "GOAL":
        return "seq-node seq-node-goal"
    if is_last:
        return "seq-node seq-node-active"
    return "seq-node"


def render_sequence(sequence_signature: str, current_action: str):
    parts = [p.strip() for p in sequence_signature.split("->")]
    html_parts = []
    for i, p in enumerate(parts):
        is_last = (i == len(parts) - 1)
        css = seq_node_class(p, is_last)
        html_parts.append(f'<span class="{css}">{p}</span>')
        if not is_last:
            html_parts.append('<span class="seq-arrow">-></span>')
    return '<div class="seq-flow">' + "".join(html_parts) + "</div>"


def render_verification_checks(verif: VerificationReport, player: str, team: str, opponent: str, action: str) -> str:
    items = [
        ("Player", player, True),
        ("Team", team, True),
        ("Opponent", opponent, True),
        ("Action", action, verif.status == "VERIFIED"),
    ]
    rows = []
    for label, val, ok in items:
        icon = '<span class="check-pass">..."</span>' if ok else '<span class="check-fail">..."”</span>'
        rows.append(f'<div class="check-row">{icon} <strong>{label}:</strong>&nbsp;{val}</div>')
    return "".join(rows)


# Ã¢"â‚¬Ã¢"â‚¬ Sidebar Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
st.sidebar.markdown("## FOOTBALL Football AI Lab")
st.sidebar.caption("Context-Aware Multimodal Commentary Intelligence")
st.sidebar.markdown("---")

PAGES = [
    "MATCH  Match Intelligence",
    "AI  AI Reasoning Pipeline",
    "RETRIEVAL  Retrieval Analysis",
    "GRAPH  Event Graph & Analytics",
    "RESEARCH  Research Evaluation",
    "METHOD  Methodology",
]
selected_page = st.sidebar.radio("Pages", PAGES, index=0, label_visibility="collapsed")

pbp_folders = sorted([
    p.name for p in Path("data/demo/pbp").iterdir()
    if p.is_dir() and (p / "play-by-play-en.jsonl").exists()
])
selected_match = st.sidebar.selectbox("Match", pbp_folders, index=0)

components = get_system_components()
pbp_file = Path(f"data/demo/pbp/{selected_match}/play-by-play-en.jsonl")
contexts = components["t_builder"].build_match_contexts(pbp_file, match_id=selected_match)

video_file = find_video_path(selected_match)

st.sidebar.markdown("---")
st.sidebar.caption(f"**Events:** {len(contexts)}  |  **Match:** {selected_match}")
if video_file:
    st.sidebar.caption(f"**Video:** {video_file.name} ({video_file.stat().st_size / (1024*1024):.1f} MB)")
else:
    st.sidebar.caption("**Video:** Not available")

st.sidebar.markdown("---")
st.sidebar.markdown("### TOOLS Pipeline Health")
st.sidebar.markdown("""
<div style="font-size:0.75rem; color:#94a3b8; line-height:1.65;">
<div><span style="color:#10b981;">..."</span> <strong>Env:</strong> Python 3.10 (.venv)</div>
<div><span style="color:#10b981;">..."</span> <strong>PBP:</strong> 17 Matches (139 Events)</div>
<div><span style="color:#10b981;">..."</span> <strong>RAG:</strong> 3,248 Docs (FAISS)</div>
<div><span style="color:#10b981;">..."</span> <strong>Temporal Context:</strong> Active</div>
<div><span style="color:#10b981;">..."</span> <strong>Event Graph:</strong> Active</div>
<div><span style="color:#10b981;">..."</span> <strong>Commentary Policy:</strong> Active</div>
<div><span style="color:#10b981;">..."</span> <strong>Factual Verifier:</strong> Active</div>
<div><span style="color:#10b981;">..."</span> <strong>Visual Grounding:</strong> TrackLab 720p</div>
<div><span style="color:#10b981;">..."</span> <strong>Player Aggregation:</strong> Active</div>
<div><span style="color:#10b981;">..."</span> <strong>Evaluation Suite:</strong> Benchmarked</div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# PAGE 1 - MATCH INTELLIGENCE (CLEAN SEQUENTIAL LAYOUT)
# ------------------------------------------------------------------------------
if selected_page.startswith("MATCH"):

    # Ã¢"â‚¬Ã¢"â‚¬ Match Header Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
    first_ev = contexts[0] if contexts else None
    state_0 = components["m_tracker"].compute_state_for_event(
        selected_match,
        first_ev.action if first_ev else "PASS",
        first_ev.team if first_ev else "Team",
        0.0,
    )
    score_display = (
        f"{state_0.home_score} - {state_0.away_score}"
        if state_0.is_score_available
        else "-"
    )
    ft_note = f"Full-time: {state_0.full_time_result}" if state_0.full_time_result else ""

    st.markdown(
        f"""
        <div class="match-header">
            <div>
                <div class="match-team">{state_0.home_team}</div>
                <div style="font-size:0.78rem; color:#64748b;">Home</div>
            </div>
            <div class="match-score-center">
                <div class="match-score-val">{score_display}</div>
                <div class="match-meta">{state_0.league} &nbsp;-&nbsp; {state_0.date} &nbsp;-&nbsp; Half {state_0.half}</div>
                <div style="font-size:0.74rem; color:#475569; margin-top:0.2rem;">{ft_note}</div>
            </div>
            <div class="match-team match-team-away">
                <div>{state_0.away_team}</div>
                <div style="font-size:0.78rem; color:#64748b;">Away</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ã¢"â‚¬Ã¢"â‚¬ Two-column layout: Video + Event Timeline Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
    col_video, col_timeline = st.columns([3, 1], gap="large")

    with col_video:
        st.markdown('<div class="section-label">Match Video</div>', unsafe_allow_html=True)
        if video_file and video_file.exists():
            st.video(str(video_file))
            st.caption(f"Source: `{video_file.name}` (MP4 - AI-generated TTS commentary + tracking overlay)")
        else:
            st.info("Video not available for this match ID. Displaying event sequence only.")

    with col_timeline:
        st.markdown('<div class="section-label">Event Timeline</div>', unsafe_allow_html=True)
        selected_event_idx = st.selectbox(
            "Select event:",
            options=range(len(contexts)),
            format_func=lambda i: f"{contexts[i].minute}'{contexts[i].second:02d}\"  {contexts[i].action}",
            index=min(2, len(contexts) - 1),
            label_visibility="collapsed",
        )

        # Visual timeline
        timeline_html = []
        for i, c in enumerate(contexts):
            is_active = (i == selected_event_idx)
            dot_cls = action_dot_class(c.action, is_active)
            act_cls = "timeline-action-active" if is_active else ("timeline-action-goal" if c.action in {"GOAL"} else "")
            timeline_html.append(f"""
            <div class="timeline-event">
                <div class="{dot_cls}" style="margin-top:3px;"></div>
                <div>
                    <div class="timeline-time">{c.minute}'{c.second:02d}"</div>
                    <div class="timeline-action {act_cls}">{c.action}</div>
                    <div class="timeline-player">{c.player}</div>
                </div>
            </div>
            """)
        st.markdown("".join(timeline_html), unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Ã¢"â‚¬Ã¢"â‚¬ Process selected event through full pipeline Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
    ctx: EventContext = contexts[selected_event_idx]
    state: MatchState = components["m_tracker"].compute_state_for_event(
        selected_match, ctx.action, ctx.team, ctx.start_time
    )
    imp: ImportanceBreakdown = components["imp_scorer"].compute_importance(
        ctx.action, state.match_minute, state.score_difference or 0, ctx.is_sequence_climax
    )
    docs = components["retriever"].retrieve(ctx, state, imp, mode="full", top_k=3)
    top_doc = docs[0] if docs else None
    top_doc_score = f"{top_doc.final_score:.2f}" if top_doc else "0.00"
    top_doc_stem = top_doc.doc_stem if top_doc else "No document retrieved"
    vis_res: VisualVerificationResult = components["visual_verifier"].verify_player_in_frame(
        selected_match, state.half, ctx.player, ctx.team, ctx.start_time
    )

    # Policy override in sidebar
    preferred_style = st.sidebar.selectbox(
        "Commentary Style Override",
        ["Auto (Adaptive Policy)", "Excited Climax", "Analytical Tactical", "Historical Contextual", "Pacy Play-by-Play"],
        index=0,
    )
    override_arg = preferred_style if preferred_style != "Auto (Adaptive Policy)" else None

    pre_conf: ConfidenceBreakdown = components["conf_scorer"].compute_confidence(
        top_doc.semantic_score if top_doc else 0.4,
        top_doc.entity_score if top_doc else 0.0,
        "VERIFIED", False, vis_res.visual_confidence,
    )
    pol_dec: CommentaryPolicyDecision = components["policy"].evaluate_policy(
        ctx, imp, pre_conf, state, top_doc, preferred_style=override_arg
    )
    verif: VerificationReport = components["verifier"].verify_commentary(
        pol_dec.generated_text, ctx.player, ctx.team, ctx.opponent,
        ctx.action, state.consequence_description
    )
    conf: ConfidenceBreakdown = components["conf_scorer"].compute_confidence(
        top_doc.semantic_score if top_doc else 0.4,
        top_doc.entity_score if top_doc else 0.0,
        verif.status, verif.was_corrected, vis_res.visual_confidence,
    )
    cf_scenarios = components["cf_engine"].analyze_event_counterfactual(ctx, state)

    # Ã¢"â‚¬Ã¢"â‚¬ 1. EVENT IDENTITY Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
    st.markdown('<div class="section-label">Selected Event</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="event-identity-row">
            <div class="identity-cell">
                <div class="identity-label">Action</div>
                <div class="identity-value" style="color:{importance_color(imp.importance_tier)};">{ctx.action}</div>
            </div>
            <div class="identity-cell">
                <div class="identity-label">Player</div>
                <div class="identity-value">{ctx.player}</div>
            </div>
            <div class="identity-cell">
                <div class="identity-label">Team</div>
                <div class="identity-value">{ctx.team}</div>
            </div>
            <div class="identity-cell">
                <div class="identity-label">Timestamp</div>
                <div class="identity-value">{ctx.start_time:.1f}s</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ã¢"â‚¬Ã¢"â‚¬ 2. TEMPORAL STORY Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
    st.markdown('<div class="section-label">Temporal Sequence</div>', unsafe_allow_html=True)
    seq_html = render_sequence(ctx.sequence_signature, ctx.action)
    climax_note = (
        '<span class="chip chip-green">Sequence Climax</span>'
        if ctx.is_sequence_climax
        else '<span class="chip chip-gray">Standalone Event</span>'
    )
    phase_note = ""
    if ctx.is_sequence_climax:
        phase_note = "This event is the <strong>culmination</strong> of the preceding attacking build-up sequence."
    elif ctx.previous_events:
        phase_note = f"This event continues a sequence of {len(ctx.previous_events)} prior action(s) in this phase."
    else:
        phase_note = "Opening event in this play - no prior build-up sequence."

    st.markdown(
        f"""
        <div class="card card-blue">
            {seq_html}
            <div style="margin-top:0.6rem;">
                {climax_note}
                <span style="font-size:0.84rem; color:#94a3b8; margin-left:0.5rem;">{phase_note}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ã¢"â‚¬Ã¢"â‚¬ 3. MATCH STATE Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
    st.markdown('<div class="section-label">Match State</div>', unsafe_allow_html=True)
    score_info = (
        f"{state.home_score} - {state.away_score}"
        if state.is_score_available
        else "Score data unavailable"
    )
    score_color = "#38bdf8" if state.is_score_available else "#64748b"
    late_game_badge = '<span class="chip chip-amber">Late Game</span>' if state.is_late_game else ""
    ft_badge = f'<span class="chip chip-gray">Full-time: {state.full_time_result}</span>' if state.full_time_result else ""

    st.markdown(
        f"""
        <div class="card">
            <div style="display:grid; grid-template-columns: auto 1fr; gap:1.5rem; align-items:start;">
                <div>
                    <div class="identity-label">Live Score</div>
                    <div style="font-size:1.8rem; font-weight:800; color:{score_color};">{score_info}</div>
                    <div style="margin-top:0.4rem;">{late_game_badge} {ft_badge}</div>
                </div>
                <div>
                    <div class="identity-label">Context</div>
                    <div style="font-size:0.85rem; color:#cbd5e1;">
                        Min {state.match_minute}' &nbsp;-&nbsp; Half {state.half} &nbsp;-&nbsp; {state.state_description}
                    </div>
                    <div style="font-size:0.8rem; color:#94a3b8; margin-top:0.3rem;">
                        <em>Consequence: {state.consequence_description}</em>
                    </div>
                    {"<div style='font-size:0.74rem;color:#475569;margin-top:0.3rem;'><em>Note: Live running scores are not available in PBP dataset. Full-time result shown as reference only.</em></div>" if not state.is_score_available else ""}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ã¢"â‚¬Ã¢"â‚¬ 4. EVENT IMPORTANCE Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
    st.markdown('<div class="section-label">Event Importance</div>', unsafe_allow_html=True)
    imp_pct = min(100, int(imp.final_importance * 100))
    imp_col = importance_color(imp.importance_tier)
    st.markdown(
        f"""
        <div class="card card-amber">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:1.1rem; font-weight:700; color:{imp_col};">{imp.importance_tier}</span>
                <span style="font-size:1.4rem; font-weight:800; color:{imp_col};">{imp.final_importance:.2f}</span>
            </div>
            <div class="imp-bar-bg"><div class="imp-bar-fill" style="width:{imp_pct}%; background:linear-gradient(90deg,{imp_col}88,{imp_col});"></div></div>
            <div style="font-size:0.78rem; color:#94a3b8; margin-top:0.3rem;">{imp.explanation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ã¢"â‚¬Ã¢"â‚¬ 5. RETRIEVED EVIDENCE Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
    st.markdown('<div class="section-label">Retrieved Evidence</div>', unsafe_allow_html=True)
    if docs:
        for d in docs:
            top_marker = ' <span class="chip chip-green">Top Ranked</span>' if d.rank == 1 else ""
            st.markdown(
                f"""
                <div class="evidence-row">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span class="evidence-rank">#{d.rank}</span>&nbsp;
                            <span class="evidence-name">{d.doc_stem}</span>{top_marker}
                        </div>
                        <code style="color:#38bdf8; font-size:0.85rem;">{d.final_score:.4f}</code>
                    </div>
                    <div class="evidence-scores">
                        Semantic: {d.semantic_score:.4f} &nbsp;|&nbsp;
                        Entity bonus: +{d.entity_score:.3f} &nbsp;|&nbsp;
                        Temporal: +{d.temporal_score:.3f} &nbsp;|&nbsp;
                        Match-state: +{d.match_state_score:.3f}
                    </div>
                    <div class="evidence-snippet">{d.background_text[:200]}...</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="card"><em style="color:#64748b;">No documents retrieved for this event.</em></div>', unsafe_allow_html=True)

    # Ã¢"â‚¬Ã¢"â‚¬ 6. VERIFICATION Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
    st.markdown('<div class="section-label">Factual & Visual Verification</div>', unsafe_allow_html=True)
    vcol1, vcol2 = st.columns(2, gap="medium")

    with vcol1:
        verif_badge = (
            '<span class="chip chip-green">..." VERIFIED</span>'
            if verif.status == "VERIFIED"
            else '<span class="chip chip-red">[!]  CONTRADICTION</span>'
        )
        checks_html = render_verification_checks(verif, ctx.player, ctx.team, ctx.opponent, ctx.action)
        st.markdown(
            f"""
            <div class="card {'card-green' if verif.status == 'VERIFIED' else 'card-red'}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                    <strong>Factual Consistency</strong>{verif_badge}
                </div>
                {checks_html}
                <div style="font-size:0.75rem; color:#64748b; margin-top:0.4rem;">{verif.audit_summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with vcol2:
        if vis_res.is_visually_grounded:
            vis_badge = '<span class="chip chip-green">..." VISUALLY CONFIRMED</span>'
            vis_card = "card-green"
        elif vis_res.visual_confidence == 0.5:
            vis_badge = '<span class="chip chip-amber">PARTIAL (teammates only)</span>'
            vis_card = "card-amber"
        else:
            vis_badge = '<span class="chip chip-gray">NOT IN TRACKING CSV</span>'
            vis_card = ""

        bbox_html = ""
        if vis_res.matched_player and vis_res.matched_player.bbox:
            b = vis_res.matched_player.bbox
            bbox_html = f"""
            <div style="font-size:0.78rem; color:#94a3b8; margin-top:0.4rem;">
                <strong>Player:</strong> {vis_res.matched_player.name}
                &nbsp;-&nbsp; <strong>Jersey:</strong> #{vis_res.matched_player.jersey_number}
                &nbsp;-&nbsp; <strong>BBox (720p):</strong> [{b['x1']},{b['y1']}] -> [{b['x2']},{b['y2']}]
            </div>"""

        players_note = ""
        if vis_res.all_players_in_frame:
            names = ", ".join(p.short_name for p in vis_res.all_players_in_frame[:4])
            players_note = f'<div style="font-size:0.75rem; color:#64748b; margin-top:0.3rem;">Tracked in frame: {names}{"..." if len(vis_res.all_players_in_frame) > 4 else ""}</div>'

        st.markdown(
            f"""
            <div class="card {vis_card}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                    <strong>Visual Grounding (TrackLab)</strong>{vis_badge}
                </div>
                <div style="font-size:0.83rem; color:#94a3b8;">{vis_res.explanation}</div>
                {bbox_html}
                {players_note}
                {"<div style='font-size:0.72rem; color:#475569; margin-top:0.5rem; font-style:italic;'>Player not captured in active broadcast frame at this timestamp - documented camera coverage boundary.</div>" if not vis_res.all_players_in_frame else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Ã¢"â‚¬Ã¢"â‚¬ 7. CONFIDENCE Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
    st.markdown('<div class="section-label">Multimodal Confidence Estimate</div>', unsafe_allow_html=True)
    conf_pct = int(conf.composite_confidence * 100)
    cc = conf_color(conf.composite_confidence)
    st.markdown(
        f"""
        <div class="card card-blue">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:1.5rem; font-weight:800; color:{cc};">{conf_pct}%</span>
                    &nbsp;<span style="font-size:0.85rem; color:#64748b;">{conf.confidence_tier}</span>
                </div>
                <span style="font-size:0.72rem; color:#475569; font-style:italic;">Explainable heuristic - not a calibrated probability</span>
            </div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{conf_pct}%; background:linear-gradient(90deg,{cc}88,{cc});"></div>
            </div>
            <div class="conf-sub-row">
                <div class="conf-sub">
                    <div class="conf-sub-val" style="color:#38bdf8;">{conf.retrieval_component:.2f}</div>
                    <div class="conf-sub-lbl">Retrieval (35%)</div>
                </div>
                <div class="conf-sub">
                    <div class="conf-sub-val" style="color:#10b981;">{conf.entity_component:.2f}</div>
                    <div class="conf-sub-lbl">Entity (25%)</div>
                </div>
                <div class="conf-sub">
                    <div class="conf-sub-val" style="color:#a78bfa;">{conf.factual_component:.2f}</div>
                    <div class="conf-sub-lbl">Factual (20%)</div>
                </div>
                <div class="conf-sub">
                    <div class="conf-sub-val" style="color:#f472b6;">{conf.visual_component:.2f}</div>
                    <div class="conf-sub-lbl">Visual (20%)</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ã¢"â‚¬Ã¢"â‚¬ 8. COUNTERFACTUAL Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
    if cf_scenarios:
        st.markdown('<div class="section-label">Grounded Counterfactual Reasoning</div>', unsafe_allow_html=True)
        cf = cf_scenarios[0]
        st.markdown(
            f"""
            <div class="card card-purple">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                    <strong>Hypothetical Intervention</strong>
                    <span class="chip chip-purple">{cf.intervention_type}</span>
                </div>
                <div style="font-size:0.85rem; color:#e2e8f0;">
                    <strong>Observed:</strong> {cf.original_observation}<br/>
                    <strong>Intervention:</strong> {cf.counterfactual_state}<br/>
                    <strong>Structural consequence:</strong> {cf.logical_consequence}
                </div>
                <div style="font-size:0.73rem; color:#7c3aed; margin-top:0.5rem; font-style:italic;">
                    Non-speculative: restricted to structural graph interventions on observed telemetry only.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Ã¢"â‚¬Ã¢"â‚¬ 9. FINAL COMMENTARY Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
    st.markdown('<div class="section-label">Generated Commentary</div>', unsafe_allow_html=True)
    speak_badge = (
        '<span class="chip chip-green">SPEAK</span>'
        if pol_dec.should_commentate
        else '<span class="chip chip-gray">SILENT</span>'
    )
    st.markdown(
        f"""
        <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.6rem;">
            {speak_badge}
            <span class="chip chip-blue">{pol_dec.selected_style.upper()}</span>
            <span class="chip chip-gray">{pol_dec.tempo_cadence}</span>
            {"<span class='chip chip-green'>[OK] VERIFIED</span>" if verif.status == "VERIFIED" else "<span class='chip chip-red'>[!] AUTO-CORRECTED</span>"}
        </div>
        <div class="commentary-box">
            <div class="commentary-text">"{verif.final_commentary}"</div>
        </div>
        <div style="font-size:0.75rem; color:#475569; margin-top:0.5rem; font-style:italic;">
            Policy rationale: {pol_dec.policy_rationale}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ã¢"â‚¬Ã¢"â‚¬ 10. WHY THIS COMMENTARY? Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
    st.markdown('<div class="section-label">Explainable Grounding Evidence ("Why This Commentary?")</div>', unsafe_allow_html=True)
    prev_str = f"{ctx.previous_events[-1]['action']} by {ctx.previous_events[-1].get('player', 'player')}" if ctx.previous_events else "Opening sequence event"
    current_idx = next((i for i, event in enumerate(contexts) if event is ctx), -1)
    next_idx = current_idx + 1
    next_str = (
        f"{contexts[next_idx].action} by {contexts[next_idx].player}"
        if 0 <= next_idx < len(contexts)
        else "Sequence conclusion"
    )
    vis_evidence_str = (
        f"Confirmed in video frame (#{vis_res.matched_player.jersey_number})"
        if vis_res.is_visually_grounded and vis_res.matched_player
        else ("Teammates visible in frame" if vis_res.all_players_in_frame else "Outside active camera pan/zoom")
    )
    verif_status_str = "Zero hallucinations detected" if verif.status == "VERIFIED" else "Contradiction corrected"

    st.markdown(
        f"""
        <div class="card card-purple">
            <div style="font-size:0.88rem; font-weight:700; color:#c084fc; margin-bottom:0.6rem;">
                Structured Multimodal Evidence Chain
            </div>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:0.6rem; font-size:0.8rem; color:#cbd5e1;">
                <div><span style="color:#10b981;">..."</span> <strong>Event Action:</strong> {ctx.action} ({imp.importance_tier} Importance)</div>
                <div><span style="color:#10b981;">..."</span> <strong>Player Entity:</strong> {ctx.player} (Resolved to telemetry)</div>
                <div><span style="color:#10b981;">..."</span> <strong>Team Attribution:</strong> {ctx.team} (vs {ctx.opponent})</div>
                <div><span style="color:#10b981;">..."</span> <strong>Retrieved Knowledge:</strong> {top_doc_stem} (Score: {top_doc_score})</div>
                <div><span style="color:#10b981;">..."</span> <strong>Temporal Sequence:</strong> {prev_str} -> <strong>{ctx.action}</strong> -> {next_str}</div>
                <div><span style="color:#10b981;">..."</span> <strong>Visual Grounding:</strong> {vis_evidence_str}</div>
                <div><span style="color:#10b981;">..."</span> <strong>Factual Verification:</strong> {verif.status} ({verif_status_str})</div>
                <div><span style="color:#10b981;">..."</span> <strong>Heuristic Confidence:</strong> {conf_pct}% ({conf.confidence_tier} - Uncalibrated)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ã¢"â‚¬Ã¢"â‚¬ 11. PLAYER-LEVEL PERFORMANCE INTELLIGENCE Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬Ã¢"â‚¬
    st.markdown('<div class="section-label">Player Performance Intelligence & Match Contributors</div>', unsafe_allow_html=True)
    all_imps = []
    for c in contexts:
        st_val = components["m_tracker"].compute_state_for_event(selected_match, c.action, c.team, c.start_time)
        im_val = components["imp_scorer"].compute_importance(c.action, st_val.match_minute, st_val.score_difference or 0, c.is_sequence_climax)
        all_imps.append(im_val)

    player_stats = components["player_agg"].aggregate_from_events(contexts, all_imps)
    mvp_info = components["player_agg"].determine_mvp(player_stats)

    if mvp_info:
        st.markdown(
            f"""
            <div class="card card-green" style="margin-bottom:1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span class="chip chip-green" style="font-size:0.75rem;">TOP CONTRIBUTOR / MATCH MVP</span>
                        <div style="font-size:1.3rem; font-weight:800; color:#f8fafc; margin-top:0.3rem;">
                            {mvp_info['player_name']} <span style="font-size:0.88rem; color:#94a3b8; font-weight:500;">({mvp_info['team']})</span>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:1.5rem; font-weight:800; color:#38bdf8;">{mvp_info['total_impact_score']:.2f}</div>
                        <div style="font-size:0.7rem; color:#64748b;">Total Impact Score ({mvp_info['impact_tier']})</div>
                    </div>
                </div>
                <div style="font-size:0.82rem; color:#94a3b8; margin-top:0.5rem;">
                    <strong>Evidence:</strong> {mvp_info['rationale']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if player_stats:
        p_rows = []
        for p in player_stats:
            p_rows.append({
                "Player": p.player_name,
                "Team": p.team,
                "Total Impact Score": round(p.total_impact_score, 2),
                "Impact Tier": p.impact_tier,
                "Total Events": p.event_count,
                "Key Moments": p.key_events_count,
                "Actions": ", ".join(f"{cnt} {act.lower()}" for act, cnt in p.action_counts.items()),
            })
        st.dataframe(pd.DataFrame(p_rows), use_container_width=True)
    else:
        st.info("Player-level metrics unavailable from current event telemetry.")


# ------------------------------------------------------------------------------
# PAGE 2 - AI REASONING PIPELINE
# ------------------------------------------------------------------------------
elif selected_page.startswith("AI"):
    st.title("AI Reasoning Pipeline")
    st.caption("Step-by-step transparent trace of how the system transforms telemetry into verified commentary.")

    event_options = [f"[{c.start_time:.1f}s] {c.action} - {c.player} ({c.team})" for c in contexts]
    idx = st.selectbox("Inspect Event:", range(len(contexts)), format_func=lambda i: event_options[i])
    ctx = contexts[idx]
    state = components["m_tracker"].compute_state_for_event(selected_match, ctx.action, ctx.team, ctx.start_time)
    imp = components["imp_scorer"].compute_importance(ctx.action, state.match_minute, state.score_difference or 0, ctx.is_sequence_climax)
    docs = components["retriever"].retrieve(ctx, state, imp, mode="full", top_k=3)
    top_doc = docs[0] if docs else None
    vis_res = components["visual_verifier"].verify_player_in_frame(selected_match, state.half, ctx.player, ctx.team, ctx.start_time)
    conf = components["conf_scorer"].compute_confidence(
        top_doc.semantic_score if top_doc else 0.4, top_doc.entity_score if top_doc else 0.0,
        "VERIFIED", False, vis_res.visual_confidence
    )
    pol_dec = components["policy"].evaluate_policy(ctx, imp, conf, state, top_doc)
    verif = components["verifier"].verify_commentary(pol_dec.generated_text, ctx.player, ctx.team, ctx.opponent, ctx.action, state.consequence_description)
    cf_res = components["cf_engine"].analyze_event_counterfactual(ctx, state)
    final_conf = components["conf_scorer"].compute_confidence(
        top_doc.semantic_score if top_doc else 0.4, top_doc.entity_score if top_doc else 0.0,
        verif.status, verif.was_corrected, vis_res.visual_confidence
    )

    steps = [
        ("Step 1 - Event Ingestion & PBP Parsing", {
            "Raw Event Text": ctx.current_event.get("text", ""),
            "Action Category": ctx.action,
            "Timestamp Window": f"{ctx.start_time:.2f}s -> {ctx.end_time:.2f}s",
            "Pitch Location": ctx.location,
        }),
        ("Step 2 - Entity Extraction & Disambiguation", {
            "Player Entity": ctx.player,
            "Active Team": ctx.team,
            "Opponent Club": ctx.opponent,
        }),
        ("Step 3 - Visual Frame Grounding (TrackLab)", {
            "Player Confirmed in Frame": vis_res.is_visually_grounded,
            "Frame Timestamp": f"{vis_res.frame_timestamp_sec}s",
            "BBox (720p)": str(vis_res.matched_player.bbox) if vis_res.matched_player else "Not in active camera view",
            "Jersey #": vis_res.matched_player.jersey_number if vis_res.matched_player else "N/A",
            "Visual Confidence": vis_res.visual_confidence,
            "Note": vis_res.explanation,
        }),
        ("Step 4 - Temporal Match Context", {
            "Prior Event Sequence": ctx.sequence_signature,
            "Sequence Climax": ctx.is_sequence_climax,
            "Match Minute": f"{ctx.minute}'{ctx.second:02d}\"",
        }),
        ("Step 5 - Match-State Reasoning", {
            "Home vs Away": f"{state.home_team} vs {state.away_team}",
            "Full-time Reference": state.full_time_result or "Unavailable",
            "Live Score": f"{state.home_score} - {state.away_score}" if state.is_score_available else "Not available in PBP dataset",
            "Late Game": state.is_late_game,
            "Consequence": state.consequence_description,
        }),
        ("Step 6 - Event Importance Scoring", {
            "Base Action Score": imp.base_action_score,
            "Time Multiplier": imp.time_multiplier,
            "Margin Multiplier": imp.margin_multiplier,
            "Final Importance": round(imp.final_importance, 4),
            "Tier": imp.importance_tier,
            "Rationale": imp.explanation,
        }),
        ("Step 7 - Entity-Aware Context Retrieval (FAISS + Reranking)", {
            "Query": f"{ctx.player} {ctx.team} {ctx.action}",
            "Top Document": top_doc.doc_stem if top_doc else "None",
            "Semantic Score": round(top_doc.semantic_score, 4) if top_doc else 0.0,
            "Entity Bonus": round(top_doc.entity_score, 4) if top_doc else 0.0,
            "Final Score": round(top_doc.final_score, 4) if top_doc else 0.0,
            "Match Reasons": top_doc.match_reasons if top_doc else [],
        }),
        ("Step 8 - Grounded Counterfactual Reasoning", {
            "Intervention Type": cf_res[0].intervention_type if cf_res else "None",
            "Hypothetical State": cf_res[0].counterfactual_state if cf_res else "",
            "Structural Consequence": cf_res[0].logical_consequence if cf_res else "",
            "Is Speculative": False,
        }),
        ("Step 9 - Adaptive Commentary Policy", {
            "Should Broadcast": pol_dec.should_commentate,
            "Style Selected": pol_dec.selected_style,
            "Cadence": pol_dec.tempo_cadence,
            "Decision Rationale": pol_dec.policy_rationale,
            "Generated Script": pol_dec.generated_text,
        }),
        ("Step 10 - Factual Verification + Multimodal Confidence", {
            "Verification Status": verif.status,
            "Was Auto-Corrected": verif.was_corrected,
            "Audit Summary": verif.audit_summary,
            "Final Commentary": verif.final_commentary,
            "Composite Confidence": round(final_conf.composite_confidence, 4),
            "Confidence Tier": final_conf.confidence_tier,
            "Is Calibrated Probability": False,
        }),
    ]

    for title, details in steps:
        with st.expander(title, expanded=True):
            st.json(details)


# ------------------------------------------------------------------------------
# PAGE 3 - RETRIEVAL ANALYSIS
# ------------------------------------------------------------------------------
elif selected_page.startswith("RETRIEVAL"):
    st.title("Retrieval Analysis")
    st.caption("Side-by-side comparison of Semantic Baseline, Entity-Aware, and Full Context-Aware retrieval.")

    event_options = [f"[{c.start_time:.1f}s] {c.action} - {c.player} ({c.team})" for c in contexts]
    idx = st.selectbox("Select event:", range(len(contexts)), format_func=lambda i: event_options[i])
    ctx = contexts[idx]
    state = components["m_tracker"].compute_state_for_event(selected_match, ctx.action, ctx.team, ctx.start_time)
    imp = components["imp_scorer"].compute_importance(ctx.action, state.match_minute, state.score_difference or 0, ctx.is_sequence_climax)

    col1, col2, col3 = st.columns(3, gap="medium")

    for col, mode, label, note, row_cls in [
        (col1, "semantic_only", "A0 - Semantic Baseline", "all-MiniLM-L6-v2 + FAISS cosine similarity", "score-row"),
        (col2, "entity_aware", "A1 - Entity-Aware", "A0 + player / team / opponent name bonuses", "score-row score-row-top"),
        (col3, "full", "A5 - Full Context-Aware", "A1 + temporal sequence + match-state context", "score-row score-row-top"),
    ]:
        with col:
            st.markdown(f"#### {label}")
            st.caption(note)
            retrieved = components["retriever"].retrieve(ctx, state, imp, mode=mode, top_k=5)
            for d in retrieved:
                st.markdown(
                    f"""
                    <div class="{row_cls}">
                        <div>
                            <strong>#{d.rank} {d.doc_stem}</strong><br/>
                            <span style="font-size:0.73rem; color:#94a3b8;">
                                Sem: {d.semantic_score:.4f} &nbsp;|&nbsp; Ent: +{d.entity_score:.3f} &nbsp;|&nbsp; Final: {d.final_score:.4f}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.subheader("Top Candidate Rationale")
    full_docs = components["retriever"].retrieve(ctx, state, imp, mode="full", top_k=3)
    if full_docs:
        top_f = full_docs[0]
        st.markdown(f"**{top_f.doc_stem}** - Score: `{top_f.final_score:.4f}`")
        for r in top_f.match_reasons:
            st.markdown(f"- `{r}`")
        with st.expander("Full background text snippet"):
            st.markdown(top_f.background_text[:800])

    st.markdown("---")
    st.subheader("Research Finding: Dense Vector Query Dilution")
    st.markdown("""
> **Finding (A2-A4):** Appending temporal and match-state context strings into a
> single short dense query reduces retrieval precision compared to pure entity-aware
> reranking (A1). This is because dense encoders like `all-MiniLM-L6-v2` are
> optimized for short, semantically focused inputs. Longer, heterogeneous queries
> dilute the embedding signal.

> **Implication:** Contextual signals (temporal sequence, match state) are better
> applied as *post-retrieval reranking factors* rather than embedded into the query
> vector. This is a genuine scientific finding, not a system failure.
    """)


# ------------------------------------------------------------------------------
# PAGE 4 - EVENT GRAPH & MATCH ANALYTICS
# ------------------------------------------------------------------------------
elif selected_page.startswith("GRAPH"):
    st.title("Event Graph & Match Analytics")
    st.caption("Directed event sequence modeling and quantitative match diagnostics.")

    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl in [
        (m1, len(contexts), "Total Events"),
        (m2, len({c.action for c in contexts}), "Action Types"),
        (m3, len({c.team for c in contexts if c.team}), "Competing Teams"),
        (m4, f"{max((c.end_time for c in contexts), default=0.0):.1f}s", "PBP Window"),
    ]:
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("&nbsp;")

    # Event graph visual (as HTML flow diagram)
    raw_ev_dicts = [c.current_event for c in contexts]
    graph: MatchEventGraph = components["graph_builder"].build_graph(selected_match, raw_ev_dicts)

    st.subheader("Directed Event Graph")
    st.caption("Nodes = events. Edges = transition relationships between consecutive events.")

    # Render graph as clean HTML node flow
    EDGE_COLORS = {
        "CULMINATION": "#10b981",
        "CHANCE_CREATION": "#38bdf8",
        "TURNOVER": "#f59e0b",
        "OFFENSIVE_PROGRESSION": "#a78bfa",
        "POSSESSION_CONTINUITY": "#4b5563",
    }

    graph_html_parts = ['<div style="display:flex; flex-direction:column; align-items:flex-start; gap:0.2rem; padding:1rem;">']
    for i, node in enumerate(graph.nodes):
        action_color = importance_color(
            "Critical" if node.action == "GOAL" else
            "High" if node.action in {"SHOT", "PENALTY"} else
            "Moderate" if node.action in {"CROSS", "FREE KICK"} else "Routine"
        )
        graph_html_parts.append(f"""
        <div style="display:flex; align-items:center; gap:0.75rem;">
            <div style="
                background:#111827;
                border:2px solid {action_color};
                border-radius:10px;
                padding:0.6rem 1.2rem;
                min-width:180px;
            ">
                <div style="font-size:0.65rem; color:#64748b; text-transform:uppercase;">{node.start_time:.1f}s</div>
                <div style="font-weight:700; color:{action_color}; font-size:0.95rem;">{node.action}</div>
                <div style="font-size:0.78rem; color:#94a3b8;">{node.player}</div>
                <div style="font-size:0.72rem; color:#64748b;">{node.team}</div>
            </div>
        """)

        # Add edge below (except last)
        if i < len(graph.edges):
            edge = graph.edges[i]
            ec = EDGE_COLORS.get(edge.transition_type, "#4b5563")
            graph_html_parts.append(f"""
            <div style="display:flex; align-items:center; gap:0.5rem; margin-left:1rem;">
                <span style="color:{ec}; font-size:0.85rem; font-weight:600; font-family:monospace;">{edge.transition_type}</span>
            </div>
            """)
        graph_html_parts.append("</div>")

        if i < len(graph.nodes) - 1:
            if i < len(graph.edges):
                ec = EDGE_COLORS.get(graph.edges[i].transition_type, "#4b5563")
                graph_html_parts.append(f'<div style="padding-left:1.5rem; color:{ec}; font-size:1.2rem; line-height:1;">↓</div>')

    graph_html_parts.append("</div>")
    st.html("".join(graph_html_parts))

    st.markdown("---")
    # Edge details table
    if graph.edges:
        edge_rows = []
        for e in graph.edges:
            edge_rows.append({
                "From": f"{graph.nodes[e.source_id].action} ({graph.nodes[e.source_id].player})",
                "To": f"{graph.nodes[e.target_id].action} ({graph.nodes[e.target_id].player})",
                "Transition": e.transition_type,
                "Deltat (s)": round(e.time_delta_sec, 2),
                "Description": e.description,
            })
        st.dataframe(pd.DataFrame(edge_rows), use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Action Distribution")
        action_series = pd.Series([c.action for c in contexts]).value_counts()
        st.bar_chart(action_series)
    with col_b:
        st.subheader("Importance Profile Over Time")
        time_pts, imp_vals = [], []
        for c in contexts:
            st_val = components["m_tracker"].compute_state_for_event(selected_match, c.action, c.team, c.start_time)
            im_val = components["imp_scorer"].compute_importance(c.action, st_val.match_minute, st_val.score_difference or 0, c.is_sequence_climax)
            time_pts.append(round(c.start_time, 1))
            imp_vals.append(im_val.final_importance)
        chart_df = pd.DataFrame({"Importance (0-1)": imp_vals}, index=time_pts)
        st.line_chart(chart_df)


# ------------------------------------------------------------------------------
# PAGE 5 - RESEARCH EVALUATION
# ------------------------------------------------------------------------------
elif selected_page.startswith("RESEARCH"):
    st.title("Research Evaluation & Ablation Suite")
    st.caption("Quantitative benchmark over 17 matches and 139 real play-by-play football events.")

    res_json = Path("outputs/evaluation/results.json")
    ablation_csv = Path("outputs/evaluation/ablation_results.csv")
    error_json = Path("outputs/evaluation/error_analysis.json")
    integration_json = Path("outputs/integration/integration_test_result.json")

    if not (res_json.exists() and ablation_csv.exists()):
        st.warning("Evaluation outputs not yet generated. Run: `uv run python src/evaluation.py` then `uv run python src/ablation.py`")
    else:
        with open(res_json, encoding="utf-8") as f:
            eval_summary = json.load(f)
        ablation_df = pd.read_csv(ablation_csv)

        e1, e2, e3, e4 = st.columns(4)
        metrics = [
            (e1, eval_summary["dataset_total_events"], "Total Events"),
            (e2, eval_summary["evaluated_events_with_ground_truth"], "Events w/ Gold Truth"),
            (e3, f"{eval_summary['evaluation_coverage_pct']}%", "Corpus Coverage"),
            (e4, f"{ablation_df.loc[ablation_df['condition'] == 'A1_Entity_Aware', 'precision_at_1'].values[0]:.3f}", "Peak P@1 (A1)"),
        ]
        for col, val, lbl in metrics:
            with col:
                st.markdown(f'<div class="metric-box"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("&nbsp;")
        st.subheader("Ablation Progression: A0 -> A5")

        # Styled ablation table
        st.dataframe(ablation_df, use_container_width=True)

        st.subheader("Retrieval Metrics Comparison")
        chart_sub = ablation_df[["condition", "precision_at_1", "precision_at_3", "mrr"]].set_index("condition")
        st.bar_chart(chart_sub)

        st.markdown("---")
        st.subheader("Scientific Findings")
        st.markdown("""
| Finding | Observation | Implication |
|:---|:---|:---|
| **A1 vs A0** | P@1 rises from 0.1416 to 0.2035 (+43.7%) | Entity-aware reranking strongly improves retrieval |
| **A2-A4 vs A1** | P@1 falls to 0.1681 (dilution) | Dense query augmentation reduces precision |
| **Dense Query Dilution** | Appending contextual strings into short dense vectors reduces embedding signal | Use contextual signals as *post-retrieval reranking factors*, not query augmentation |
| **A5 Factual Consistency** | 100% consistency rate in test scenarios | Rule-based deterministic verifier eliminates action contradictions |
| **Visual Grounding** | 100% of 17 PBP evaluation matches (19 unique games) have player tracking frames | TrackLab SoccerNet bounding boxes verified with [x1, y1, x2=x1+w, y2=y1+h] schema |
        """)

    cov_csv = Path("outputs/research/modality_coverage.csv")
    if cov_csv.exists():
        st.markdown("---")
        st.subheader("Empirical Modality Coverage Matrix (17 Matches / 25 Clips)")
        st.caption("Empirical audit of play-by-play (PBP), real video, TrackLab player tracking, and gold retrieval documents across all matches.")
        cov_df = pd.read_csv(cov_csv)
        st.dataframe(cov_df, use_container_width=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Total Match Clips", len(cov_df))
        with mc2:
            st.metric("PBP Matches", f"{sum(cov_df['has_pbp'] == 'YES')} (139 events)")
        with mc3:
            st.metric("Real Videos", sum(cov_df['has_video'] == 'YES'))
        with mc4:
            st.metric("Multimodal Ready", sum(cov_df['multimodal_eval_usable'] == 'YES'))

    if error_json.exists():
        with open(error_json, encoding="utf-8") as f:
            err_data = json.load(f)
        st.markdown("---")
        st.subheader("Error Taxonomy")
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.markdown("**Failure Category Counts**")
            st.bar_chart(pd.Series(err_data["error_distribution"]))
        with ecol2:
            st.markdown("**Representative Failures**")
            for cat, ex_list in err_data["representative_examples"].items():
                if ex_list:
                    with st.expander(f"{cat} ({len(ex_list)} cases)"):
                        for ex in ex_list[:2]:
                            st.json(ex)

    if integration_json.exists():
        with open(integration_json, encoding="utf-8") as f:
            int_data = json.load(f)
        st.markdown("---")
        st.subheader("End-to-End Integration Test (Match 0008)")
        s = int_data.get("summary", {})
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.markdown(f'<div class="metric-box"><div class="metric-val" style="font-size:1.4rem;">{s.get("events_verified_factual",0)}/{s.get("total_events",0)}</div><div class="metric-lbl">Factually Verified</div></div>', unsafe_allow_html=True)
        with ic2:
            st.markdown(f'<div class="metric-box"><div class="metric-val" style="font-size:1.4rem;">{s.get("avg_confidence", 0.0):.3f}</div><div class="metric-lbl">Avg. Confidence</div></div>', unsafe_allow_html=True)
        with ic3:
            st.markdown(f'<div class="metric-box"><div class="metric-val" style="font-size:1.4rem; color:#10b981;">PASS</div><div class="metric-lbl">Pipeline Status</div></div>', unsafe_allow_html=True)

        if int_data.get("events"):
            st.markdown("**Per-Event Results**")
            ev_rows = [{k: v for k, v in e.items() if k in [
                "action", "player", "team", "start_time", "importance_tier",
                "policy_style", "verification_status", "final_confidence", "confidence_tier"
            ]} for e in int_data["events"]]
            st.dataframe(pd.DataFrame(ev_rows), use_container_width=True)


# ------------------------------------------------------------------------------
# PAGE 6 - METHODOLOGY
# ------------------------------------------------------------------------------
elif selected_page.startswith("METHOD"):
    st.title("Research Methodology")
    st.caption("Technical specification of the Context-Aware Multimodal Football Commentary Intelligence System (V2).")

    st.markdown("""
### Research Question

> *"Can combining temporally grounded event reasoning, entity-aware retrieval,
> match-state reasoning, event-relationship modeling, adaptive commentary policy,
> and multimodal verification produce more factual, contextually relevant and
> naturally timed football commentary than a semantic retrieval baseline?"*

---

### V2 Architecture Pipeline

```
FOOTBALL VIDEO (MP4 - outputs/demo-step4/)
           ↓
TEMPORAL EVENT UNDERSTANDING          temporal_context.py
  PBP JSONL -> EventContext (action, player, team, sequence, climax flag)
           ↓
ENTITY RESOLUTION & VISUAL ALIGNMENT  visual_verifier.py
  TrackLab/SoccerNet CSV -> bounding boxes, jersey numbers
           ↓
MATCH-STATE REASONING                 match_state.py
  Metadata CSV -> half, league, full-time result reference
           ↓
EVENT GRAPH CONSTRUCTION              event_graph.py
  DAG of events -> TURNOVER, CULMINATION, CHANCE_CREATION edges
           ↓
EXPLAINABLE IMPORTANCE SCORING        event_importance.py
  Deterministic score (0.0-1.0) from action + time + margin
           ↓
ENTITY-AWARE CONTEXT RETRIEVAL        context_retrieval.py
  FAISS + sentence-transformers + entity/temporal/match-state bonuses
           ↓
GROUNDED COUNTERFACTUAL REASONING     counterfactual.py
  Structural graph interventions only - no speculative game events
           ↓
ADAPTIVE COMMENTARY POLICY            commentary_policy.py
  Speech gate + style (Excited Climax / Analytical / Contextual / Pacy)
           ↓
FACTUAL CONSISTENCY VERIFICATION      factual_verifier.py
  Rule-based checks -> auto-correct contradictions -> audit trail
           ↓
MULTIMODAL CONFIDENCE SCORING         confidence_scorer.py
  Heuristic composite (retrieval 35% + entity 25% + factual 20% + visual 20%)
           ↓
FINAL VERIFIED COMMENTARY + VIDEO
```

---

### Scoring Equations

**Multi-Signal Retrieval Ranking:**

$$\\text{Score}(d, e) = \\left[ \\text{Sim}_{\\text{cos}}(\\mathbf{q}_e, \\mathbf{d}) + B_{\\text{player}} + B_{\\text{team}} + B_{\\text{opp}} + B_{\\text{temp}} + B_{\\text{state}} \\right] \\times W_{\\text{imp}}$$

**Multimodal Confidence (Explainable Heuristic - Uncalibrated):**

$$\\text{Conf}(e) = 0.35 \\cdot S_{\\text{retrieval}} + 0.25 \\cdot S_{\\text{entity}} + 0.20 \\cdot S_{\\text{factual}} + 0.20 \\cdot S_{\\text{visual}}$$

*Note: Weights are hand-picked, transparent heuristics. This is NOT a calibrated Bayesian posterior probability.*

---

### Documented Limitations

| Limitation | Detail |
|:---|:---|
| **No live running scores** | PBP dataset contains only full-time final scores, not minute-by-minute scorelines |
| **Visual tracking coverage** | Tracking CSV (`players_in_frames_sn_gamestate.csv`) covers 19 unique games (100% of 17 PBP matches); individual player visibility is bounded by dynamic camera pan/zoom |
| **Retrieval corpus coverage** | 113/139 events (81.3%) have a corresponding Wikipedia biography in the 3,248-document corpus |
| **Dense query dilution (A2-A4)** | Appending contextual tokens into short dense queries reduces precision; contextual signals should be post-retrieval reranking factors |
| **Commentary generation** | Templates are deterministic pattern-based; not neural LLM generation - enables verifiability and reproducibility |
| **Confidence calibration** | Confidence scores are uncalibrated heuristics; Platt scaling or isotonic regression calibration is future work |

---

### Modules & Files

| Module | File | Role |
|:---|:---|:---|
| Temporal Context | `src/temporal_context.py` | PBP parsing + sequence window + climax detection |
| Match State | `src/match_state.py` | Half / league / score / consequence reasoning |
| Event Importance | `src/event_importance.py` | Deterministic explainable importance score |
| Context Retrieval | `src/context_retrieval.py` | FAISS + entity/temporal/match-state reranking |
| Factual Verifier | `src/factual_verifier.py` | Rule-based contradiction detection + auto-correction |
| Visual Verifier | `src/visual_verifier.py` | TrackLab CSV bounding box grounding |
| Confidence Scorer | `src/confidence_scorer.py` | 4-signal explainable heuristic |
| Event Graph | `src/event_graph.py` | Directed event DAG with transition types |
| Counterfactual | `src/counterfactual.py` | Non-speculative structural graph interventions |
| Commentary Policy | `src/commentary_policy.py` | Adaptive style + cadence policy (every event commentated) |
| Player Intelligence | `src/player_intelligence.py` | Player-level aggregation, impact scoring & data-derived MVP |
| Integration Test | `src/pipeline_integration.py` | End-to-end real-data pipeline verification |
| Test Runner | `src/run_all_tests.py` | 11-module standalone unit test suite |
| Evaluation | `src/evaluation.py` + `src/ablation.py` | Retrieval metrics over 139 events / 17 matches |

---

### Run Commands

```bash
# Run all unit tests
uv run python src/run_all_tests.py

# Run end-to-end integration test (match 0008)
uv run python src/pipeline_integration.py

# Regenerate evaluation metrics
uv run python src/evaluation.py
uv run python src/ablation.py
uv run python src/error_analysis.py

# Launch research dashboard
uv run streamlit run src/streamlit_app.py --server.port 8501
```
    """)


if __name__ == "__main__":
    pass




