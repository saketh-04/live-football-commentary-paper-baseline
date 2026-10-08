"""
player_intelligence.py

Player-Level Performance Intelligence & Event Aggregation Engine.

Aggregates structured event telemetry into player-level match contributions:
- Event counts and action breakdown (goals, shots, crosses, passes, tackles, etc.)
- Key / high-leverage event contribution
- Explainable Total Impact Score: sum(event_importance) over player actions
- Data-derived MVP / Top Contributors ranking
- Strictly derived from real event telemetry — no unsupported statistics (no fake xG/xA).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import collections

from temporal_context import EventContext
from event_importance import ImportanceBreakdown


@dataclass
class PlayerMatchStats:
    """Aggregated match contributions for a single player derived strictly from event data."""
    player_name: str
    team: str
    event_count: int
    action_counts: Dict[str, int]
    key_events_count: int
    total_impact_score: float
    avg_impact_score: float
    impact_tier: str  # 'High', 'Moderate', 'Routine'
    events_summary: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_name": self.player_name,
            "team": self.team,
            "event_count": self.event_count,
            "action_counts": self.action_counts,
            "key_events_count": self.key_events_count,
            "total_impact_score": round(self.total_impact_score, 3),
            "avg_impact_score": round(self.avg_impact_score, 3),
            "impact_tier": self.impact_tier,
            "events_summary": self.events_summary,
        }


class PlayerIntelligenceAggregator:
    """Aggregates match events and computes transparent player impact metrics."""

    def aggregate_from_events(
        self,
        contexts: List[EventContext],
        importances: List[ImportanceBreakdown],
    ) -> List[PlayerMatchStats]:
        """
        Aggregates a match sequence of EventContexts and ImportanceBreakdowns into PlayerMatchStats.
        """
        if not contexts:
            return []

        player_data: Dict[str, Dict[str, Any]] = collections.defaultdict(
            lambda: {
                "team": "",
                "actions": collections.Counter(),
                "key_events": 0,
                "impact_scores": [],
                "events_summary": [],
            }
        )

        for ctx, imp in zip(contexts, importances):
            p = ctx.player.strip()
            if not p or p.lower() in {"unknown", ""}:
                continue

            entry = player_data[p]
            if not entry["team"] and ctx.team:
                entry["team"] = ctx.team

            entry["actions"][ctx.action] += 1
            entry["impact_scores"].append(imp.final_importance)

            if imp.final_importance >= 0.65 or imp.importance_tier in {"Critical", "High"}:
                entry["key_events"] += 1

            summary_str = f"{ctx.start_time:.1f}s {ctx.action} ({imp.importance_tier}, imp={imp.final_importance:.2f})"
            entry["events_summary"].append(summary_str)

        results: List[PlayerMatchStats] = []
        for p_name, data in player_data.items():
            ev_count = len(data["impact_scores"])
            tot_impact = sum(data["impact_scores"])
            avg_impact = tot_impact / ev_count if ev_count > 0 else 0.0

            if tot_impact >= 1.0 or data["key_events"] >= 2:
                tier = "High"
            elif tot_impact >= 0.45 or data["key_events"] >= 1:
                tier = "Moderate"
            else:
                tier = "Routine"

            stats = PlayerMatchStats(
                player_name=p_name,
                team=data["team"] or "Unknown Club",
                event_count=ev_count,
                action_counts=dict(data["actions"]),
                key_events_count=data["key_events"],
                total_impact_score=tot_impact,
                avg_impact_score=avg_impact,
                impact_tier=tier,
                events_summary=data["events_summary"],
            )
            results.append(stats)

        # Sort descending by total impact score, then key events, then event count
        results.sort(
            key=lambda s: (s.total_impact_score, s.key_events_count, s.event_count),
            reverse=True,
        )
        return results

    def get_top_contributors(
        self,
        stats: List[PlayerMatchStats],
        top_n: int = 3,
    ) -> List[PlayerMatchStats]:
        """Returns the top N contributors ranked by data-derived impact."""
        return stats[:top_n]

    def determine_mvp(
        self,
        stats: List[PlayerMatchStats],
    ) -> Optional[Dict[str, Any]]:
        """
        Determines the MVP based strictly on data-derived impact metrics.
        Returns None if evidence is insufficient or no events recorded.
        """
        if not stats or stats[0].total_impact_score <= 0:
            return None

        top = stats[0]
        # Build explanation of the evidence
        action_parts = [f"{count} {act.lower()}{'s' if count > 1 else ''}" for act, count in top.action_counts.items()]
        actions_str = ", ".join(action_parts) if action_parts else "active play"

        rationale = (
            f"Ranked #1 with {top.total_impact_score:.2f} total impact score across {top.event_count} event(s) "
            f"({actions_str}), contributing {top.key_events_count} high-leverage key moment(s)."
        )

        return {
            "player_name": top.player_name,
            "team": top.team,
            "total_impact_score": round(top.total_impact_score, 3),
            "impact_tier": top.impact_tier,
            "event_count": top.event_count,
            "key_events_count": top.key_events_count,
            "action_counts": top.action_counts,
            "rationale": rationale,
        }


if __name__ == "__main__":
    from temporal_context import TemporalContextBuilder
    from event_importance import EventImportanceScorer
    from match_state import MatchStateTracker

    pbp = Path("data/demo/pbp/0008/play-by-play-en.jsonl")
    t_b = TemporalContextBuilder()
    imp_s = EventImportanceScorer()
    m_t = MatchStateTracker()

    contexts = t_b.build_match_contexts(pbp)
    importances = []
    for c in contexts:
        st = m_t.compute_state_for_event("0008", c.action, c.team, c.start_time)
        imp = imp_s.compute_importance(c.action, st.match_minute, 0, c.is_sequence_climax)
        importances.append(imp)

    agg = PlayerIntelligenceAggregator()
    player_stats = agg.aggregate_from_events(contexts, importances)

    print("=== Player Performance Intelligence Test (Match 0008) ===")
    print(f"Total Active Players Aggregated: {len(player_stats)}\n")
    for s in player_stats:
        print(f"Player: {s.player_name} ({s.team})")
        print(f"  Impact: {s.impact_tier} (Score: {s.total_impact_score:.3f}, Avg: {s.avg_impact_score:.3f})")
        print(f"  Events: {s.event_count} | Key Events: {s.key_events_count}")
        print(f"  Actions: {s.action_counts}")
        print(f"  Timeline: {', '.join(s.events_summary)}")
        print()

    mvp = agg.determine_mvp(player_stats)
    if mvp:
        print("=== Top Contributor / MVP ===")
        print(f"MVP: {mvp['player_name']} ({mvp['team']})")
        print(f"Impact: {mvp['total_impact_score']} ({mvp['impact_tier']})")
        print(f"Rationale: {mvp['rationale']}")
    else:
        print("MVP ranking unavailable from current evidence.")
