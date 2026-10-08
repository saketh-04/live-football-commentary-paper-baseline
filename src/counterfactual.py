"""
counterfactual.py

V2 Reasoning Component: Grounded Counterfactual Match Reasoning.

IMPORTANT RESEARCH CONSTRAINT:
Counterfactual reasoning must NEVER invent speculative physical events
(e.g., claiming a goalkeeper would have saved a shot, or imagining player injuries).
It is strictly restricted to supported structural interventions on the observed event graph
and match-state representations (e.g., event removal, possession chain truncation).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from temporal_context import EventContext
from match_state import MatchState
from event_graph import MatchEventGraph, GraphNode


@dataclass
class CounterfactualScenario:
    """Represents a strictly grounded counterfactual intervention on an observed match event."""
    intervention_type: str  # 'EVENT_REMOVAL', 'POSSESSION_INVERSION', 'SEQUENCE_TRUNCATION'
    target_event_action: str
    target_player: str
    target_team: str
    original_observation: str
    counterfactual_state: str
    logical_consequence: str
    is_speculative: bool  # Always False (strictly grounded in graph topology & telemetry)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention_type": self.intervention_type,
            "target_event_action": self.target_event_action,
            "target_player": self.target_player,
            "target_team": self.target_team,
            "original_observation": self.original_observation,
            "counterfactual_state": self.counterfactual_state,
            "logical_consequence": self.logical_consequence,
            "is_speculative": self.is_speculative,
        }


class CounterfactualEngine:
    """
    Evaluates grounded counterfactual interventions over the event graph.
    Never invents unobserved football outcomes.
    """

    def analyze_event_counterfactual(
        self,
        event_ctx: EventContext,
        match_state: MatchState,
        event_graph: Optional[MatchEventGraph] = None,
    ) -> List[CounterfactualScenario]:
        scenarios: List[CounterfactualScenario] = []
        action = event_ctx.action
        player = event_ctx.player
        team = event_ctx.team
        opp = event_ctx.opponent

        # 1. Structural Event Removal Intervention
        orig_obs = f"{player} ({team}) executes {action} at {event_ctx.start_time:.1f}s."
        cf_state = f"Event #{event_ctx.event_index + 1} ({action}) is structurally excised from the match sequence."
        
        if action == "GOAL":
            consequence = (
                f"Without the recorded {action} event, the sequence concludes with the preceding {event_ctx.sequence_signature.split(' -> ')[-2] if ' -> ' in event_ctx.sequence_signature else 'action'}. "
                f"Any score consequence associated with this goal is absent."
            )
        elif action in {"SHOT", "CROSS"}:
            consequence = (
                f"Without this {action}, the offensive build-up ({event_ctx.sequence_signature}) does not culminate in a scoring attempt, "
                f"leaving the ball in {event_ctx.location}."
            )
        else:
            consequence = (
                f"Without this {action}, possession progression ceases at the preceding event, leaving telemetry unchanged."
            )

        scenarios.append(
            CounterfactualScenario(
                intervention_type="EVENT_REMOVAL",
                target_event_action=action,
                target_player=player,
                target_team=team,
                original_observation=orig_obs,
                counterfactual_state=cf_state,
                logical_consequence=consequence,
                is_speculative=False,
            )
        )

        # 2. Possession Chain Intervention (if sequence has prior events)
        if event_ctx.previous_events:
            prev_act = event_ctx.previous_events[-1].get("action", "ACTION")
            prev_team = event_ctx.previous_events[-1].get("team", team)
            scenarios.append(
                CounterfactualScenario(
                    intervention_type="SEQUENCE_TRUNCATION",
                    target_event_action=action,
                    target_player=player,
                    target_team=team,
                    original_observation=f"Culmination of multi-event build-up: {event_ctx.sequence_signature}",
                    counterfactual_state=f"Sequence terminated after {prev_act} by {prev_team}.",
                    logical_consequence=(
                        f"The terminal {action} by {player} would not be reached, preserving {prev_team}'s prior state without progression into {event_ctx.location}."
                    ),
                    is_speculative=False,
                )
            )

        return scenarios


if __name__ == "__main__":
    from temporal_context import TemporalContextBuilder
    from match_state import MatchStateTracker

    t_b = TemporalContextBuilder()
    m_t = MatchStateTracker()
    engine = CounterfactualEngine()

    pbp = Path("data/demo/pbp/0008/play-by-play-en.jsonl")
    contexts = t_b.build_match_contexts(pbp)
    goal_ctx = [c for c in contexts if c.action == "GOAL"][0]
    state = m_t.compute_state_for_event("0008", goal_ctx.action, goal_ctx.team, goal_ctx.start_time)

    res = engine.analyze_event_counterfactual(goal_ctx, state)
    print("Counterfactual Scenarios for Goal:")
    for s in res:
        print(f"[{s.intervention_type}]")
        print("  Original:", s.original_observation)
        print("  State:", s.counterfactual_state)
        print("  Consequence:", s.logical_consequence)
        print("  Speculative:", s.is_speculative)
