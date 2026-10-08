"""
event_importance.py

Research Contribution 3: Explainable Event Importance.
Computes a transparent, heuristic importance score (0.0 to 1.0) for every
football event based on action criticality, match minute, score differential,
and sequence position. Weights are fully configurable.
"""

from dataclasses import dataclass
from typing import Dict, Any, List


# Base weights by action type (heuristic, explainable)
DEFAULT_ACTION_WEIGHTS: Dict[str, float] = {
    "GOAL": 1.00,
    "PENALTY": 0.95,
    "RED CARD": 0.90,
    "SHOT": 0.70,
    "HEADER": 0.65,
    "SAVE": 0.65,
    "YELLOW CARD": 0.50,
    "CROSS": 0.45,
    "FREE KICK": 0.40,
    "DRIVE": 0.35,
    "BALL PLAYER BLOCK": 0.35,
    "HIGH PASS": 0.30,
    "PASS": 0.25,
    "THROW IN": 0.20,
    "OUT": 0.15,
}


@dataclass
class ImportanceBreakdown:
    """Detailed explainable breakdown of how the importance score was calculated."""
    event_action: str
    base_action_score: float
    time_multiplier: float
    margin_multiplier: float
    sequence_bonus: float
    final_importance: float
    importance_tier: str  # Critical, High, Moderate, Routine
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_action": self.event_action,
            "base_action_score": round(self.base_action_score, 3),
            "time_multiplier": round(self.time_multiplier, 3),
            "margin_multiplier": round(self.margin_multiplier, 3),
            "sequence_bonus": round(self.sequence_bonus, 3),
            "final_importance": round(self.final_importance, 3),
            "importance_tier": self.importance_tier,
            "explanation": self.explanation,
        }


class EventImportanceScorer:
    """Calculates transparent, configurable event importance scores."""

    def __init__(
        self,
        action_weights: Dict[str, float] = None,
        late_game_boost: float = 0.15,
        close_game_boost: float = 0.10,
        sequence_climax_boost: float = 0.10,
    ):
        self.action_weights = action_weights or DEFAULT_ACTION_WEIGHTS
        self.late_game_boost = late_game_boost
        self.close_game_boost = close_game_boost
        self.sequence_climax_boost = sequence_climax_boost

    def compute_importance(
        self,
        action: str,
        match_minute: int = 45,
        score_diff: int = 0,
        is_sequence_climax: bool = False,
    ) -> ImportanceBreakdown:
        act_upper = action.strip().upper()
        base = self.action_weights.get(act_upper, 0.25)

        reasons = [f"Base score for {act_upper}: {base:.2f}"]

        # 1. Match Timing Multiplier (events in late game min 75+ have higher tension)
        time_mult = 1.0
        if match_minute >= 75:
            time_mult += self.late_game_boost
            reasons.append(f"Late-game tension boost (min {match_minute}'): +{self.late_game_boost:.2f}")
        elif match_minute <= 10:
            time_mult += 0.05
            reasons.append(f"Early-game setting boost (min {match_minute}'): +0.05")

        # 2. Score Margin Multiplier (close games <= 1 goal differential carry more drama)
        margin_mult = 1.0
        if abs(score_diff) <= 1:
            margin_mult += self.close_game_boost
            reasons.append(f"Close-match context (|diff|={abs(score_diff)}): +{self.close_game_boost:.2f}")

        # 3. Sequence Climax Bonus (culmination of build-up)
        seq_bonus = 0.0
        if is_sequence_climax:
            seq_bonus += self.sequence_climax_boost
            reasons.append(f"Sequence climax bonus: +{self.sequence_climax_boost:.2f}")

        # Combine
        raw_score = (base * time_mult * margin_mult) + seq_bonus
        final_score = min(1.0, max(0.0, raw_score))

        # Assign tier
        if final_score >= 0.85:
            tier = "Critical"
        elif final_score >= 0.60:
            tier = "High"
        elif final_score >= 0.35:
            tier = "Moderate"
        else:
            tier = "Routine"

        explanation = "; ".join(reasons)

        return ImportanceBreakdown(
            event_action=act_upper,
            base_action_score=base,
            time_multiplier=time_mult,
            margin_multiplier=margin_mult,
            sequence_bonus=seq_bonus,
            final_importance=final_score,
            importance_tier=tier,
            explanation=explanation,
        )


if __name__ == "__main__":
    scorer = EventImportanceScorer()
    # Test GOAL in 88th minute of a 1-goal game
    res = scorer.compute_importance(
        action="GOAL",
        match_minute=88,
        score_diff=1,
        is_sequence_climax=True,
    )
    print("Importance breakdown:")
    print(res.to_dict())
