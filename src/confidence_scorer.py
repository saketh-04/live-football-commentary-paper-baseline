"""
confidence_scorer.py

V2 Multimodal Component: Explainable Multimodal Confidence Scoring.

IMPORTANT SCIENTIFIC NOTICE:
This confidence score is an EXPLAINABLE HEURISTIC combining multiple system signals,
NOT a statistically calibrated Bayesian posterior probability.
Weights are hand-picked, transparent, and configurable.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ConfidenceBreakdown:
    """Detailed explainable breakdown of the multimodal confidence score."""
    composite_confidence: float
    retrieval_component: float
    entity_component: float
    factual_component: float
    visual_component: float
    confidence_tier: str  # 'High Confidence', 'Moderate Confidence', 'Tentative / Low'
    is_statistically_calibrated: bool  # Always False (honest scientific disclosure)
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite_confidence": round(self.composite_confidence, 3),
            "retrieval_component": round(self.retrieval_component, 3),
            "entity_component": round(self.entity_component, 3),
            "factual_component": round(self.factual_component, 3),
            "visual_component": round(self.visual_component, 3),
            "confidence_tier": self.confidence_tier,
            "is_statistically_calibrated": self.is_statistically_calibrated,
            "explanation": self.explanation,
        }


class MultimodalConfidenceScorer:
    """
    Computes an explainable, transparent composite score from 0.0 to 1.0.
    Combines:
    - Semantic retrieval cosine similarity (normalized)
    - Entity awareness bonus
    - Factual verification status
    - Visual frame grounding (TrackLab bounding box confirmation)
    """

    def __init__(
        self,
        w_retrieval: float = 0.35,
        w_entity: float = 0.25,
        w_factual: float = 0.20,
        w_visual: float = 0.20,
    ):
        self.w_retrieval = w_retrieval
        self.w_entity = w_entity
        self.w_factual = w_factual
        self.w_visual = w_visual

    def compute_confidence(
        self,
        semantic_score: float,
        entity_score: float,
        verification_status: str,
        was_corrected: bool,
        visual_confidence: float,
    ) -> ConfidenceBreakdown:
        # 1. Retrieval signal (semantic cosine similarity typically in 0.3 - 0.7 range, normalized)
        norm_sem = min(1.0, max(0.0, (semantic_score - 0.2) / 0.6))

        # 2. Entity signal (max entity bonus is ~0.48: player 0.30 + player_text 0.10 + team 0.05 + opp 0.03)
        norm_ent = min(1.0, max(0.0, entity_score / 0.45))

        # 3. Factual verification signal
        if verification_status == "VERIFIED" and not was_corrected:
            fact_score = 1.0
        elif was_corrected:
            fact_score = 0.70  # Auto-corrected to factual alignment
        else:
            fact_score = 0.20  # Unresolved contradiction

        # 4. Visual grounding signal (from visual_verifier: 1.0 exact bbox, 0.5 teammates, 0.0 none)
        vis_score = min(1.0, max(0.0, visual_confidence))

        # Composite weighted sum
        raw_composite = (
            (self.w_retrieval * norm_sem)
            + (self.w_entity * norm_ent)
            + (self.w_factual * fact_score)
            + (self.w_visual * vis_score)
        )
        final_conf = min(1.0, max(0.0, raw_composite))

        if final_conf >= 0.75:
            tier = "High Confidence"
        elif final_conf >= 0.50:
            tier = "Moderate Confidence"
        else:
            tier = "Tentative / Low"

        explanation = (
            f"Heuristic breakdown: Retrieval ({norm_sem:.2f} * {self.w_retrieval:.2f}) + "
            f"Entity ({norm_ent:.2f} * {self.w_entity:.2f}) + "
            f"Factual ({fact_score:.2f} * {self.w_factual:.2f}) + "
            f"Visual ({vis_score:.2f} * {self.w_visual:.2f}) = {final_conf:.3f}"
        )

        return ConfidenceBreakdown(
            composite_confidence=final_conf,
            retrieval_component=norm_sem,
            entity_component=norm_ent,
            factual_component=fact_score,
            visual_component=vis_score,
            confidence_tier=tier,
            is_statistically_calibrated=False,
            explanation=explanation,
        )


if __name__ == "__main__":
    scorer = MultimodalConfidenceScorer()
    res = scorer.compute_confidence(
        semantic_score=0.48,
        entity_score=0.35,
        verification_status="VERIFIED",
        was_corrected=False,
        visual_confidence=1.0,
    )
    print("Confidence Scorer Test:")
    print("  Composite Confidence:", res.composite_confidence)
    print("  Tier:", res.confidence_tier)
    print("  Explanation:", res.explanation)
    print("  Dict:", res.to_dict())
