"""
context_retrieval.py

Research Contribution 4: Context-Aware Retrieval Engine.
Extends the existing FAISS + Sentence-Transformers retrieval with:
1. Semantic Similarity (FAISS IndexFlatIP cosine similarity)
2. Entity Awareness (Player filename/text, Team, Opponent)
3. Temporal Context (Sequence buildup, recent players/actions)
4. Match-State Context (Match phase, score situation, goal consequence)
5. Event Importance (Action impact weighting)

Provides transparent scoring where every candidate document exposes:
- semantic_score
- entity_score (player_bonus + team_bonus + opponent_bonus)
- temporal_score
- match_state_score
- final_score
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from improved_rag import ImprovedFootballRAG
from temporal_context import EventContext
from match_state import MatchState
from event_importance import ImportanceBreakdown


@dataclass
class ScoredDocument:
    """Document with complete explainable score breakdown across all research dimensions."""
    rank: int
    filename: str
    doc_stem: str
    background_text: str
    semantic_score: float
    entity_score: float
    player_bonus: float
    team_bonus: float
    opponent_bonus: float
    temporal_score: float
    match_state_score: float
    importance_weight: float
    final_score: float
    match_reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "filename": self.filename,
            "doc_stem": self.doc_stem,
            "semantic_score": round(self.semantic_score, 4),
            "entity_score": round(self.entity_score, 4),
            "player_bonus": round(self.player_bonus, 4),
            "team_bonus": round(self.team_bonus, 4),
            "opponent_bonus": round(self.opponent_bonus, 4),
            "temporal_score": round(self.temporal_score, 4),
            "match_state_score": round(self.match_state_score, 4),
            "importance_weight": round(self.importance_weight, 4),
            "final_score": round(self.final_score, 4),
            "match_reasons": self.match_reasons,
            "text_snippet": self.background_text[:250],
        }


class ContextAwareRetriever:
    """Multi-stage retrieval and reranking engine."""

    def __init__(
        self,
        rag_engine: ImprovedFootballRAG,
        player_name_bonus: float = 0.30,
        player_text_bonus: float = 0.10,
        team_bonus: float = 0.05,
        opponent_bonus: float = 0.03,
        temporal_bonus: float = 0.08,
        match_state_bonus: float = 0.05,
    ):
        self.rag = rag_engine
        self.player_name_bonus = player_name_bonus
        self.player_text_bonus = player_text_bonus
        self.team_bonus = team_bonus
        self.opponent_bonus = opponent_bonus
        self.temporal_bonus = temporal_bonus
        self.match_state_bonus = match_state_bonus

    def retrieve(
        self,
        event_context: EventContext,
        match_state: Optional[MatchState] = None,
        importance: Optional[ImportanceBreakdown] = None,
        mode: str = "full",  # 'semantic_only', 'entity_aware', 'temporal', 'match_state', 'full'
        top_k: int = 3,
        candidate_pool: int = 60,
    ) -> List[ScoredDocument]:
        """
        Executes retrieval under specified ablation/research mode.
        """
        player = event_context.player
        team = event_context.team
        opponent = event_context.opponent
        action = event_context.action
        curr_text = event_context.current_event.get("text", "")

        # Formulate query: Keep core entities prominent
        if mode in {"semantic_only", "entity_aware"}:
            query = f"{player} {team} {action} {curr_text}"
        else:
            seq = event_context.sequence_signature or action
            query = f"{player} {team} {action} {seq} {opponent} {curr_text}"

        raw_candidates = self.rag.retrieve_raw(query, top_k=candidate_pool)
        scored_docs: List[ScoredDocument] = []

        player_clean = player.lower().replace("_", " ").strip()
        team_clean = team.lower().strip()
        opp_clean = opponent.lower().strip()

        # Previous entities for temporal relevance
        prev_players = [p.lower().replace("_", " ").strip() for p in event_context.to_dict().get("previous_players", []) if p]

        for cand in raw_candidates:
            filename = cand["filename"]
            stem = Path(filename).stem.replace("_", " ")
            stem_lower = stem.lower()
            text_lower = cand["background"].lower()

            sem_score = float(cand["score"])
            p_bonus = 0.0
            t_bonus = 0.0
            o_bonus = 0.0
            temp_bonus = 0.0
            ms_bonus = 0.0
            reasons = []

            # 1. Entity bonuses
            if mode != "semantic_only":
                # Player
                if player_clean and player_clean in stem_lower:
                    p_bonus = self.player_name_bonus
                    reasons.append(f"Player exact match in title (+{p_bonus:.2f})")
                elif player_clean and player_clean in text_lower:
                    p_bonus = self.player_text_bonus
                    reasons.append(f"Player mentioned in article (+{p_bonus:.2f})")

                # Team
                if team_clean and team_clean in text_lower:
                    t_bonus = self.team_bonus
                    reasons.append(f"Club '{team}' cited (+{t_bonus:.2f})")

                # Opponent
                if opp_clean and opp_clean in text_lower:
                    o_bonus = self.opponent_bonus
                    reasons.append(f"Opponent '{opponent}' cited (+{o_bonus:.2f})")

            # 2. Temporal Context bonus
            if mode in {"temporal", "match_state", "full"}:
                # Check if document relates to previous actors in the build-up
                for prev_p in prev_players:
                    if prev_p and (prev_p in stem_lower or prev_p in text_lower):
                        temp_bonus += self.temporal_bonus
                        reasons.append(f"Involved in sequence build-up with {prev_p} (+{self.temporal_bonus:.2f})")
                        break

                if event_context.is_sequence_climax and action in {"GOAL", "SHOT"}:
                    if "goal" in text_lower or "score" in text_lower:
                        temp_bonus += 0.02
                        reasons.append("Climax event goalscoring context (+0.02)")

            # 3. Match State bonus
            if mode in {"match_state", "full"} and match_state:
                if match_state.is_late_game and ("derby" in text_lower or "winner" in text_lower or "final" in text_lower):
                    ms_bonus += self.match_state_bonus
                    reasons.append("Late-match high stakes context (+0.05)")
                if match_state.is_score_available and match_state.home_score is not None:
                    # Relevance to league/season context
                    if match_state.league.lower() in text_lower:
                        ms_bonus += 0.02
                        reasons.append(f"League context '{match_state.league}' (+0.02)")

            # Importance weighting
            imp_weight = 1.0
            if mode == "full" and importance:
                # Modulate final ranking score by importance
                imp_weight = 1.0 + (importance.final_importance * 0.05)

            entity_total = p_bonus + t_bonus + o_bonus
            total_score = (sem_score + entity_total + temp_bonus + ms_bonus) * imp_weight

            scored_docs.append(
                ScoredDocument(
                    rank=0,
                    filename=filename,
                    doc_stem=stem,
                    background_text=cand["background"],
                    semantic_score=sem_score,
                    entity_score=entity_total,
                    player_bonus=p_bonus,
                    team_bonus=t_bonus,
                    opponent_bonus=o_bonus,
                    temporal_score=temp_bonus,
                    match_state_score=ms_bonus,
                    importance_weight=imp_weight,
                    final_score=total_score,
                    match_reasons=reasons,
                )
            )

        # Sort descending by final score
        scored_docs.sort(key=lambda d: d.final_score, reverse=True)
        for i, doc in enumerate(scored_docs[:top_k], 1):
            doc.rank = i

        return scored_docs[:top_k]


if __name__ == "__main__":
    from temporal_context import TemporalContextBuilder
    from match_state import MatchStateTracker
    from event_importance import EventImportanceScorer

    rag = ImprovedFootballRAG()
    t_builder = TemporalContextBuilder()
    m_tracker = MatchStateTracker()
    imp_scorer = EventImportanceScorer()

    pbp = Path("data/demo/pbp/0008/play-by-play-en.jsonl")
    contexts = t_builder.build_match_contexts(pbp)
    goal_ctx = [c for c in contexts if c.action == "GOAL"][0]

    state = m_tracker.compute_state_for_event(
        goal_ctx.match_id, goal_ctx.action, goal_ctx.team, goal_ctx.start_time
    )
    imp = imp_scorer.compute_importance(
        goal_ctx.action, state.match_minute, state.score_difference or 0, goal_ctx.is_sequence_climax
    )

    retriever = ContextAwareRetriever(rag)
    modes = ["semantic_only", "entity_aware", "full"]
    print("\n--- RETRIEVAL COMPARISON ---")
    for m in modes:
        res = retriever.retrieve(goal_ctx, state, imp, mode=m, top_k=2)
        print(f"\nMode: {m}")
        for r in res:
            print(f"  #{r.rank} {r.doc_stem} | Final: {r.final_score:.4f} (Sem: {r.semantic_score:.4f}, Ent: {r.entity_score:.4f}, Temp: {r.temporal_score:.4f})")
