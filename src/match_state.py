"""
match_state.py

Research Contribution 2: Match-State Reasoning.
Tracks home/away teams, match phase, pitch minute, and full-time historical context.
Strict scientific honesty regarding scores:
- The dataset (PBP events & sample_metadata.csv) contains full-time match scores,
  but does NOT track live in-game running score progression for individual events.
- Therefore, event-level live scores are honestly flagged as unavailable
  ("Score information unavailable" / "Score consequence unavailable")
  rather than falsely extrapolating 90-minute full-time scores to mid-game timestamps.
- If live running score data is explicitly provided, goal consequences (equalizer,
  go-ahead goal, lead extender) are computed using exact before/after logic.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import re
import pandas as pd


@dataclass
class MatchState:
    """State of the match at a given event timestamp."""
    match_id: str
    league: str
    date: str
    home_team: str
    away_team: str
    half: int
    match_minute: int
    is_late_game: bool
    full_time_result: Optional[str] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    score_difference: Optional[int] = None
    leader: Optional[str] = None
    state_description: str = "Score information unavailable"
    is_tied: bool = False
    is_score_available: bool = False
    consequence_description: str = "Score consequence unavailable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "league": self.league,
            "date": self.date,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "half": self.half,
            "match_minute": self.match_minute,
            "is_late_game": self.is_late_game,
            "full_time_result": self.full_time_result,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "score_difference": self.score_difference,
            "leader": self.leader,
            "state_description": self.state_description,
            "is_tied": self.is_tied,
            "is_score_available": self.is_score_available,
            "consequence_description": self.consequence_description,
        }

    def summary_display(self) -> str:
        if not self.is_score_available:
            return f"{self.home_team} vs {self.away_team} | Score information unavailable | Half {self.half}, Min {self.match_minute}'"
        score_str = f"{self.home_team} {self.home_score} - {self.away_score} {self.away_team}"
        return f"{score_str} ({self.state_description}) | Half {self.half}, Min {self.match_minute}'"


class MatchStateTracker:
    """Extracts and maintains match state from sample_metadata.csv and event stream."""

    def __init__(self, metadata_csv: Path = Path("data/demo/sample_metadata.csv")):
        self.metadata_csv = Path(metadata_csv)
        self.metadata_lookup: Dict[str, Dict[str, Any]] = {}
        self._load_metadata()

    def _load_metadata(self):
        if not self.metadata_csv.exists():
            return
        df = pd.read_csv(self.metadata_csv)
        for _, row in df.iterrows():
            m_id = f"{int(row['id']):04d}"
            game_str = str(row["game"])
            parts = game_str.split("/")
            league = parts[0].replace("_", " ").title() if len(parts) > 0 else "Unknown League"
            date = ""
            home_team = "Home"
            away_team = "Away"
            home_score = None
            away_score = None

            if len(parts) >= 3:
                filename = parts[2]
                match = re.search(r"(\d{4}-\d{2}-\d{2})\s*-\s*\S+\s+(.*?)\s+(\d+)\s*-\s*(\d+)\s+(.*)", filename)
                if match:
                    date = match.group(1)
                    home_team = match.group(2).strip()
                    home_score = int(match.group(3))
                    away_score = int(match.group(4))
                    away_team = match.group(5).strip()

            self.metadata_lookup[m_id] = {
                "id": m_id,
                "game_str": game_str,
                "league": league,
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "final_home_score": home_score,
                "final_away_score": away_score,
                "half": int(row.get("half", 1)),
                "start_offset": float(row.get("start", 0)),
                "end_offset": float(row.get("end", 0)),
            }

    def compute_state_for_event(
        self,
        match_id: str,
        event_action: str,
        event_team: str,
        event_start_time: float,
        live_home_score: Optional[int] = None,
        live_away_score: Optional[int] = None,
    ) -> MatchState:
        meta = self.metadata_lookup.get(f"{int(match_id):04d}" if match_id.isdigit() else match_id)
        
        # Calculate pitch minute
        if meta:
            half = meta["half"]
            minute_base = 0 if half == 1 else 45
            match_minute = int(minute_base + (meta["start_offset"] + event_start_time) // 60)
            league = meta["league"]
            date = meta["date"]
            home_team = meta["home_team"]
            away_team = meta["away_team"]
            ft_result = f"{home_team} {meta['final_home_score']} - {meta['final_away_score']} {away_team}" if meta["final_home_score"] is not None else None
        else:
            half = 1
            match_minute = int(event_start_time // 60)
            league = "Football"
            date = "2015/2016"
            home_team = event_team
            away_team = "Opponent"
            ft_result = None

        is_late = match_minute >= 75

        # Check if live running score was provided
        if live_home_score is None or live_away_score is None:
            # Live in-game running score is not present in PBP or sample metadata
            return MatchState(
                match_id=match_id,
                league=league,
                date=date,
                home_team=home_team,
                away_team=away_team,
                half=half,
                match_minute=match_minute,
                is_late_game=is_late,
                full_time_result=ft_result,
                home_score=None,
                away_score=None,
                score_difference=None,
                leader=None,
                state_description="Score information unavailable",
                is_tied=False,
                is_score_available=False,
                consequence_description="Score consequence unavailable",
            )

        # If live score is known, perform match-state reasoning
        diff = live_home_score - live_away_score
        is_tied = (diff == 0)
        leader = None
        if diff > 0:
            leader = home_team
            state_desc = f"{home_team} leads by {diff}"
        elif diff < 0:
            leader = away_team
            state_desc = f"{away_team} leads by {abs(diff)}"
        else:
            state_desc = "Match is level"

        consequence = ""
        if event_action.upper() == "GOAL":
            if is_tied:
                consequence = f"Critical equalizer for {event_team}!"
            elif leader == event_team and abs(diff) == 1:
                consequence = f"Go-ahead goal for {event_team}!"
            elif leader == event_team:
                consequence = f"Lead extended to {abs(diff)} goals for {event_team}."
            else:
                consequence = f"{event_team} pulls a goal back!"
        else:
            consequence = "Neutral play progression"

        return MatchState(
            match_id=match_id,
            league=league,
            date=date,
            home_team=home_team,
            away_team=away_team,
            half=half,
            match_minute=match_minute,
            is_late_game=is_late,
            full_time_result=ft_result,
            home_score=live_home_score,
            away_score=live_away_score,
            score_difference=diff,
            leader=leader,
            state_description=state_desc,
            is_tied=is_tied,
            is_score_available=True,
            consequence_description=consequence,
        )


if __name__ == "__main__":
    tracker = MatchStateTracker()
    state = tracker.compute_state_for_event(
        match_id="0008",
        event_action="GOAL",
        event_team="Bayer Leverkusen",
        event_start_time=14.8,
    )
    print("Match State:", state.summary_display())
    print("Consequence:", state.consequence_description)
    print("Full Dict:", state.to_dict())
