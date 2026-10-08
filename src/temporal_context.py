"""
temporal_context.py

Research Contribution 1: Temporal Match Context.
Football events are sequential, not isolated occurrences. This module builds
a structured representation of an event within its immediate temporal and
match context (previous events, match minute, elapsed time, and sequence flow).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import json


@dataclass
class EventContext:
    """Structured representation of an event in its temporal sequence."""
    match_id: str
    match_file: str
    event_index: int
    total_events: int
    current_event: Dict[str, Any]
    previous_events: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    minute: int = 0
    second: int = 0
    action: str = ""
    player: str = ""
    team: str = ""
    opponent: str = ""
    location: str = ""
    sequence_signature: str = ""
    is_sequence_climax: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "match_file": self.match_file,
            "event_index": self.event_index,
            "total_events": self.total_events,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": round(self.duration, 2),
            "minute": self.minute,
            "second": self.second,
            "action": self.action,
            "player": self.player,
            "team": self.team,
            "opponent": self.opponent,
            "location": self.location,
            "sequence_signature": self.sequence_signature,
            "is_sequence_climax": self.is_sequence_climax,
            "previous_actions": [e.get("action", "") for e in self.previous_events],
            "previous_players": [e.get("player") or e.get("name", "") for e in self.previous_events],
            "current_text": self.current_event.get("text", ""),
        }

    def summary_text(self) -> str:
        """Human-readable description of temporal context."""
        prev_str = " -> ".join([e.get("action", "") for e in self.previous_events]) if self.previous_events else "None (Opening sequence)"
        return (
            f"Event #{self.event_index + 1}/{self.total_events} [{self.start_time:.1f}s - {self.end_time:.1f}s] "
            f"| Prior Sequence: {prev_str} -> [{self.action}]"
        )


class TemporalContextBuilder:
    """Constructs temporal context from raw play-by-play jsonl data."""

    def __init__(self, history_window: int = 3):
        self.history_window = history_window

    def build_match_contexts(self, pbp_file: Path, match_id: Optional[str] = None) -> List[EventContext]:
        pbp_file = Path(pbp_file)
        if not pbp_file.exists():
            raise FileNotFoundError(f"PBP file not found: {pbp_file}")

        if match_id is None:
            match_id = pbp_file.parent.name

        raw_events = []
        with open(pbp_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_events.append(json.loads(line))

        if not raw_events:
            return []

        # Inferred teams
        all_teams = {e.get("team") for e in raw_events if e.get("team")}
        contexts: List[EventContext] = []

        for idx, ev in enumerate(raw_events):
            team = ev.get("team", "Unknown")
            other_teams = [t for t in all_teams if t != team]
            opponent = other_teams[0] if other_teams else "Unknown"

            # Start and end timestamps
            try:
                start_t = float(ev.get("start_time", 0.0))
            except (ValueError, TypeError):
                start_t = 0.0

            try:
                end_t = float(ev.get("end_time", start_t + 1.0))
            except (ValueError, TypeError):
                end_t = start_t + 1.0

            duration = max(0.0, end_t - start_t)
            minute = int(start_t // 60)
            second = int(start_t % 60)

            # Previous N events
            start_window = max(0, idx - self.history_window)
            prev_events = raw_events[start_window:idx]

            # Sequence signature (e.g. PASS -> CROSS -> GOAL)
            seq_actions = [p.get("action", "ACTION") for p in prev_events] + [ev.get("action", "ACTION")]
            sequence_signature = " -> ".join(seq_actions)

            # High climax check: GOAL or SHOT after a build-up
            action = ev.get("action", "").upper()
            is_climax = action in {"GOAL", "SHOT"} and len(prev_events) >= 1

            player = ev.get("player") or ev.get("name") or "Unknown"

            ctx = EventContext(
                match_id=match_id,
                match_file=str(pbp_file),
                event_index=idx,
                total_events=len(raw_events),
                current_event=ev,
                previous_events=prev_events,
                start_time=start_t,
                end_time=end_t,
                duration=duration,
                minute=minute,
                second=second,
                action=action,
                player=player,
                team=team,
                opponent=opponent,
                location=ev.get("location", "Field"),
                sequence_signature=sequence_signature,
                is_sequence_climax=is_climax,
            )
            contexts.append(ctx)

        return contexts


if __name__ == "__main__":
    builder = TemporalContextBuilder(history_window=3)
    sample_file = Path("data/demo/pbp/0008/play-by-play-en.jsonl")
    contexts = builder.build_match_contexts(sample_file)
    print(f"Loaded {len(contexts)} temporal contexts from {sample_file}:")
    for c in contexts:
        print(c.summary_text())
        print(f"  Climax event: {c.is_sequence_climax} | Sequence: {c.sequence_signature}")
