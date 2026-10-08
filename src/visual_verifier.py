"""
visual_verifier.py

V2 Multimodal Component: Visual Verification & Frame Grounding.
Connects commentary entities to real video telemetry from:
  data/from_video/players_in_frames_sn_gamestate.csv

BOUNDING BOX SCHEMA (SoccerNet / TrackLab convention):
  The CSV columns are named x1_720p / y1_720p / x2_720p / y2_720p, but the
  actual semantics follow the [x, y, w, h] convention:
    x1_720p  -> top-left x coordinate (pixels, 1280-wide 720p frame)
    y1_720p  -> top-left y coordinate (pixels, 720-high 720p frame)
    x2_720p  -> bounding-box WIDTH in pixels  (NOT the right x coordinate)
    y2_720p  -> bounding-box HEIGHT in pixels (NOT the bottom y coordinate)

  Evidence: 101 / 112 rows satisfy x2_720p < x1_720p when treated as coords,
  but x2_720p values (11–443) are consistent with player-box widths at 720p.
  Conversion applied here: stored_x2 = x1 + w, stored_y2 = y1 + h.

VisualPlayerDetection.bbox exposes the corrected [x1, y1, x2, y2] corners
and always satisfies: x1 < x2 and y1 < y2.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import pandas as pd


@dataclass
class VisualPlayerDetection:
    """Bounding box and identification for a player visible in video frame."""
    name: str
    short_name: str
    team: str
    jersey_number: int
    country: str
    bbox: Dict[str, int]  # {'x1': top-left-x, 'y1': top-left-y, 'x2': bottom-right-x, 'y2': bottom-right-y}
    bbox_wh: Dict[str, int]  # raw source values: {'x': ..., 'y': ..., 'w': ..., 'h': ...}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "short_name": self.short_name,
            "team": self.team,
            "jersey_number": self.jersey_number,
            "country": self.country,
            "bbox": self.bbox,
            "bbox_wh_raw": self.bbox_wh,
        }


@dataclass
class VisualVerificationResult:
    """Visual grounding outcome for commentary."""
    is_visually_grounded: bool
    matched_player: Optional[VisualPlayerDetection]
    all_players_in_frame: List[VisualPlayerDetection]
    frame_timestamp_sec: float
    visual_confidence: float  # 1.0 = exact player in frame, 0.5 = team players in frame, 0.0 = no tracking
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_visually_grounded": self.is_visually_grounded,
            "matched_player": self.matched_player.to_dict() if self.matched_player else None,
            "all_players_in_frame": [p.to_dict() for p in self.all_players_in_frame],
            "frame_timestamp_sec": self.frame_timestamp_sec,
            "visual_confidence": round(self.visual_confidence, 3),
            "explanation": self.explanation,
        }


class VisualFrameVerifier:
    """Queries TrackLab/SoccerNet tracking CSV for player bounding boxes."""

    def __init__(self, tracking_csv: Path = Path("data/from_video/players_in_frames_sn_gamestate.csv")):
        self.tracking_csv = Path(tracking_csv)
        self.df: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self):
        if self.tracking_csv.exists():
            self.df = pd.read_csv(self.tracking_csv)
        else:
            self.df = pd.DataFrame()

    def verify_player_in_frame(
        self,
        match_id: str,
        half: int,
        target_player: str,
        target_team: str,
        event_time_sec: float,
        time_tolerance_sec: float = 60.0,
    ) -> VisualVerificationResult:
        if self.df is None or self.df.empty:
            return VisualVerificationResult(
                is_visually_grounded=False,
                matched_player=None,
                all_players_in_frame=[],
                frame_timestamp_sec=event_time_sec,
                visual_confidence=0.0,
                explanation="Visual tracking dataset not available.",
            )

        m_num = f"{int(match_id):04d}" if match_id.isdigit() else match_id

        # Filter by half
        sub = self.df[self.df["half"] == half]

        # Filter by match substring (e.g. "2015-08-29" or team names)
        # Check if match_id maps to game string via sample_metadata
        meta_csv = Path("data/demo/sample_metadata.csv")
        game_filter = None
        if meta_csv.exists():
            m_df = pd.read_csv(meta_csv)
            m_row = m_df[m_df["id"] == int(match_id)] if match_id.isdigit() else m_df[m_df["id"].astype(str) == match_id]
            if not m_row.empty:
                game_filter = m_row.iloc[0]["game"]

        if game_filter:
            sub = sub[sub["game"] == game_filter]
        else:
            return VisualVerificationResult(
                is_visually_grounded=False,
                matched_player=None,
                all_players_in_frame=[],
                frame_timestamp_sec=event_time_sec,
                visual_confidence=0.0,
                explanation=f"Match {match_id} could not be resolved to a tracking game identifier in metadata.",
            )

        if sub.empty:
            return VisualVerificationResult(
                is_visually_grounded=False,
                matched_player=None,
                all_players_in_frame=[],
                frame_timestamp_sec=event_time_sec,
                visual_confidence=0.0,
                explanation=f"No tracking frames recorded for match {match_id} in half {half}.",
            )

        # Find closest recorded frame time
        unique_times = sub["time"].unique()
        # In tracking CSV, time is in match seconds (e.g. 871)
        # Compare with start_offset + event_time_sec
        meta_offset = 0.0
        if game_filter and not m_row.empty:
            meta_offset = float(m_row.iloc[0].get("start", 0))

        actual_match_time = meta_offset + event_time_sec
        closest_time = min(unique_times, key=lambda t: abs(t - actual_match_time))

        frame_rows = sub[sub["time"] == closest_time]
        detections: List[VisualPlayerDetection] = []
        # Frame dimensions for 720p video
        FRAME_W, FRAME_H = 1280, 720
        for _, r in frame_rows.iterrows():
            # CSV stores [x, y, w, h] where x2_720p=width and y2_720p=height
            x1 = int(r["x1_720p"])
            y1 = int(r["y1_720p"])
            w  = int(r["x2_720p"])  # width (column is named x2 but stores width)
            h  = int(r["y2_720p"])  # height (column is named y2 but stores height)
            # Convert to [x1, y1, x2, y2] corner coordinates
            x2 = x1 + w
            y2 = y1 + h
            # Clamp to frame bounds
            x1 = max(0, min(x1, FRAME_W))
            y1 = max(0, min(y1, FRAME_H))
            x2 = max(0, min(x2, FRAME_W))
            y2 = max(0, min(y2, FRAME_H))
            # Validate invariants
            if x2 <= x1 or y2 <= y1:
                # Degenerate box — skip this detection rather than store invalid data
                continue
            detections.append(
                VisualPlayerDetection(
                    name=str(r["name"]),
                    short_name=str(r["short_name"]),
                    team=str(r["team"]),
                    jersey_number=int(r["jersey_number"]),
                    country=str(r["country"]),
                    bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    bbox_wh={"x": int(r["x1_720p"]), "y": int(r["y1_720p"]),
                             "w": w, "h": h},
                )
            )

        # Match player name
        p_clean = re.sub(r"[^\w\s]", "", target_player).lower().strip()
        matched = None
        for det in detections:
            d_short = det.short_name.lower().strip()
            d_name = det.name.lower().strip()
            if p_clean in d_short or d_short in p_clean or p_clean in d_name:
                matched = det
                break

        if matched:
            return VisualVerificationResult(
                is_visually_grounded=True,
                matched_player=matched,
                all_players_in_frame=detections,
                frame_timestamp_sec=float(closest_time),
                visual_confidence=1.0,
                explanation=f"Visually confirmed in video frame at {closest_time}s: {matched.name} (#{matched.jersey_number}) with bounding box {matched.bbox}.",
            )

        # If target player not detected, check if teammates are visible
        teammates = [d for d in detections if target_team.lower() in d.team.lower()]
        if teammates:
            return VisualVerificationResult(
                is_visually_grounded=False,
                matched_player=None,
                all_players_in_frame=detections,
                frame_timestamp_sec=float(closest_time),
                visual_confidence=0.5,
                explanation=f"Player '{target_player}' not in active camera view at {closest_time}s. {len(teammates)} teammates visible in frame ({', '.join([t.short_name for t in teammates])}).",
            )

        return VisualVerificationResult(
            is_visually_grounded=False,
            matched_player=None,
            all_players_in_frame=detections,
            frame_timestamp_sec=float(closest_time),
            visual_confidence=0.0,
            explanation=f"Neither player '{target_player}' nor club '{target_team}' confirmed in active frame at {closest_time}s.",
        )


if __name__ == "__main__":
    verifier = VisualFrameVerifier()
    res = verifier.verify_player_in_frame(
        match_id="0008",
        half=2,
        target_player="Kresic",
        target_team="Bayer Leverkusen",
        event_time_sec=14.8,
    )
    print("Visual Grounding Test (Match 0008, Kresic):")
    print("  Grounded:", res.is_visually_grounded)
    print("  Confidence:", res.visual_confidence)
    print("  Explanation:", res.explanation)
    if res.matched_player:
        print("  BBox:", res.matched_player.bbox)
    print(f"  Total players in frame: {len(res.all_players_in_frame)}")
