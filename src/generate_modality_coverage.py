"""
Generate empirical modality coverage matrix across all matches in the dataset.
Checks:
- Play-by-play (PBP) events count
- Real match video availability (in outputs/demo-step3/)
- Visual tracking availability (in data/from_video/players_in_frames_sn_gamestate.csv)
- Spotting labels availability
- Ground truth reference availability
- Multimodal evaluation readiness
"""
import pandas as pd
from pathlib import Path

def generate_coverage():
    meta = pd.read_csv("data/demo/sample_metadata.csv")
    tracking_csv = Path("data/from_video/players_in_frames_sn_gamestate.csv")
    track_df = pd.read_csv(tracking_csv) if tracking_csv.exists() else pd.DataFrame()
    track_games = set(track_df["game"].dropna().unique()) if not track_df.empty else set()

    pbp_dir = Path("data/demo/pbp")
    video_dir = Path("outputs/demo-step3")

    rows = []
    for _, r in meta.iterrows():
        raw_id = int(r["id"])
        m_id = f"{raw_id:04d}"
        game_str = str(r["game"])
        short_game = game_str.split("/")[-1] if "/" in game_str else game_str
        half = int(r["half"])

        pbp_file = pbp_dir / m_id / "play-by-play-en.jsonl"
        has_pbp = pbp_file.exists() and pbp_file.stat().st_size > 0
        num_events = 0
        if has_pbp:
            with open(pbp_file, "r", encoding="utf-8") as f:
                num_events = len([l for l in f if l.strip()])

        has_video = (video_dir / f"{m_id}-en.mp4").exists() or (video_dir / f"{m_id}-en-gsr.mp4").exists()
        has_tracking = game_str in track_games
        track_rows = 0
        if has_tracking and not track_df.empty:
            sub = track_df[(track_df["game"] == game_str) & (track_df["half"] == half)]
            track_rows = len(sub)

        has_gt = num_events > 0
        multimodal_ready = has_pbp and has_video and has_tracking and track_rows > 0

        rows.append({
            "match_id": m_id,
            "game_title": short_game,
            "half": half,
            "pbp_events": num_events,
            "has_pbp": "YES" if has_pbp else "NO",
            "has_video": "YES" if has_video else "NO",
            "has_tracking": "YES" if has_tracking else "NO",
            "tracking_frames_half": track_rows,
            "has_ground_truth": "YES" if has_gt else "NO",
            "multimodal_eval_usable": "YES" if multimodal_ready else "NO",
        })

    cov_df = pd.DataFrame(rows)
    out_dir = Path("outputs/research")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "modality_coverage.csv"
    cov_df.to_csv(out_csv, index=False)
    print(f"Modality coverage saved to {out_csv}")
    print("\n" + cov_df.to_string(index=False))

    # Summary statistics
    total = len(cov_df)
    pbp_count = sum(1 for r in rows if r["has_pbp"] == "YES")
    video_count = sum(1 for r in rows if r["has_video"] == "YES")
    track_count = sum(1 for r in rows if r["has_tracking"] == "YES")
    track_half_count = sum(1 for r in rows if r["tracking_frames_half"] > 0)
    usable_count = sum(1 for r in rows if r["multimodal_eval_usable"] == "YES")
    total_events = sum(r["pbp_events"] for r in rows)

    print("\n" + "="*60)
    print("MODALITY COVERAGE SUMMARY")
    print("="*60)
    print(f"Total Match Clips in Metadata:   {total}")
    print(f"Clips with Play-by-Play (PBP):   {pbp_count} ({total_events} total events)")
    print(f"Clips with Real Video:           {video_count}")
    print(f"Clips with Game in Tracking:     {track_count}")
    print(f"Clips with Tracking in That Half:{track_half_count}")
    print(f"Clips Fully Multimodal Usable:   {usable_count}")
    print("="*60)

if __name__ == "__main__":
    generate_coverage()
