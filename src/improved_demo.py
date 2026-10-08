"""
improved_demo.py

Local, API-free demonstration of an entity-aware retrieval improvement
over the original OpenAI-embedding based pipeline in this repository.

Original paper pipeline:
    event/action spotting -> additional information retrieval
    -> OpenAI embeddings -> FAISS -> OpenAI LLM -> generated commentary

This script (our improvement), for the retrieval + generation stages only:
    real play-by-play event -> entity extraction (player/team/opponent)
    -> Sentence-Transformers (all-MiniLM-L6-v2) embeddings -> FAISS
    -> entity-aware reranking -> local/deterministic commentary generation

Nothing in src/soccer_bg_commentary/ is modified. This is a standalone,
additional script that reuses ImprovedFootballRAG from improved_rag.py
for the embedding/FAISS parts only.
"""

import json
import re
from pathlib import Path

from improved_rag import ImprovedFootballRAG

PBP_FILE = Path("data/demo/pbp/0008/play-by-play-en.jsonl")
OUTPUT_DIR = Path("outputs/improved-demo")

PLAYER_NAME_BONUS = 0.30
PLAYER_TEXT_BONUS = 0.10
TEAM_BONUS = 0.05
OPPONENT_BONUS = 0.03


def _read_raw_events(pbp_file: Path) -> list[dict]:
    events = []
    with open(pbp_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _build_event_dict(event: dict, teams: set, pbp_file: Path) -> dict:
    other_teams = [t for t in teams if t != event["team"]]
    opponent = other_teams[0] if other_teams else "Unknown"
    return {
        "match_file": str(pbp_file),
        "start_time": event["start_time"],
        "end_time": event["end_time"],
        "text": event["text"],
        "action": event["action"],
        "location": event["location"],
        "player": event["name"],
        "team": event["team"],
        "opponent": opponent,
        "all_teams_in_match": sorted(teams),
    }


def load_all_events(pbp_file: Path) -> list[dict]:
    raw_events = _read_raw_events(pbp_file)
    teams = {e["team"] for e in raw_events}
    return [_build_event_dict(e, teams, pbp_file) for e in raw_events]


def load_real_event(pbp_file: Path) -> dict:
    raw_events = _read_raw_events(pbp_file)
    teams = {e["team"] for e in raw_events}
    goal_events = [e for e in raw_events if e["action"] == "GOAL"]
    if not goal_events:
        raise RuntimeError(f"No GOAL event found in {pbp_file}")
    return _build_event_dict(goal_events[0], teams, pbp_file)


def baseline_retrieve(rag: ImprovedFootballRAG, event: dict, top_k: int = 3) -> list[dict]:
    query = f"{event['player']} {event['team']} {event['action']} {event['text']}"
    results = rag.retrieve_raw(query, top_k=top_k)
    for r in results:
        r["semantic_score"] = r["score"]
        r["entity_bonus"] = 0.0
        r["final_score"] = r["score"]
    return results


def entity_aware_retrieve(rag: ImprovedFootballRAG, event: dict, top_k: int = 3, pool: int = 30) -> list[dict]:
    query = f"{event['player']} {event['team']} {event['action']} {event['text']}"
    candidates = rag.retrieve_raw(query, top_k=pool)

    player_name = event["player"]
    player_tokens = [t for t in re.split(r"\s+", player_name) if len(t) > 2]
    team_name = event["team"]
    opponent_name = event["opponent"]

    for r in candidates:
        semantic_score = r["score"]
        bonus = 0.0
        filename_lower = r["filename"].lower()
        text_lower = r["background"].lower()

        player_in_filename = any(tok.lower() in filename_lower for tok in player_tokens)
        player_in_text = any(tok.lower() in text_lower for tok in player_tokens)

        if player_in_filename:
            bonus += PLAYER_NAME_BONUS
        elif player_in_text:
            bonus += PLAYER_TEXT_BONUS

        if team_name.lower() in text_lower:
            bonus += TEAM_BONUS

        if opponent_name.lower() in text_lower:
            bonus += OPPONENT_BONUS

        r["semantic_score"] = semantic_score
        r["entity_bonus"] = round(bonus, 4)
        r["final_score"] = round(semantic_score + bonus, 4)

    candidates.sort(key=lambda r: r["final_score"], reverse=True)
    return candidates[:top_k]


def generate_local_commentary(event: dict, top_context: dict) -> str:
    player = event["player"]
    team = event["team"]
    opponent = event["opponent"]
    action = event["action"]
    context_subject = Path(top_context["filename"]).stem.replace("_", " ") if top_context else None

    if action == "GOAL":
        line = f"GOAL! {player} scores for {team} against {opponent}!"
    elif action == "SHOT":
        line = f"{player} lets fly with a shot for {team} against {opponent}."
    elif action == "CROSS":
        line = f"{player} whips in a cross for {team} against {opponent}."
    else:
        line = f"{player} of {team} is involved in a {action.lower()} against {opponent}."

    if context_subject and context_subject.lower() in player.lower():
        line += f" Background on {context_subject} is available from the retrieved player profile."

    return line


def build_pipeline_diagram() -> str:
    return (
        "MATCH\n"
        "  |\n"
        "  v\n"
        "EVENT (real play-by-play record)\n"
        "  |\n"
        "  v\n"
        "CONTEXT (player / team / opponent entities)\n"
        "  |\n"
        "  v\n"
        "RETRIEVAL (Sentence-Transformers + FAISS, entity-aware rerank)\n"
        "  |\n"
        "  v\n"
        "COMMENTARY (local/deterministic template generation)\n"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    event = load_real_event(PBP_FILE)
    rag = ImprovedFootballRAG()

    baseline_results = baseline_retrieve(rag, event, top_k=3)
    improved_results = entity_aware_retrieve(rag, event, top_k=3)

    top_context = improved_results[0] if improved_results else None
    commentary = generate_local_commentary(event, top_context)

    lines = []
    lines.append("=" * 60)
    lines.append("SOCCER COMMENTARY DEMO (local, API-free improved prototype)")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"MATCH FILE:\n{event['match_file']}")
    lines.append(f"TEAMS IN MATCH: {', '.join(event['all_teams_in_match'])}")
    lines.append("")
    lines.append("EVENT:")
    lines.append(f"  [{event['start_time']}-{event['end_time']}] {event['text']}")
    lines.append(f"  action={event['action']}  location={event['location']}")
    lines.append("")
    lines.append("ENTITIES:")
    lines.append(f"  Player:   {event['player']}")
    lines.append(f"  Team:     {event['team']}")
    lines.append(f"  Opponent: {event['opponent']}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("BASELINE RETRIEVAL (semantic similarity only)")
    lines.append("-" * 60)
    for i, r in enumerate(baseline_results, 1):
        lines.append(f"{i}. {r['filename']}")
        lines.append(f"   semantic_score={r['semantic_score']:.4f}  entity_bonus={r['entity_bonus']:.4f}  final_score={r['final_score']:.4f}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("IMPROVED ENTITY-AWARE RETRIEVAL")
    lines.append("-" * 60)
    for i, r in enumerate(improved_results, 1):
        lines.append(f"{i}. {r['filename']}")
        lines.append(f"   semantic_score={r['semantic_score']:.4f}  entity_bonus={r['entity_bonus']:.4f}  final_score={r['final_score']:.4f}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("SELECTED CONTEXT (top improved result)")
    lines.append("-" * 60)
    if top_context:
        lines.append(f"Source: {top_context['filename']}")
        lines.append(top_context["background"][:400].strip())
    lines.append("")
    lines.append("-" * 60)
    lines.append("GENERATED COMMENTARY (local/deterministic, not an LLM)")
    lines.append("-" * 60)
    lines.append(commentary)
    lines.append("")
    lines.append("-" * 60)
    lines.append("PIPELINE")
    lines.append("-" * 60)
    lines.append(build_pipeline_diagram())
    lines.append("=" * 60)
    lines.append("ORIGINAL PAPER REPRODUCTION STATUS")
    lines.append("=" * 60)
    lines.append(
        "Original repository successfully installed and original pipeline was\n"
        "executed through the OpenAI embedding stage. Full original generation\n"
        "could not be completed because the configured OpenAI API account has\n"
        "insufficient quota (HTTP 429 / 'no credits remaining')."
    )

    report_text = "\n".join(lines)
    print(report_text)

    (OUTPUT_DIR / "demo_report.txt").write_text(report_text, encoding="utf-8")

    json_payload = {
        "event": event,
        "baseline_retrieval": [
            {k: v for k, v in r.items() if k != "background"} | {"background_preview": r["background"][:300]}
            for r in baseline_results
        ],
        "improved_retrieval": [
            {k: v for k, v in r.items() if k != "background"} | {"background_preview": r["background"][:300]}
            for r in improved_results
        ],
        "weights": {
            "player_name_in_filename_bonus": PLAYER_NAME_BONUS,
            "player_name_in_text_bonus": PLAYER_TEXT_BONUS,
            "team_bonus": TEAM_BONUS,
            "opponent_bonus": OPPONENT_BONUS,
            "note": "Hand-picked for demonstration purposes only, not scientifically tuned.",
        },
        "commentary": commentary,
        "commentary_method": "local/deterministic template-based generation (no LLM, no API)",
    }
    with open(OUTPUT_DIR / "commentary.json", "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {OUTPUT_DIR / 'demo_report.txt'}")
    print(f"Saved: {OUTPUT_DIR / 'commentary.json'}")


if __name__ == "__main__":
    main()
