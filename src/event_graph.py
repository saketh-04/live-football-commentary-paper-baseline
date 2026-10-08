"""
event_graph.py

V2 Architecture Component: Directed Event Graph.
Models sequences of play-by-play events as an in-memory directed acyclic graph (DAG):
- Nodes: Discrete football events (player, team, action, timestamp, pitch zone).
- Edges: Transition relationships between consecutive events:
  - POSSESSION_CONTINUITY: Consecutive events by the same club
  - TURNOVER: Change of possession between clubs
  - OFFENSIVE_PROGRESSION: Ball advanced toward the opponent's penalty box
  - CHANCE_CREATION: Transition into a shot or cross
  - SEQUENCE_CULMINATION: Culmination into a goal or block

Extracts the exact causal and temporal predecessor path leading to any target event.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class GraphNode:
    """Represents an event node in the match sequence DAG."""
    node_id: int
    action: str
    player: str
    team: str
    start_time: float
    end_time: float
    location: str
    is_terminal: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "action": self.action,
            "player": self.player,
            "team": self.team,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "location": self.location,
            "is_terminal": self.is_terminal,
        }


@dataclass
class GraphEdge:
    """Directed temporal edge connecting event nodes."""
    source_id: int
    target_id: int
    transition_type: str  # 'POSSESSION_CONTINUITY', 'TURNOVER', 'OFFENSIVE_PROGRESSION', 'CHANCE_CREATION', 'CULMINATION'
    time_delta_sec: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "transition_type": self.transition_type,
            "time_delta_sec": round(self.time_delta_sec, 2),
            "description": self.description,
        }


@dataclass
class MatchEventGraph:
    """Full directed graph representation of a match event stream."""
    match_id: str
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def get_predecessor_chain(self, target_node_id: int, max_depth: int = 4) -> List[GraphNode]:
        """Returns the sequential predecessor path leading to target_node_id."""
        chain = []
        cur_id = target_node_id
        depth = 0
        while depth < max_depth:
            node = next((n for n in self.nodes if n.node_id == cur_id), None)
            if not node:
                break
            chain.append(node)
            incoming = [e for e in self.edges if e.target_id == cur_id]
            if not incoming:
                break
            cur_id = incoming[0].source_id
            depth += 1
        return list(reversed(chain))


class EventGraphBuilder:
    """Builds a MatchEventGraph from raw or structured play-by-play events."""

    def build_graph(self, match_id: str, events: List[Dict[str, Any]]) -> MatchEventGraph:
        graph = MatchEventGraph(match_id=match_id)
        if not events:
            return graph

        # Create nodes
        for idx, ev in enumerate(events):
            act = ev.get("action", "UNKNOWN").upper()
            player = ev.get("player") or ev.get("name") or "Unknown"
            team = ev.get("team", "Unknown")
            start_t = float(ev.get("start_time", 0.0))
            end_t = float(ev.get("end_time", start_t + 1.0))
            loc = ev.get("location", "Field")
            is_term = act in {"GOAL", "OUT"} or idx == len(events) - 1

            node = GraphNode(
                node_id=idx,
                action=act,
                player=player,
                team=team,
                start_time=start_t,
                end_time=end_t,
                location=loc,
                is_terminal=is_term,
            )
            graph.nodes.append(node)

        # Create edges between consecutive events
        for i in range(len(graph.nodes) - 1):
            src = graph.nodes[i]
            tgt = graph.nodes[i + 1]
            dt = max(0.0, tgt.start_time - src.end_time)

            if src.team != tgt.team:
                t_type = "TURNOVER"
                desc = f"Possession ceded from {src.team} to {tgt.team}"
            elif tgt.action in {"GOAL", "BALL PLAYER BLOCK"}:
                t_type = "CULMINATION"
                desc = f"{src.action} culminated in {tgt.action}"
            elif tgt.action in {"SHOT", "CROSS"}:
                t_type = "CHANCE_CREATION"
                desc = f"Build-up {src.action} created chance ({tgt.action})"
            elif "box" in tgt.location.lower() and "box" not in src.location.lower():
                t_type = "OFFENSIVE_PROGRESSION"
                desc = f"Progressed ball from {src.location} into {tgt.location}"
            else:
                t_type = "POSSESSION_CONTINUITY"
                desc = f"Maintained possession ({src.player} -> {tgt.player})"

            edge = GraphEdge(
                source_id=src.node_id,
                target_id=tgt.node_id,
                transition_type=t_type,
                time_delta_sec=dt,
                description=desc,
            )
            graph.edges.append(edge)

        return graph


if __name__ == "__main__":
    builder = EventGraphBuilder()
    test_events = [
        {"action": "CROSS", "name": "Boateng", "team": "Bayern Munich", "start_time": 13.76, "end_time": 14.04, "location": "OUT"},
        {"action": "SHOT", "name": "Kresic", "team": "Bayer Leverkusen", "start_time": 14.24, "end_time": 14.60, "location": "Right top box"},
        {"action": "GOAL", "name": "Kresic", "team": "Bayer Leverkusen", "start_time": 14.80, "end_time": 16.30, "location": "Right top box"},
    ]
    g = builder.build_graph("0008", test_events)
    print(f"Graph built: {len(g.nodes)} nodes, {len(g.edges)} edges")
    for e in g.edges:
        print(f"  [{e.source_id}] -> [{e.target_id}] ({e.transition_type}): {e.description}")
    chain = g.get_predecessor_chain(target_node_id=2)
    print("Predecessor chain to Goal:", [f"{n.player} ({n.action})" for n in chain])
