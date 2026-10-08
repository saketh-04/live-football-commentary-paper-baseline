# Context-Aware Multimodal Football Commentary Intelligence System
## Complete Research, Implementation, UI & Presentation Defense Guide

**Project Title:** Context-Aware Multimodal Football Commentary Intelligence System  
**Repository:** `https://github.com/saketh-04/live-football-commentary-paper-baseline`  
**Evaluation Scope:** 17 Matches, 139 Real Play-by-Play Football Events, 3,248 Biography Documents, SoccerNet / TrackLab Player Tracking  
**Target Audience:** Presentation Defense for Faculty Reviewers, Technical Evaluators, and Teammates  

---

## IMPORTANT PRESENTATION RULES

1. **Never claim unavailable data exists.** If the live running score is not present in the play-by-play telemetry, explicitly say: *"Live score telemetry is unavailable at this timestamp; the system deliberately abstains rather than hallucinating mid-game scorelines."*
2. **Never call the heuristic confidence a calibrated probability.** State clearly: *"This is an explainable composite heuristic combining retrieval, entity, factual, and visual signals, not a statistically calibrated Bayesian posterior probability."*
3. **Never call the event impact score an official football player rating.** State clearly: *"This is an internal event-importance aggregation over recorded telemetry, not an official FIFA, UEFA, or Opta player match rating."*
4. **Never claim 100% system accuracy.** Say: *"The deterministic factual verifier achieved 100% consistency on tested action contradictions by rule-based enforcement, but retrieval precision at 1 is 20.35%."*
5. **Never claim SOTA (State of the Art) without benchmark evidence.** Frame improvements strictly relative to the baseline: *"Our entity-aware reranking improves Precision@1 by +43.7% relative to the unweighted semantic retrieval baseline."*
6. **Never claim novelty without evidence.** State specifically what is implemented: *"Our contribution is the integration of structured telemetry, entity-aware reranking, graph dependency modeling, and post-generation deterministic verification into a unified broadcast pipeline."*
7. **Explain negative evaluation results honestly.** Be proud to highlight: *"Appending long temporal and match-state context strings into short dense queries caused query dilution (P@1 dropped from 0.2035 to 0.1681). This is a valuable finding demonstrating that contextual signals must be applied as post-retrieval reranking factors rather than dense query concatenations."*
8. **Distinguish video content from Streamlit UI overlays.** Do not claim the video MP4 file dynamically computes or displays Streamlit metrics; the video is the visual broadcast medium, while Streamlit computes and renders the telemetry intelligence overlays.
9. **Distinguish retrieved evidence from ground truth.** A retrieved Wikipedia document about a player is external background knowledge, not a live event confirmation.
10. **Distinguish factual verification from visual verification.** Factual verification deterministically validates textual claims against structured play-by-play telemetry; visual verification checks TrackLab bounding-box tracking coordinates in the 720p broadcast video frames.

---

# PART 1 — BASE PAPER / ORIGINAL BASELINE

### Simple Explanation (Layman's Terms)
The original project was an open-source prototype (`soccer-bg-commentary` developed by Japanese sports-AI researchers) designed to solve a simple problem: during live football broadcasts, there are frequent moments of silence between major events. The original system attempted to fill these quiet periods with "color commentary" (background facts about players or teams, like transfer fees or past records) by querying Wikipedia articles using OpenAI embeddings (`text-embedding-ada-002`) and prompting `gpt-4o` to generate short commentary sentences.

### Technical Explanation
The base system is an automated background commentary pipeline (`付加的情報の提供システムスクリプト` - Additional Information Provision System Script) utilizing:
- **Upstream Tooling:** Built on top of `zaemon1251-hesty/sn-script`, `tracklab` (SoccerNet player tracking / `sn-gamestate`), and `soccer-bg-script` (Action Spotting).
- **Play-by-Play (PBP) Input:** English and Japanese event streams (`play-by-play-en.jsonl`).
- **Retrieval Engine:** LangChain-based vector store using OpenAI `text-embedding-ada-002` (or local TF-IDF) over 3,249 Wikipedia player biography files (`data/addinfo_retrieval/`).
- **Generation Model:** OpenAI `gpt-4o` with strict prompt templates (`INSTRUCTION`), or a basic template dictionary (`play_by_play.py`).
- **Speech Synthesis:** Piper text-to-speech converting generated SRT subtitle files into WAV audio tracks (`srt_to_wav.py`).

### Structural Workflow of the Original Baseline
```
Raw Video & Spotting Labels (CSV)
  → Spotting Module (Heuristic 0/1 gate for comment insertion)
  → Query Construction (Raw text concatenation of previous comments + player names)
  → LangChain Retriever (OpenAI text-embedding-ada-002 + FAISS / TF-IDF)
  → GPT-4o Generation (Unconstrained generative prompt)
  → Piper TTS Audio / SRT Subtitles
```

### Baseline Summary Specification
- **Problem:** Filling broadcast pauses with background color commentary facts.
- **Input:** Play-by-play JSONL logs, Action spotting CSV, player tracking CSV, Wikipedia articles.
- **Processing:** Raw query string concatenation (previous comments + frame players + game metadata).
- **Model:** OpenAI `text-embedding-ada-002` embeddings + `gpt-4o` LLM + Piper TTS.
- **Output:** Subtitle files (`commentary.srt`), JSONL logs, and composite MP4 video with overlaid speech.
- **Limitations:**
  1. *Closed-Source / API Dependency:* Relied entirely on proprietary OpenAI APIs (`text-embedding-ada-002`, `gpt-4o`); could not run offline or without paid API quotas.
  2. *Lexical & Entity Distractor Vulnerability:* Raw query concatenation caused semantic drift. For example, querying "Boateng" returned Kevin-Prince Boateng or Stefan Aigner due to high topical overlap in German football text, with zero entity-level reranking.
  3. *No Temporal Sequence Understanding:* Treated each event in isolation or as an undifferentiated text blob of prior utterances, failing to model structured phases like `CROSS → SHOT → GOAL`.
  4. *No Match-State Tracking:* Did not model game minute, match tension, score differential, or whether an action was a sequence climax.
  5. *No Event Importance Modeling:* Treated routine midfield passes identically to decisive penalty kicks.
  6. *Uncontrolled Hallucinations:* Prompting `gpt-4o` freely produced factual errors, such as inventing goal outcomes or claiming saves on conceded goals.
  7. *No Visual Coordinate Grounding:* Player tracking was merely scraped as a name list without verifying whether bounding boxes existed or satisfied coordinate invariants.
  8. *No Confidence Metric:* Provided zero indication of whether retrieved facts or generated claims were trustworthy.
  9. *No Player-Level Intelligence:* Did not aggregate match events into cumulative player contributions or identify a match MVP.
  10. *No Formal Evaluation:* Lacked quantitative retrieval benchmarking (Precision@K, MRR), ablation studies, or error taxonomies.

---

# PART 2 — BASELINE VS ENHANCED SYSTEM

### Code-Level Feature Comparison Table

| Feature Dimension | Original Baseline (`src/soccer_bg_commentary/`) | Our Enhanced System (`src/`) | Why Added | How It Works | Evidence in Code | Evidence in UI | Research Benefit | Known Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Offline Vector Retrieval** | OpenAI `text-embedding-ada-002` API | Local `all-MiniLM-L6-v2` + FAISS `IndexFlatIP` | Eliminates external API dependencies and costs | Sentence-Transformers encodes 3,248 Wikipedia biographies into 384-d normalized vectors cached to disk | `src/improved_rag.py:10-53` | Page 3 (Retrieval), Page 5 (Evaluation) | 100% reproducible, zero-cost, local execution | Lower embedding dimension (384 vs 1536) |
| **Entity-Aware Reranking** | None (unweighted dense search) | Multi-signal entity scoring ($B_{\text{player}}, B_{\text{team}}, B_{\text{opp}}$) | Prevents semantically similar distractor articles from outranking true subjects | Boosts candidate score if player/team matches document title or text ($+0.30$ title, $+0.10$ text, $+0.05$ team) | `src/context_retrieval.py:73-86, 137-156` | Page 1 ("Retrieved Evidence"), Page 3 (A1 column) | P@1 rises from 0.1416 to 0.2035 (+43.7% relative gain) | Does not resolve entities absent from the Wikipedia corpus |
| **Temporal Context Window** | Concatenation of raw string utterances | Structured `EventContext` with sequence signatures and climax detection | Football events are sequential chains (`PASS → CROSS → GOAL`) | Tracks previous $N$ events, constructs sequence path, detects whether current event is a culmination | `src/temporal_context.py:17-68, 70-155` | Page 1 ("Temporal Sequence"), Page 2 (Step 4) | Models attack momentum and identifies sequence climaxes | Fixed sliding window size ($N=3$) |
| **Match-State Reasoning** | None (only static game title string) | Dynamic `MatchState` tracking match minute, half, late-game tension, score consequence | Football commentary tone depends heavily on game phase and scoreline | Computes minute from video offset + event timestamp; flags late-game ($\ge 75'$); checks score availability | `src/match_state.py:24-70, 72-180` | Page 1 ("Match State" card), Page 2 (Step 5) | Enables context-dependent commentary and consequence reasoning | Live running scores unavailable in PBP data; flagged honestly |
| **Explainable Event Importance** | None (all spotted timestamps treated equally) | Deterministic 4-tier importance scorer ($0.0 - 1.0$) | Commentary should prioritize high-leverage actions over routine events | Base action weight modulated by match minute boost ($+0.15$), score margin ($+0.10$), sequence climax ($+0.10$) | `src/event_importance.py:14-31, 59-133` | Page 1 ("Event Importance" bar), Page 4 (Profile) | Fully transparent, configurable action prioritization | Heuristic weights; not trained via reinforcement learning |
| **Directed Event Graph** | None | Directed Acyclic Graph (`MatchEventGraph`) with typed transitions | Models causal and tactical event transitions across the pitch | Connects events via semantic edges: `POSSESSION_CONTINUITY`, `TURNOVER`, `CHANCE_CREATION`, `CULMINATION` | `src/event_graph.py:22-97, 99-170` | Page 4 ("Directed Event Graph" HTML flow) | Extracts causal predecessor paths leading to scoring chances | Operates on discrete PBP events rather than continuous trajectories |
| **Counterfactual Reasoning** | None | Grounded, non-speculative structural interventions | Understands structural impact of specific actions without hallucination | Evaluates `EVENT_REMOVAL` and `SEQUENCE_TRUNCATION` on graph topology without inventing alternate physical reality | `src/counterfactual.py:23-45, 47-130` | Page 1 ("Counterfactual Reasoning" card), Page 2 (Step 8) | Strictly non-speculative; mathematically grounded in graph | Limited to graph removal; cannot simulate complex alternate tactics |
| **Adaptive Commentary Policy** | Static random rate selection (`force_rate`, `default_rate`) | Multi-register adaptive policy with speech gate | Selects appropriate broadcast style and cadence for each situation | Evaluates importance, climax status, entity depth to assign `Excited Climax`, `Analytical Tactical`, `Contextual`, or `Routine` | `src/commentary_policy.py:51-70, 131-255` | Page 1 ("Generated Commentary" badge), Page 2 (Step 9) | Dynamic broadcast cadence aligned with match tension | Template-based phrasing ensures verification safety |
| **Factual Verification** | None (unvalidated GPT-4o output) | Deterministic rule-based `FactualConsistencyVerifier` | Prevents hallucinations (wrong players, swapped teams, contradictory actions) | Extracts claims from commentary text and validates against telemetry; auto-corrects contradictions with audit log | `src/factual_verifier.py:18-58, 59-220` | Page 1 ("Factual Consistency" card), Page 5 (A5 metric) | Guarantees 100% action consistency on tested contradictions | Regex and dictionary-based; not open-domain NLI |
| **Visual Grounding** | Scraped name strings from video CSV | TrackLab 720p bounding box verifier with schema correction | Grounds commentary in confirmed visual video evidence | Resolves TrackLab $[x_1, y_1, w, h]$ schema into valid $[x_1, y_1, x_2, y_2]$ coordinates; verifies on-screen presence | `src/visual_verifier.py:32-73, 75-230` | Page 1 ("Visual Grounding" card), Page 2 (Step 3) | Verifies whether player was actually in broadcast camera frame | Player off-screen due to camera pan is honestly reported |
| **Multimodal Confidence** | None | 4-signal composite heuristic score ($0.0 - 1.0$) | Informs broadcaster of reliability across all multimodal signals | Weighted sum: $0.35 \cdot \text{Retrieval} + 0.25 \cdot \text{Entity} + 0.20 \cdot \text{Factual} + 0.20 \cdot \text{Visual}$ | `src/confidence_scorer.py:17-39, 41-125` | Page 1 ("Multimodal Confidence" gauge), Page 2 (Step 10) | Explainable breakdown of confidence contributors | Explicitly documented as an uncalibrated heuristic |
| **Player Intelligence & MVP** | None | `PlayerIntelligenceAggregator` with cumulative impact & MVP ranking | Provides match-level player performance intelligence | Sums event importances per player, counts key moments, ranks contributors, derives MVP with evidence | `src/player_intelligence.py:24-48, 50-180` | Page 1 ("Player Performance Intelligence" table & MVP) | Synthesizes telemetry into post-match player contribution | Derived strictly from event data; no fake xG/xA claims |
| **Interactive Dashboard** | Basic React / Node.js prototype (`front/`) | Comprehensive 6-page Streamlit Research Dashboard | Interactive live exploration of complete multimodal reasoning | Real-time multi-match selector, synchronized video, pipeline stepper, retrieval ablation, graph analytics | `src/streamlit_app.py:1-1576` | Live web application at port 8501 | Allows faculty and reviewers to audit every system decision | Runs on local Streamlit server |
| **Formal Evaluation & Ablation** | None | Benchmark over 17 matches (139 events) with error taxonomy | Scientific rigor and empirical validation of research hypotheses | Evaluates P@1, P@3, R@5, MRR across conditions A0–A5; categorizes failure modes in error taxonomy | `src/evaluation.py`, `src/ablation.py`, `src/error_analysis.py` | Page 5 ("Research Evaluation & Ablation Suite") | Quantifies impact of each module and reveals query dilution | Evaluated over 113 events with ground truth coverage |

### Final Summary: Baseline vs Enhanced System

```
ORIGINAL BASELINE (zaemon1251-hesty)         OUR ENHANCED SYSTEM
-----------------------------------         -------------------
OpenAI API Dependent ($)                    100% Local & Free (Sentence-Transformers + FAISS)
Unweighted Dense Query Concatenation        Entity-Aware Multi-Signal Reranking (+43.7% P@1)
Isolated Event Processing                   Temporal Sequence Modeling + Climax Detection
No Match-State Awareness                    Match Minute + Score State Tracking
No Importance Scoring                       Explainable 4-Tier Event Importance
No Structural Event Modeling                Directed Event Acyclic Graph (DAG)
No Causal Analysis                          Grounded Counterfactual Graph Interventions
Static Output Generation                    Adaptive Commentary Policy (4 Speech Registers)
Ungrounded Hallucinations                   Deterministic Factual Verifier (Auto-Correction)
Raw CSV String Scraping                     TrackLab 720p Bounding Box Coordinate Grounding
No Reliability Measure                      Explainable 4-Signal Multimodal Confidence Score
No Player-Level Aggregation                 Player Intelligence Engine + Data-Derived MVP
Unbenchmarked Prototype                     Formal Ablation Suite (A0-A5) + Error Taxonomy
Bare Subtitle Scripts                       Interactive 6-Page Streamlit Research Platform
```

---

# PART 3 — COMPLETE PIPELINE TRACE

The system executes a strictly ordered 14-stage multimodal intelligence pipeline:

```
[1] Football Video & PBP Event Stream
         ↓
[2] Temporal Context Window Construction (src/temporal_context.py)
         ↓
[3] Entity Resolution & Telemetry Disambiguation
         ↓
[4] Match-State Tracking (src/match_state.py)
         ↓
[5] Directed Event Graph Construction (src/event_graph.py)
         ↓
[6] Explainable Event Importance Scoring (src/event_importance.py)
         ↓
[7] Context-Aware / Entity-Aware Retrieval (src/context_retrieval.py)
         ↓
[8] Grounded Counterfactual Reasoning (src/counterfactual.py)
         ↓
[9] Adaptive Commentary Policy Decision (src/commentary_policy.py)
         ↓
[10] Grounded Commentary Draft Generation
         ↓
[11] Factual Consistency Verification (src/factual_verifier.py)
         ↓
[12] Visual Grounding Verification (src/visual_verifier.py)
         ↓
[13] Multimodal Confidence Estimation (src/confidence_scorer.py)
         ↓
[14] Player-Level Performance Intelligence & MVP Selection (src/player_intelligence.py)
```

### Detailed Stage-by-Stage Breakdown (with Match 0012 Example)

#### Stage 1: Structured Event Ingestion
- **Input:** Raw JSONL lines from `data/demo/pbp/0012/play-by-play-en.jsonl`.
- **Processing:** Parses timestamps, action categories, player names, club affiliations, and pitch zones.
- **Output:** Raw dictionary representation: `{"start_time": "14.480", "end_time": "15.980", "text": "And it's a goal! Manchester United delivers!", "action": "GOAL", "location": "OUT", "name": "Blind", "team": "Manchester United"}`.
- **Why Needed:** Standardizes unstructured or loosely structured event logs into typed data.
- **Implementation:** `src/temporal_context.py:TemporalContextBuilder.build_match_contexts`.
- **Match 0012 Example:** Event #5 at 14.48s: Daley Blind scores for Manchester United.
- **What Can Go Wrong:** Missing fields, non-numeric timestamps, or corrupted JSON strings.
- **Fallback:** Defaults start time to 0.0s, duration to 1.0s, and player/team to "Unknown".

#### Stage 2: Temporal Context Modeling
- **Input:** Current event + sliding history window of prior $N=3$ events.
- **Processing:** Assembles the sequence path; evaluates whether the action represents an attacking culmination.
- **Output:** `EventContext` object with `sequence_signature = "CROSS -> SHOT -> BALL PLAYER BLOCK -> GOAL"` and `is_sequence_climax = True`.
- **Why Needed:** Commentary must reflect build-up narrative rather than viewing actions in isolation.
- **Implementation:** `src/temporal_context.py:118-152`.
- **Match 0012 Example:** Sequence culminates after crosses from Bailly and Blind and a blocked shot.
- **What Can Go Wrong:** Events out of chronological order in raw data.
- **Fallback:** Sequence resets to current event alone; `is_sequence_climax` defaults to `False`.

#### Stage 3: Entity Resolution
- **Input:** Player string `"Blind"`, team `"Manchester United"`.
- **Processing:** Normalizes strings, resolves active team and identifies opponent (`"Leicester"`).
- **Output:** Resolved entity tuple: `(player="Blind", team="Manchester United", opponent="Leicester")`.
- **Why Needed:** Eliminates capitalization and spacing mismatches before database queries.
- **Implementation:** `src/temporal_context.py:94-102`.
- **Match 0012 Example:** Identifies Manchester United as active club and Leicester as defending opponent.
- **What Can Go Wrong:** Ambiguous surnames (e.g., "Boateng" in Bayern vs Leverkusen).
- **Fallback:** Retains raw surname and applies team context for downstream disambiguation.

#### Stage 4: Match-State Tracking
- **Input:** Match ID `"0012"`, timestamp 14.48s, metadata from `data/demo/sample_metadata.csv`.
- **Processing:** Computes match minute (first half, minute 14); checks live score availability.
- **Output:** `MatchState(match_id="0012", home_team="Manchester United", away_team="Leicester", half=1, match_minute=14, is_score_available=False, state_description="Score information unavailable", consequence_description="Score consequence unavailable")`.
- **Why Needed:** Prevents commentary from inventing fictional running scores.
- **Implementation:** `src/match_state.py:MatchStateTracker.compute_state_for_event`.
- **Match 0012 Example:** Minute 14, Half 1. Full-time result (4-1) stored for reference only.
- **What Can Go Wrong:** Raw telemetry does not contain live running scoreboards.
- **Fallback:** Explicitly flags `is_score_available = False` and suppresses score change phrasing.

#### Stage 5: Directed Event Graph Construction
- **Input:** Match event stream for Match 0012.
- **Processing:** Constructs graph nodes and derives directed transition edges (`CHANCE_CREATION`, `CULMINATION`).
- **Output:** `MatchEventGraph` with 5 nodes and 4 directed edges.
- **Why Needed:** Explicitly models causal dependencies between passes, shots, and blocks.
- **Implementation:** `src/event_graph.py:EventGraphBuilder.build_graph`.
- **Match 0012 Example:** Edge from Node 2 (`SHOT`) to Node 3 (`BALL PLAYER BLOCK`) labeled `CULMINATION`.
- **What Can Go Wrong:** Disconnected timestamps or isolated single events.
- **Fallback:** Builds isolated node graph without edges.

#### Stage 6: Explainable Event Importance Scoring
- **Input:** Action `"GOAL"`, minute 14, score difference 0, `is_sequence_climax = True`.
- **Processing:** Base score for `GOAL` is 1.00; sequence climax adds $+0.10$; score clamped to $[0.0, 1.0]$.
- **Output:** `ImportanceBreakdown(base=1.00, time_mult=1.0, margin_mult=1.0, seq_bonus=0.10, final_importance=1.00, tier="Critical")`.
- **Why Needed:** Governs broadcast volume, styling, and commentary urgency.
- **Implementation:** `src/event_importance.py:EventImportanceScorer.compute_importance`.
- **Match 0012 Example:** Blind's goal achieves maximum score 1.00 (`Critical`).
- **What Can Go Wrong:** Unrecognized action type.
- **Fallback:** Assigns default base score 0.25 (`Routine` tier).

#### Stage 7: Context-Aware / Entity-Aware Retrieval
- **Input:** Query `"Blind Manchester United GOAL"`, candidate pool $K=60$.
- **Processing:** FAISS cosine search + entity reranking bonuses ($+0.30$ title match for "Daley Blind").
- **Output:** Top document: `Daley_Blind.txt` with final score `0.8527` (Semantic: 0.5527, Entity bonus: +0.3000).
- **Why Needed:** Retrieves accurate player biographical background while rejecting distractors.
- **Implementation:** `src/context_retrieval.py:ContextAwareRetriever.retrieve`.
- **Match 0012 Example:** Accurately retrieves Daley Blind's biography over other defenders.
- **What Can Go Wrong:** Player missing from Wikipedia corpus (e.g., Eric Bailly in Match 0012 Event 1).
- **Fallback:** Returns highest semantic candidate; entity bonus remains 0.0; confidence is lowered.

#### Stage 8: Grounded Counterfactual Reasoning
- **Input:** Event #5 (`GOAL`), sequence signature, match state.
- **Processing:** Applies structural event removal intervention on graph topology.
- **Output:** `CounterfactualScenario(intervention="EVENT_REMOVAL", consequence="Without the recorded GOAL event, the sequence concludes with the preceding action (BALL PLAYER BLOCK). Any score consequence associated with this goal is absent.")`.
- **Why Needed:** Enables causal reasoning about event importance without speculative simulation.
- **Implementation:** `src/counterfactual.py:CounterfactualEngine.analyze_event_counterfactual`.
- **Match 0012 Example:** Identifies that removing Blind's goal halts the sequence at Slimani's block.
- **What Can Go Wrong:** Attempting to predict hypothetical future physics (e.g., "keeper would have saved").
- **Fallback:** Strictly constrained to structural graph topology; non-speculative flag set to `True`.

#### Stage 9: Adaptive Commentary Policy
- **Input:** Action `"GOAL"`, importance 1.00, climax `True`.
- **Processing:** Matches `ALWAYS_SPEAK_ACTIONS`; selects `Excited Climax` style with `Urgent / Climax` tempo.
- **Output:** `CommentaryPolicyDecision(should_commentate=True, selected_style="Excited Climax", tempo="Urgent / Climax")`.
- **Why Needed:** Adjusts broadcast energy and wording to match match context.
- **Implementation:** `src/commentary_policy.py:AdaptiveCommentaryPolicy.evaluate_policy`.
- **Match 0012 Example:** Triggers excited climax register for goal event.
- **What Can Go Wrong:** Repetitive identical events causing broadcast fatigue.
- **Fallback:** Novelty gate compresses consecutive repeated actions into concise phrasing.

#### Stage 10: Grounded Commentary Generation
- **Input:** Event attributes, policy style, match state.
- **Processing:** Constructs grounded script adhering to factual constraints (no score claim if unavailable).
- **Output:** `"GOAL! What a finish by Blind for Manchester United!"`.
- **Why Needed:** Produces fluent, broadcast-ready commentary text.
- **Implementation:** `src/commentary_policy.py:204-255`.
- **Match 0012 Example:** Generates excited finish commentary without hallucinating scoreline.
- **What Can Go Wrong:** Inserting speculative score claims like "takes the lead".
- **Fallback:** `_score_phrase` returns empty string when score is unavailable.

#### Stage 11: Factual Consistency Verification
- **Input:** Generated text, expected player `"Blind"`, team `"Manchester United"`, action `"GOAL"`.
- **Processing:** Verifies action keywords, checks team swap, verifies player mention.
- **Output:** `VerificationReport(status="VERIFIED", was_corrected=False, checks=[Action: OK, Team: OK, Player: OK])`.
- **Why Needed:** Eliminates factual contradictions before public broadcast.
- **Implementation:** `src/factual_verifier.py:FactualConsistencyVerifier.verify_commentary`.
- **Match 0012 Example:** All 4 verification checks pass with zero contradictions.
- **What Can Go Wrong:** Commentary text says "Great save to deny Blind" on a GOAL event.
- **Fallback:** Auto-corrector detects contradiction and replaces text with verified template.

#### Stage 12: Visual Grounding Verification
- **Input:** Match `"0012"`, half 1, player `"Blind"`, timestamp 14.48s.
- **Processing:** Queries TrackLab tracking CSV for match 0012; checks for player bounding box.
- **Output:** `VisualVerificationResult(is_grounded=True, bbox=[x1, y1, x2, y2], confidence=1.0)` or `False` if outside camera pan.
- **Why Needed:** Proves whether the player was physically visible on screen during the broadcast action.
- **Implementation:** `src/visual_verifier.py:VisualFrameVerifier.verify_player_in_frame`.
- **Match 0012 Example:** Checks 11 tracked players in Match 0012 frame data.
- **What Can Go Wrong:** Player is off-screen due to camera zoom on the ball carrier.
- **Fallback:** Reports `visual_confidence = 0.5` if teammates visible, or `0.0` with explicit camera boundary note.

#### Stage 13: Multimodal Confidence Estimation
- **Input:** Retrieval score, entity bonus, verification status (`VERIFIED`), visual confidence.
- **Processing:** Weighted composite: $0.35 \cdot \text{Ret} + 0.25 \cdot \text{Ent} + 0.20 \cdot \text{Fact} + 0.20 \cdot \text{Vis}$.
- **Output:** `ConfidenceBreakdown(composite_confidence=0.72, tier="Moderate Confidence", is_calibrated=False)`.
- **Why Needed:** Provides transparent signal quality indication to broadcast operators.
- **Implementation:** `src/confidence_scorer.py:MultimodalConfidenceScorer.compute_confidence`.
- **Match 0012 Example:** Computes composite score for Daley Blind's goal.
- **What Can Go Wrong:** Confusing composite heuristic with statistical probability.
- **Fallback:** Explicitly displays warning: "Explainable heuristic - not a calibrated probability."

#### Stage 14: Player Intelligence Aggregation & MVP Selection
- **Input:** All 5 EventContexts and ImportanceBreakdowns for Match 0012.
- **Processing:** Sums importance per player: Blind (Cross 0.50 + Shot 0.87 + Goal 1.00 = 2.37); Bailly (0.50); Slimani (0.39).
- **Output:** Daley Blind ranked #1 (`High` impact tier, Total Impact: 2.37, 2 key events) → Designated MVP.
- **Why Needed:** Provides post-match player contributions derived strictly from event telemetry.
- **Implementation:** `src/player_intelligence.py:PlayerIntelligenceAggregator`.
- **Match 0012 Example:** Daley Blind designated Top Contributor / MVP with transparent breakdown.
- **What Can Go Wrong:** Claiming ungrounded statistics (e.g., inventing expected goals xG).
- **Fallback:** Strictly restricts metrics to observed action counts and cumulative importance sums.

---

# PART 4 — WEBSITE PAGE-BY-PAGE AUDIT

The interactive Streamlit research platform (`src/streamlit_app.py`) provides 6 comprehensive pages.

```
STREAMLIT RESEARCH PLATFORM SITEMAP
├── PAGE 1: MATCH — Match Intelligence (Sequential End-to-End Broadcast Hub)
├── PAGE 2: AI — AI Reasoning Pipeline (10-Step Sequential Telemetry Trace)
├── PAGE 3: RETRIEVAL — Retrieval Analysis (A0 vs A1 vs A5 Side-by-Side Comparison)
├── PAGE 4: GRAPH — Event Graph & Analytics (DAG Flow, Transition Edges & Profiles)
├── PAGE 5: RESEARCH — Research Evaluation (Ablation Benchmark, Modality Matrix & Errors)
└── PAGE 6: METHOD — Methodology (Technical Architecture & Mathematical Formulations)
```

---

### PAGE 1 — MATCH INTELLIGENCE

#### A. Purpose of the Page
The flagship operational screen. It provides a clean, sequential broadcast view combining live video playback, play-by-play event selection, multimodal verification, counterfactual analysis, grounded commentary, and player performance intelligence.

#### B. Top-to-Bottom Section Breakdown
1. **Match Header (`lines 556-590`):**
   - *Visible Content:* Home Team Name, Away Team Name, Live Score Display, Competition Name, Match Date, Half, Full-Time Reference Result.
   - *Displayed Numbers:* E.g., `Score: -` (or `Score data unavailable`), `Half: 1`, `Full-time: Manchester United 4 - 1 Leicester`.
   - *Code Source:* `src/match_state.py:MatchStateTracker.compute_state_for_event`.
   - *Formula:* Live score displayed only if `state.is_score_available == True`; otherwise displays `-`. Full-time score extracted via regex: `re.search(r"(\d+)\s*-\s*(\d+)", filename)`.
   - *Why It Exists:* Sets the official match context without hallucinating running scores.
2. **Match Video (`lines 595-602`):**
   - *Visible Content:* Embedded MP4 video player with standard playback controls.
   - *Displayed Video:* Resolves local MP4 file from `outputs/demo-step4/` or `outputs/demo-step3/`.
   - *Caption:* `Source: [filename].mp4 (MP4 - AI-generated TTS commentary + tracking overlay)`.
   - *Why It Exists:* Allows the evaluator to visually observe the real football sequence.
3. **Event Timeline Selector (`lines 603-630`):**
   - *Visible Content:* Dropdown selector + vertical visual timeline list showing minute/second, action name, and player.
   - *Displayed Numbers:* Event timestamp `Minute'Second"` (e.g., `00'14"`).
   - *Code Source:* `ctx.minute` and `ctx.second` computed from `ctx.start_time // 60` and `ctx.start_time % 60`.
   - *Why It Exists:* Allows immediate interactive navigation across any event in the match.
4. **Selected Event Identity Card (`lines 676-701`):**
   - *Visible Content:* 4 grid cells: Action, Player, Team, Timestamp.
   - *Displayed Numbers:* Timestamp in seconds (e.g., `14.5s`).
   - *Code Source:* Direct fields from `EventContext` (`src/temporal_context.py`).
5. **Temporal Sequence Card (`lines 702-730`):**
   - *Visible Content:* Sequence flow diagram (e.g., `CROSS -> SHOT -> GOAL`), Sequence Climax chip, phase narrative explanation.
   - *Code Source:* `render_sequence()` rendering `ctx.sequence_signature`.
   - *Why It Exists:* Demonstrates build-up context and highlights sequence climaxes.
6. **Match State Card (`lines 731-766`):**
   - *Visible Content:* Live score (`Score data unavailable`), Late Game badge, Pitch minute, State description, Consequence note.
   - *Displayed Numbers:* Pitch minute (e.g., `Min 14'`).
   - *Code Source:* `src/match_state.py`.
7. **Event Importance Card (`lines 767-784`):**
   - *Visible Content:* Tier badge (e.g., `Critical`), numeric importance score (e.g., `0.87` or `1.00`), color-coded progress bar, explanation text.
   - *Formula:* $\text{Score} = (\text{Base} \times M_{\text{time}} \times M_{\text{margin}}) + B_{\text{climax}}$.
8. **Retrieved Evidence List (`lines 785-813`):**
   - *Visible Content:* Top-3 retrieved Wikipedia documents with rank, document stem, final score, breakdown scores (Semantic, Entity, Temporal, Match-State), and text snippet.
   - *Displayed Numbers:* Final score (e.g., `0.8527`), Semantic score (e.g., `0.5527`), Entity bonus (`+0.300`).
   - *Formula:* Detailed in Part 8.
9. **Factual & Visual Verification Columns (`lines 814-878`):**
   - *Factual Box:* `VERIFIED` chip, 4 individual checks (Player, Team, Opponent, Action) with pass/fail icons, audit summary.
   - *Visual Box:* Grounding status chip (`VISUALLY CONFIRMED`, `PARTIAL`, or `NOT IN TRACKING CSV`), explanation, bounding box coordinates `[x1, y1] -> [x2, y2]`, jersey number, visible teammates list.
   - *Displayed Numbers:* 720p coordinates (e.g., `[318, 383] -> [347, 477]`), Jersey # (e.g., `#25`).
10. **Multimodal Confidence Gauge (`lines 879-918`):**
    - *Visible Content:* Large composite percentage (e.g., `41%` or `72%`), tier label (`Moderate Confidence`), progress bar, 4 sub-metric boxes: Retrieval (35%), Entity (25%), Factual (20%), Visual (20%).
    - *Formula:* Detailed in Part 11.
11. **Grounded Counterfactual Reasoning Card (`lines 919-942`):**
    - *Visible Content:* Intervention type (`EVENT_REMOVAL`), observed state, hypothetical intervention, structural consequence description.
12. **Generated Commentary Box (`lines 943-967`):**
    - *Visible Content:* `SPEAK` chip, Style badge (`EXCITED CLIMAX`), Cadence badge (`Urgent / Climax`), Verification badge (`[OK] VERIFIED`), large formatted quote of commentary text, policy rationale note.
13. **Explainable Grounding Evidence Chain ("Why This Commentary?") (`lines 968-1005`):**
    - *Visible Content:* 8-item structured audit checklist proving telemetry grounding for Action, Player, Team, Retrieved Knowledge, Temporal Sequence, Visual Grounding, Factual Verification, and Heuristic Confidence.
14. **Player Performance Intelligence & Contributor Table (`lines 1006-1055`):**
    - *Visible Content:* MVP banner with player name, club, total impact score, and rationale; interactive contributor table listing Player, Team, Total Impact Score, Impact Tier, Total Events, Key Moments, and Action breakdown.

---

### PAGE 2 — AI REASONING PIPELINE

- **Purpose:** Educational step-by-step audit tracing the 10 sequential transformations from raw play-by-play telemetry to verified broadcast script for any selected event.
- **Layout:** Event dropdown selector at top; 10 sequential expandable JSON inspection panels:
  - *Step 1:* Event Ingestion & PBP Parsing (Raw text, action category, timestamp window, pitch location).
  - *Step 2:* Entity Extraction & Disambiguation (Player, active team, defending opponent).
  - *Step 3:* Visual Frame Grounding (TrackLab confirmation, timestamp, bounding box corners, jersey #, camera coverage note).
  - *Step 4:* Temporal Match Context (Sequence signature, climax flag, match minute).
  - *Step 5:* Match-State Reasoning (Clubs, full-time reference, live score availability, consequence).
  - *Step 6:* Event Importance Scoring (Base score, multipliers, final score, tier, rationale).
  - *Step 7:* Entity-Aware Context Retrieval (Query, top document, semantic score, entity bonus, final score, match reasons).
  - *Step 8:* Grounded Counterfactual Reasoning (Intervention type, structural consequence, non-speculative flag).
  - *Step 9:* Adaptive Commentary Policy (Broadcast gate, selected style, cadence, policy rationale, script).
  - *Step 10:* Factual Verification & Multimodal Confidence (Verification status, auto-correction status, composite confidence, tier, uncalibrated disclosure).
- **What to Say in Presentation:** *"This page opens the black box. For any event, faculty can inspect the exact data structures passed between each of our 10 architectural modules."*

---

### PAGE 3 — RETRIEVAL ANALYSIS

- **Purpose:** Side-by-side scientific inspection of retrieval candidate scoring across three ablation conditions:
  - *Column 1:* `A0 - Semantic Baseline` (`all-MiniLM-L6-v2` dense cosine similarity only).
  - *Column 2:* `A1 - Entity-Aware Retrieval` (A0 + additive bonuses for player, team, opponent).
  - *Column 3:* `A5 - Full Context-Aware Retrieval` (A1 + temporal build-up + match-state context).
- **Displayed Elements:** Top-5 candidate documents per column showing rank, document stem, semantic score, entity bonus, and final score.
- **Top Candidate Rationale Card:** Displays the specific trigger reasons for the #1 document (e.g., `Player exact match in title (+0.30)`, `Club 'Chelsea' cited (+0.05)`).
- **Scientific Callout:** Features a prominent highlighted box explaining **Dense Vector Query Dilution**: why appending temporal strings into dense queries reduces embedding precision and why contextual signals belong in post-retrieval reranking.

---

### PAGE 4 — EVENT GRAPH & MATCH ANALYTICS

- **Purpose:** Graph topology inspection and quantitative match event diagnostics.
- **Displayed Metrics (Top Cards):**
  - *Total Events:* Count of events in match (e.g., `5` in Match 0012, `3` in Match 0008).
  - *Action Types:* Count of unique actions (e.g., `CROSS`, `SHOT`, `GOAL`).
  - *Competing Teams:* Count of distinct clubs (typically `2`).
  - *PBP Window:* Duration of clip in seconds (e.g., `16.0s`).
- **Directed Event Graph Display:** Visual HTML flow chart rendering each event as a bordered node connected by color-coded directed edges representing transition semantics:
  - `CULMINATION` (Green `#10b981`)
  - `CHANCE_CREATION` (Sky Blue `#38bdf8`)
  - `TURNOVER` (Amber `#f59e0b`)
  - `OFFENSIVE_PROGRESSION` (Purple `#a78bfa`)
  - `POSSESSION_CONTINUITY` (Gray `#4b5563`)
- **Edge Diagnostics Table:** Displays From Node, To Node, Transition Type, Time Delta ($\Delta t$ in seconds), and Description.
- **Charts:**
  - *Action Distribution:* Bar chart of action frequency.
  - *Importance Profile Over Time:* Line chart plotting event importance ($0.0 - 1.0$) across match timestamps, visually revealing the crescendo toward attacking climaxes.

---

### PAGE 5 — RESEARCH EVALUATION

- **Purpose:** Formal scientific evaluation benchmark over 17 matches and 139 play-by-play events.
- **Header Metric Boxes:**
  - *Total Events:* `139`
  - *Events w/ Gold Truth:* `113`
  - *Corpus Coverage:* `81.29%` (113 / 139)
  - *Peak P@1 (A1):* `0.203` (0.2035)
- **Ablation Progression Table:** Complete results across conditions A0–A5 (P@1, P@3, R@5, MRR, Factual Consistency).
- **Retrieval Metrics Comparison Chart:** Grouped bar chart comparing P@1, P@3, and MRR across conditions.
- **Scientific Insights Table:** Documented findings on Entity-Aware gains, Dense Query Dilution, and Factual Verifier efficacy.
- **Empirical Modality Coverage Matrix:** Complete table auditing all 25 clips / 17 matches across PBP, Video, Tracking, and Ground-Truth Biographies.
- **Error Taxonomy Section:**
  - *Bar Chart:* Failure category counts (`wrong_player: 94`, `semantically_similar_distractor: 97`, `missing_match_score: 113`, `missing_corpus_entity: 26`, `generation_contradiction: 6`).
  - *Representative Failure Cases:* Expandable JSON cards detailing real examples from the dataset.
- **End-to-End Integration Test Summary (Match 0008):** Verified status (3/3 factually verified, Average confidence: 0.665, Status: PASS).

---

### PAGE 6 — METHODOLOGY

- **Purpose:** Full technical and academic reference document.
- **Contents:**
  - *Formal Research Question:* Verbatim problem formulation.
  - *V2 Architecture Pipeline Diagram:* Full text-based ASCII flowchart.
  - *Scoring Equations:* Exact LaTeX formulas for Multi-Signal Retrieval Ranking and Multimodal Confidence.
  - *Documented System Limitations Table:* Explicit disclosure of missing live scores, broadcast frame boundaries, corpus coverage, query dilution, and uncalibrated confidence.
  - *Module Directory Table:* Path, filename, and architectural role of all 14 Python modules in `src/`.
  - *Reproduction Commands:* Exact `uv run` commands for running tests, regenerating benchmarks, and launching the application.

---

# PART 5 — MATCH INTELLIGENCE AUDIT: MATCH 0012 CASE STUDY

Match 0012 (`Manchester United vs Leicester`, First Half, 11.60s – 15.98s) is the premier presentation example.

```
MATCH 0012 EVENT STREAM
├── Event 1 (11.60s): CROSS by Bailly (Manchester United) [Location: Right top corner]
├── Event 2 (12.16s): CROSS by Blind (Manchester United)  [Location: Right top corner]
├── Event 3 (13.36s): SHOT by Blind (Manchester United)   [Location: Right top box]
├── Event 4 (14.00s): BALL PLAYER BLOCK by Slimani (Leicester) [Location: Right top box]
└── Event 5 (14.48s): GOAL by Blind (Manchester United)   [Location: OUT]
```

### Detailed Inspection of Selected Event: Event #3 (Blind Shot) vs Event #5 (Blind Goal)

#### 1. Match Selector
- *What:* Dropdown in sidebar set to `"0012"`.
- *Why:* Loads `data/demo/pbp/0012/play-by-play-en.jsonl`.
- *Source:* `Path("data/demo/pbp").iterdir()`.

#### 2. Match Metadata & Header
- *Home Team:* Manchester United
- *Away Team:* Leicester
- *Competition:* Premier League (derived from `sample_metadata.csv`)
- *Date:* 2016-09-24
- *Half:* 1
- *Full-Time Score:* Manchester United 4 - 1 Leicester (from folder naming metadata)
- *Live Score:* Displayed as `-` or `Score data unavailable` because live running scores are absent in PBP data.

#### 3. Video Display
- *File:* `outputs/demo-step3/0012-en.mp4` (or step4 equivalent).
- *Content:* 720p broadcast clip showing Manchester United's attacking wave culminating in Daley Blind's strike.

#### 4. Event #3: Daley Blind SHOT (13.36s)
- **Action:** `SHOT`
- **Player:** Daley Blind
- **Team:** Manchester United
- **Timestamp:** 13.36s
- **Temporal Sequence:** `CROSS -> CROSS -> SHOT`
- **Sequence Climax:** `True` (Attacking phase culminates in a scoring effort)
- **Event Importance:**
  - Base score: $0.70$ (for `SHOT`)
  - Sequence climax bonus: $+0.10$
  - Close game margin bonus: $+0.10$ (default $|0| \le 1$)
  - Time multiplier: $1.0$ (Minute 14)
  - Calculation: $(0.70 \times 1.0 \times 1.10) + 0.10 = 0.77 + 0.10 = 0.87$
  - **Displayed Value: 0.87** (`Critical` tier)
- **Retrieved Evidence:**
  - Document: `Daley_Blind.txt`
  - Semantic score: $0.512$
  - Entity bonus: $+0.300$ (Player exact match in title) $+ 0.050$ (Club Manchester United cited) $= +0.350$
  - Final Retrieval Score: $(0.512 + 0.350) \times (1.0 + 0.87 \times 0.05) \approx 0.899$
- **Factual Verification:**
  - Status: `VERIFIED`
  - Checks: Player (Blind: PASS), Team (Manchester United: PASS), Opponent (Leicester: PASS), Action (SHOT: PASS).
- **Visual Grounding:**
  - Status: Evaluated against `players_in_frames_sn_gamestate.csv`. If Blind's tracking box is present, returns `[x1, y1, x2, y2]`; if off-screen during camera pan, reports teammates visible in frame.
- **Multimodal Confidence Calculation:**
  $$\text{Retrieval Component} = \min\left(1.0, \max\left(0.0, \frac{0.512 - 0.2}{0.6}\right)\right) = \frac{0.312}{0.6} = 0.520$$
  $$\text{Entity Component} = \min\left(1.0, \frac{0.350}{0.45}\right) = 0.778$$
  $$\text{Factual Component} = 1.000 \quad (\text{VERIFIED, not corrected})$$
  $$\text{Visual Component} = 0.500 \quad (\text{Teammates in frame})$$
  $$\text{Composite} = (0.520 \times 0.35) + (0.778 \times 0.25) + (1.000 \times 0.20) + (0.500 \times 0.20)$$
  $$\text{Composite} = 0.182 + 0.195 + 0.200 + 0.100 = 0.677 \approx 68\% \quad (\text{Moderate Confidence})$$
- **Counterfactual Reasoning:**
  - Intervention: `EVENT_REMOVAL`
  - Consequence: *"Without this SHOT, the offensive build-up (CROSS -> CROSS -> SHOT) does not culminate in a scoring attempt, leaving possession in the right top box."*
- **Commentary Generation:**
  - Style: `Excited Climax`
  - Generated Text: `"Dangerous shot by Blind for Manchester United in the right top box!"`

#### 5. Player Intelligence & MVP Ranking for Match 0012
Aggregated across all 5 events:
1. **Daley Blind (Manchester United):**
   - Event 2: CROSS (Importance: $0.50$)
   - Event 3: SHOT (Importance: $0.87$)
   - Event 5: GOAL (Importance: $1.00$)
   - **Total Impact Score:** $0.50 + 0.87 + 1.00 = 2.37$
   - Key Moments: 2 (`SHOT`, `GOAL`)
   - **Designated Match MVP / Top Contributor**
2. **Eric Bailly (Manchester United):**
   - Event 1: CROSS (Importance: $0.50$)
   - Total Impact Score: $0.50$
3. **Islam Slimani (Leicester):**
   - Event 4: BALL PLAYER BLOCK (Importance: $0.39$)
   - Total Impact Score: $0.39$

---

# PART 6 — DATA UNAVAILABLE AUDIT: BUGS VS CORRECT SCIENTIFIC BEHAVIOR

| UI String / State | Where Seen | Root Cause in Data | System Classification | Scientific Justification | What to Tell Faculty |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **"Score data unavailable"** | Match Header & Match State Card | Raw PBP JSONL logs contain event actions and timestamps but **do not track live minute-by-minute running scores**. | **CORRECT BEHAVIOR** (Scientific Integrity) | Extrapolating a 90-minute full-time score (e.g., 4-1) to minute 14 of the first half would be a factual hallucination. The system deliberately abstains. | *"Our dataset only provides full-time final metadata. Claiming a running score at minute 14 would be a hallucination. The system honestly reports score telemetry as unavailable."* |
| **"Score consequence unavailable"** | Match State Context | Live running score is unavailable; therefore, whether a goal is an "equalizer" or "go-ahead goal" cannot be determined. | **CORRECT BEHAVIOR** (Constraint Enforcement) | Prevents the language generation module from fabricating tactical consequence claims like 'takes the lead'. | *"Because live score state is unavailable, our match-state tracker deliberately suppresses score consequences to guarantee zero factual hallucinations."* |
| **"Player not in active camera view"** | Visual Grounding Card | The broadcast camera dynamically pans and zooms to follow the ball. A player involved in a pass may be outside the 720p frame. | **CORRECT BEHAVIOR** (Camera Coverage Boundary) | TrackLab tracking data accurately captures players within the broadcast camera frame. Off-screen players are honestly reported as not captured. | *"The visual verifier only confirms players physically visible in the broadcast frame. It honestly flags off-screen players rather than fabricating false visual bounding boxes."* |
| **"Partial visual grounding (teammates only)"** | Visual Grounding Card | Target player is outside the frame, but 2–4 teammates from the same club are actively tracked in the frame at that timestamp. | **CORRECT BEHAVIOR** (Contextual Grounding) | Confirms that the team's attacking phase is on screen even if the individual actor is just out of frame. Sets visual confidence to 0.50. | *"Partial grounding means teammates from the active club are verified on screen, confirming team-level possession even when the specific actor is outside the camera pan."* |
| **"Player 'Bailly' does not have biography"** | Retrieval Analysis & Error Taxonomy | Eric Bailly was transferred in 2016; the Wikipedia dump in `data/addinfo_retrieval/` contains 3,248 player articles but lacks `Eric_Bailly.txt`. | **CORRECT BEHAVIOR** (Corpus Boundary) | The system honestly logs `missing_corpus_entity` in the research error taxonomy rather than inventing biographical facts. | *"This represents a legitimate knowledge corpus boundary. 81.3% of events are covered in the corpus; uncovered players are honestly logged in our error taxonomy."* |
| **URL-Encoded Document Names (e.g., `%C3%A1`)** | Retrieval Analysis table | Filenames with accented Unicode characters (e.g., Sergio García) were read directly from disk without URL unquoting. | **COSMETIC ISSUE** (Resolved) | Purely cosmetic display issue where `urllib.parse.unquote` is applied during string rendering. | *"This was a text rendering artifact from raw filesystem stems, cleanly handled via UTF-8 decoding."* |

---

# PART 7 — VIDEO CONTENT VS STREAMLIT OVERLAYS

To ensure complete clarity during the faculty presentation, understand the distinction between video artifacts and Streamlit UI components:

```
+-----------------------------------------------------------------------------------+
| STREAMLIT RESEARCH DASHBOARD (Web Browser Interface at localhost:8501)            |
|                                                                                   |
|  +---------------------------------------+  +----------------------------------+  |
|  | VIDEO PLAYER (.mp4)                   |  | EVENT TIMELINE                   |  |
|  | (Source: outputs/demo-step3/0012.mp4) |  | [00'11"] CROSS - Bailly          |  |
|  | - Pre-rendered match broadcast        |  | [00'12"] CROSS - Blind           |  |
|  | - Audio speech track (Piper TTS)      |  | [00'13"] SHOT  - Blind  <ACTIVE> |  |
|  | - Burned-in bounding box tracker      |  | [00'14"] BLOCK - Slimani         |  |
|  |   (from upstream sn-gamestate)        |  | [00'14"] GOAL  - Blind           |  |
|  +---------------------------------------+  +----------------------------------+  |
|                                                                                   |
|  ======================= STREAMLIT COMPUTED INTELLIGENCE =======================  |
|  [Selected Event Identity]   [Temporal Sequence Flow]     [Match State]           |
|  [Explainable Importance]    [Retrieved Evidence List]    [Factual Verification]  |
|  [Visual Grounding Check]    [Confidence Scorer (41%)]    [Counterfactual Engine] |
|  [Adaptive Commentary Script]                             [Player MVP Analysis]   |
+-----------------------------------------------------------------------------------+
```

### Video Content vs Streamlit UI Overlay Breakdown
1. **Video Content (Pre-rendered MP4):**
   - Contains the 720p broadcast footage from SoccerNet.
   - Contains burned-in player bounding boxes and player name tags produced during data preparation (`demo-step3` / `demo-step4`).
   - Contains the background speech audio synthesized by Piper TTS.
   - **Does NOT** dynamically display the event importance score, confidence bar, counterfactual reasoning, or factual verifier checks.
2. **Streamlit UI Overlays (Dynamic Python Execution):**
   - The interactive event dropdown, match header, importance gauges, confidence breakdowns, retrieved Wikipedia snippets, and contributor tables are **rendered dynamically by Streamlit** in real time as the user selects events.
   - The visual timeline highlights the active event and displays the preceding build-up sequence.

---

# PART 8 — RETRIEVAL SYSTEM & EVALUATION METRICS

### Technical Foundations
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`. Maps text sequences to a 384-dimensional dense semantic vector space. Optimized for rapid inference and semantic similarity.
- **Normalization:** Embeddings are L2-normalized ($\|\mathbf{v}\|_2 = 1.0$).
- **Vector Index:** FAISS `IndexFlatIP` (Flat Inner Product). Because vectors are unit normalized, inner product equals **cosine similarity**:
  $$\text{Sim}_{\text{cos}}(\mathbf{q}, \mathbf{d}) = \mathbf{q} \cdot \mathbf{d} = \sum_{i=1}^{384} q_i d_i$$
- **Corpus Scale:** 3,248 Wikipedia player biography documents (`data/addinfo_retrieval/`), indexed into 3,248 FAISS vectors. Caching fingerprint stored in `cache/improved_rag/`.

### Context-Aware Scoring Formula
$$\text{Score}(d, e) = \left[ \text{Sim}_{\text{cos}}(\mathbf{q}_e, \mathbf{d}) + B_{\text{player}} + B_{\text{team}} + B_{\text{opp}} + B_{\text{temp}} + B_{\text{state}} \right] \times W_{\text{imp}}$$

- $B_{\text{player}}$: $+0.30$ if player matches document title exactly; $+0.10$ if mentioned in body text.
- $B_{\text{team}}$: $+0.05$ if active club is cited in document text.
- $B_{\text{opp}}$: $+0.03$ if opponent club is cited in document text.
- $B_{\text{temp}}$: $+0.08$ if prior sequence actors are cited in document text.
- $B_{\text{state}}$: $+0.05$ for high-stakes late-match derby context.
- $W_{\text{imp}}$: Importance weighting multiplier: $1.0 + (\text{Importance} \times 0.05)$.

### Evaluation Metrics Defined

#### 1. Precision@1 (P@1)
- **Definition:** Proportion of evaluated events where the #1 top-ranked retrieved document is the true ground-truth player biography.
- **Formula:**
  $$\text{Precision@1} = \frac{1}{|E|} \sum_{e \in E} \mathbb{I}(\text{rank}(\text{gold}_e) == 1)$$
- **Example:** If in 113 events, the correct player biography is ranked #1 in 23 events: $23 / 113 = 0.2035$ (20.35%).

#### 2. Precision@3 (P@3)
- **Definition:** Proportion of top-3 retrieved documents that are relevant. Evaluated as whether the gold document appears within the top 3:
  $$\text{Precision@3} = \frac{1}{|E|} \sum_{e \in E} \mathbb{I}(\text{rank}(\text{gold}_e) \le 3)$$
- **Value in A1:** $0.2478$ (24.78%).

#### 3. Recall@5 (R@5)
- **Definition:** Proportion of events where the ground-truth document is retrieved anywhere within the top 5 candidates.
- **Formula:**
  $$\text{Recall@5} = \frac{1}{|E|} \sum_{e \in E} \mathbb{I}(\text{rank}(\text{gold}_e) \le 5)$$
- **Value in A1:** $0.2743$ (27.43%).

#### 4. Mean Reciprocal Rank (MRR)
- **Definition:** The average of reciprocal ranks of the first relevant document across all queries.
- **Formula:**
  $$\text{MRR} = \frac{1}{|E|} \sum_{e \in E} \frac{1}{\text{rank}(\text{gold}_e)}$$
  *(If gold document is outside candidate pool, reciprocal rank is 0).*
- **Example:** If gold doc is rank 1, score = 1.0; if rank 2, score = 0.5; if rank 3, score = 0.333.
- **Value in A1:** $0.2299$.

---

# PART 9 — ABLATION STUDY & SCIENTIFIC FINDINGS

### Full Benchmark Ablation Table (113 Ground-Truth Covered Events)

| Condition | Configuration Description | Precision@1 | Precision@3 | Recall@5 | MRR | Factual Consistency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A0** | Semantic Baseline (`all-MiniLM-L6-v2` + FAISS) | 0.1416 | 0.2124 | 0.2212 | 0.1748 | N/A |
| **A1** | **+ Entity-Aware Retrieval (Name & Club Reranking)** | **0.2035** | **0.2478** | **0.2743** | **0.2299** | N/A |
| **A2** | + Temporal Context (Sequence query augmentation) | 0.1681 | 0.2035 | 0.2212 | 0.1903 | N/A |
| **A3** | + Match-State Reasoning (Score & minute strings) | 0.1681 | 0.2035 | 0.2212 | 0.1903 | N/A |
| **A4** | + Event Importance Weighting | 0.1681 | 0.2035 | 0.2212 | 0.1903 | N/A |
| **A5** | **+ Factual Consistency Verification** | **0.1681** | **0.2035** | **0.2212** | **0.1903** | **1.0000 (100%)** |

### Verified Relative Improvements (A0 → A1)
- **Precision@1 Relative Improvement:**
  $$\frac{0.2035 - 0.1416}{0.1416} = \frac{0.0619}{0.1416} = +43.71\%$$
- **MRR Relative Improvement:**
  $$\frac{0.2299 - 0.1748}{0.1748} = \frac{0.0551}{0.1748} = +31.52\%$$

### Crucial Research Insight: Dense Query Dilution (A2–A4)
- **What Happened:** When temporal sequences (e.g., `"CROSS -> SHOT -> GOAL"`) and match-state context strings were concatenated directly into the dense query vector in conditions A2–A4, Precision@1 decreased from **0.2035 to 0.1681**.
- **Why This Occurred:** Dense embedding encoders like `all-MiniLM-L6-v2` are trained on short, focused semantic sentences. Concatenating long, heterogeneous strings (player name + opponent name + sequence history + match minute) dilutes the embedding vector, shifting the query away from the target player's biographical profile.
- **Why This is a Valuable Research Finding:** Rather than concealing this result, it demonstrates an important scientific principle: **contextual reasoning signals should be applied as post-retrieval reranking factors or prompt conditioners, never as naive dense query concatenations**.

---

# PART 10 — EVENT IMPORTANCE SCORING

### Base Action Weights Table (`src/event_importance.py`)
```python
DEFAULT_ACTION_WEIGHTS = {
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
```

### Contextual Modifiers
1. **Late-Game Tension Boost ($M_{\text{time}}$):** If $\text{minute} \ge 75'$, $M_{\text{time}} = 1.0 + 0.15 = 1.15$. (Early game $\le 10'$ adds $+0.05$).
2. **Close-Match Margin Boost ($M_{\text{margin}}$):** If $|\text{score differential}| \le 1$, $M_{\text{margin}} = 1.0 + 0.10 = 1.10$.
3. **Sequence Climax Bonus ($B_{\text{climax}}$):** If event is culmination of build-up sequence, adds $+0.10$.

### Importance Tiers
- **Critical:** $\ge 0.85$ (Red `#ef4444`)
- **High:** $\ge 0.60$ (Amber `#f59e0b`)
- **Moderate:** $\ge 0.35$ (Sky Blue `#38bdf8`)
- **Routine:** $< 0.35$ (Gray `#6b7280`)

*Important Presentation Disclosure:* This score is an internal event-importance metric derived from telemetry to modulate commentary urgency. It is **not** an official FIFA/UEFA match rating.

---

# PART 11 — MULTIMODAL CONFIDENCE SCORING

### Formula & Weights
$$\text{Conf}(e) = (0.35 \times S_{\text{retrieval}}) + (0.25 \times S_{\text{entity}}) + (0.20 \times S_{\text{factual}}) + (0.20 \times S_{\text{visual}})$$

Where normalized component scores are:
1. **Normalized Retrieval ($S_{\text{retrieval}}$):**
   $$S_{\text{retrieval}} = \min\left(1.0, \max\left(0.0, \frac{\text{Sim}_{\text{cos}} - 0.2}{0.6}\right)\right)$$
2. **Normalized Entity ($S_{\text{entity}}$):**
   $$S_{\text{entity}} = \min\left(1.0, \max\left(0.0, \frac{\text{Entity Bonus}}{0.45}\right)\right)$$
3. **Factual Verification ($S_{\text{factual}}$):**
   - $1.00$ if status is `VERIFIED` without auto-correction.
   - $0.70$ if auto-corrected to alignment.
   - $0.20$ if uncorrected contradiction exists.
4. **Visual Grounding ($S_{\text{visual}}$):**
   - $1.00$ if player's exact bounding box is confirmed in frame.
   - $0.50$ if club teammates are visible in frame.
   - $0.00$ if outside active camera pan.

### Step-by-Step Calculation Example (~41% Case)
Assume an event with:
- Normalized retrieval signal: $0.25$
- Normalized entity signal: $0.11$
- Factual verification signal: $1.00$
- Visual grounding signal (teammates only): $0.50$

$$\text{Confidence} = (0.35 \times 0.25) + (0.25 \times 0.11) + (0.20 \times 1.00) + (0.20 \times 0.50)$$
$$\text{Confidence} = 0.0875 + 0.0275 + 0.2000 + 0.1000 = 0.4150 \rightarrow \mathbf{41\% \text{ or } 42\%}$$

### Why Heuristic, Not Calibrated Probability
- A **statistically calibrated probability** requires empirical validation where an event assigned 41% confidence is correct exactly 41% of the time across a held-out test distribution (e.g., via Platt scaling or isotonic regression).
- In our system, the confidence score is an **explainable heuristic composite** designed to expose transparent multi-signal health. We explicitly document it as uncalibrated to maintain complete scientific integrity.

---

# PART 12 — PLAYER INTELLIGENCE & MVP ENGINE

### Aggregation Mechanism (`src/player_intelligence.py`)
The engine iterates over all events in a match, grouping actions by player name:
$$\text{Total Impact Score}(p) = \sum_{e \in E_p} \text{Final Importance}(e)$$
$$\text{Average Impact Score}(p) = \frac{\text{Total Impact Score}(p)}{|E_p|}$$

- **Key Events Count:** Number of events where $\text{Importance} \ge 0.65$ (or `Critical`/`High` tier).
- **Impact Tiers:**
  - `High`: $\text{Total Impact} \ge 1.0$ or $\text{Key Events} \ge 2$.
  - `Moderate`: $\text{Total Impact} \ge 0.45$ or $\text{Key Events} \ge 1$.
  - `Routine`: All other players.

### Match 0012 Contributor Breakdown
- **Daley Blind (Manchester United):**
  - Cross (12.16s): Imp = $0.50$
  - Shot (13.36s): Imp = $0.87$
  - Goal (14.48s): Imp = $1.00$
  - **Sum = 2.37** $\rightarrow$ Designated **Match MVP / Top Contributor**.
- **Eric Bailly (Manchester United):** Cross (11.60s): Imp = $0.50$ $\rightarrow$ Rank 2.
- **Islam Slimani (Leicester):** Block (14.00s): Imp = $0.39$ $\rightarrow$ Rank 3.

*Why Legitimate:* Derived strictly from verified event telemetry; does not invent ungrounded metrics like synthetic expected goals (xG).

---

# PART 13 — FACTUAL CONSISTENCY VERIFICATION

### Deterministic Verifier Implementation (`src/factual_verifier.py`)
Rather than relying on an unconstrained, opaque LLM to check itself, the verifier executes 4 deterministic rule-based checks:
1. **Action Semantics Check:** Scans commentary for contradictory action keywords using `ACTION_CONTRADICTIONS` table:
   - On a `GOAL` event, flags forbidden words: `"save"`, `"denied"`, `"wide"`, `"misses"`, `"cleared"`.
   - On a `SHOT` event, flags `"goal"`, `"scored"`.
   - On a `YELLOW CARD` event, flags `"red card"`, `"sent off"`.
2. **Team Entity Check:** Validates that active team is not swapped with opponent (e.g., claiming Leicester scored when Manchester United scored).
3. **Player Attribution Check:** Verifies player name substring presence.
4. **Score Consequence Check:** Confirms that score claims are only permitted if live score data is present.

### Auto-Correction & Audit Trail
If a contradiction is detected:
- Flags status as `CONTRADICTION DETECTED`.
- Replaces contradictory text with factually verified template aligned with ground-truth telemetry.
- Logs exact contradiction rationale into `audit_summary`.
- Sets `was_corrected = True`.
- In evaluation benchmark (A5), achieves **100% consistency** across tested contradiction scenarios.

---

# PART 14 — VISUAL GROUNDING & BOUNDING BOX SCHEMA

### TrackLab SoccerNet Dataset (`data/from_video/players_in_frames_sn_gamestate.csv`)
- Contains tracking detections across 19 unique games (100% coverage of the 17 PBP evaluation matches).

### Bounding Box Coordinate Schema Resolution
- The raw CSV contains columns labeled `x1_720p`, `y1_720p`, `x2_720p`, `y2_720p`.
- **Empirical Code Audit Finding:** 101 out of 112 rows satisfy $x2 < x1$ when treated as coordinate corners!
- **Scientific Resolution:** The dataset uses TrackLab $[x, y, w, h]$ convention where column `x2_720p` stores **bounding-box width** and `y2_720p` stores **bounding-box height**.
- **Corrected Formula:**
  $$x_2 = x_1 + w = x_1 + \text{col}(x2\_720p)$$
  $$y_2 = y_1 + h = y_1 + \text{col}(y2\_720p)$$
  Guarantees valid geometric invariants: $x_1 < x_2$ and $y_1 < y_2$, clamped to 720p broadcast bounds ($1280 \times 720$).

---

# PART 15 — GROUNDED COUNTERFACTUAL REASONING

### Scientific Constraint: Non-Speculative Interventions
In sports analytics, counterfactual reasoning often degenerates into speculative fantasy (e.g., *"If the defender was taller, he would have blocked the ball"*). Our system enforces a strict research constraint: **Counterfactuals are restricted to structural graph interventions on observed telemetry**.

### Implemented Interventions (`src/counterfactual.py`)
1. **`EVENT_REMOVAL` (Excising Single Action from Chain):**
   - *Observed:* `CROSS → SHOT → BALL PLAYER BLOCK → GOAL`.
   - *Intervention:* Excise Event #5 (`GOAL`).
   - *Structural Consequence:* Sequence terminates at Event #4 (`BALL PLAYER BLOCK`), leaving ball in right top box with zero score consequences.
2. **`SEQUENCE_TRUNCATION` (Excising Build-up):**
   - Excises preceding build-up to evaluate whether the final action was an isolated individual effort or the culmination of team possession.

---

# PART 16 — COMMENTARY GENERATION & ADAPTIVE POLICY

### Speech Policy Registers
1. **Excited Climax:** Triggered for `GOAL`, `PENALTY`, `RED CARD`, or sequence climax events with importance $\ge 0.85$. Tempo: *Urgent / Climax*.
2. **Analytical Tactical:** Triggered for build-up sequences ($N \ge 2$ prior actions) or tactical events with importance $\ge 0.65$. Tempo: *Steady Broadcast*.
3. **Contextual Play-by-Play:** Triggered when high entity background is retrieved (entity score $\ge 0.30$). Tempo: *Measured / In-Depth*.
4. **Routine / Compressed:** Triggered for repeated consecutive actions (novelty gate) to avoid broadcast fatigue. Tempo: *Concise*.
5. **Routine / Concise:** Standard play-by-play description for midfield events. Tempo: *Concise Factual*.

### Truth on Broadcast Policy
*Every structured event receives commentary in the interactive dashboard*, with importance governing speech style, verbosity, and cadence rather than silently discarding events.

---

# PART 17 — RESEARCH CONTRIBUTIONS AUDIT

| Dimension | Implemented Research Contribution | Research Idea / Hypothesis | Future Work (Post-V2) |
| :--- | :--- | :--- | :--- |
| **Retrieval** | Entity-aware reranking (+43.7% P@1 gain); FAISS + `all-MiniLM-L6-v2`. | Contextual queries would beat entity-only queries. | Dense vector fine-tuning on domain-specific football corpus. |
| **Context** | Sequential temporal context window with sequence climax detection. | Deep temporal attention across 90 minutes. | Full match-long hierarchical temporal modeling. |
| **State** | Strict score availability checking; prevents score hallucinations. | Live score consequence reasoning. | Integration of live minute-by-minute scoreboard telemetry API. |
| **Structure** | Directed Acyclic Graph (DAG) with typed football transitions. | Complete probabilistic match graph. | Continuous spatial tracking graph from GPS/optical telemetry. |
| **Causal** | Non-speculative structural graph interventions (event removal). | Counterfactual game simulation. | Physical counterfactual simulation via reinforcement learning. |
| **Verification**| Deterministic rule-based factual verifier with auto-correction. | LLM self-verification. | Open-domain neural natural language inference (NLI) verifier. |
| **Vision** | TrackLab SoccerNet 720p bounding box schema resolution. | Real-time computer vision action spotting. | End-to-end video-to-commentary foundation model (e.g., VideoLLaMA). |
| **Confidence** | Transparent 4-signal composite heuristic score. | Bayesian posterior confidence. | Statistical probability calibration via Platt scaling / isotonic regression. |
| **Players** | Data-derived player impact aggregation and match MVP ranking. | Synthetic advanced metrics (fake xG). | True optical tracking-derived physical metrics (sprints, press resistance). |

---

# PART 18 — COMPLETE ERROR & SYSTEM AUDIT

### 1. Code Compilation & Test Status
- `python -m compileall -q src/`: **PASS (100% clean compilation, 0 syntax errors across all 21 files)**.
- `src/temporal_context.py`: **PASS**.
- `src/match_state.py`: **PASS**.
- `src/event_importance.py`: **PASS**.
- `src/context_retrieval.py`: **PASS**.
- `src/factual_verifier.py`: **PASS**.
- `src/visual_verifier.py`: **PASS**.
- `src/confidence_scorer.py`: **PASS**.
- `src/event_graph.py`: **PASS**.
- `src/counterfactual.py`: **PASS**.
- `src/commentary_policy.py`: **PASS**.
- `src/player_intelligence.py`: **PASS**.
- `src/pipeline_integration.py` (Match 0008): **PASS (All 10 modules verified on real data)**.

### 2. Empirical Error Taxonomy Analysis (`outputs/evaluation/error_analysis.json`)
Evaluated across 139 events / 17 matches:
- **`semantically_similar_distractor` (97 cases):** In semantic baseline (A0), high topical overlap causes articles about other players in the same league to outscore the target player.
- **`wrong_player` (94 cases):** Even with entity reranking, players sharing common surnames (e.g., Boateng) or players with minimal Wikipedia content are outranked.
- **`missing_match_score` (113 cases):** Play-by-play telemetry lacked live running scorelines; correctly flagged as unavailable by `MatchStateTracker`.
- **`missing_corpus_entity` (26 cases):** Player absent from Wikipedia dump (e.g., Eric Bailly in Match 0012).
- **`generation_contradiction` (6 cases):** Template generated contradictory verbs (e.g., "save" on a GOAL event); detected and auto-corrected by `FactualConsistencyVerifier`.
- **`wrong_team` (0 cases):** Club attribution errors completely eliminated.
- **`ambiguous_entity` (0 cases):** Zero unresolved multi-club ambiguities.

### 3. Comprehensive Issue Severity Audit
- **CRITICAL (Show-stopping errors):** **0 found.**
- **MEDIUM (Methodological limitations):**
  - Live running score telemetry absent in PBP dataset (handled correctly via abstention).
  - Dense query dilution in conditions A2–A4 (handled correctly as documented research finding).
- **MINOR (Edge-case behaviors):**
  - Wikipedia corpus covers 81.3% of players (26 missing entities documented).
  - TrackLab tracking CSV captures players within active 720p camera view; off-screen players honestly flagged.
- **COSMETIC (Visual / formatting details):**
  - URL-encoded strings in filesystem stems (e.g., `%C3%A1` in Sergio García) handled via unquoting.

---

# PART 19 — PRESENTATION SCRIPT & DEFENSE DELIVERY

### Master Presentation Flow (16 Steps)

```
1. Introduction & Title
2. Problem Statement in Sports Broadcasting
3. Original Baseline System (zaemon1251-hesty)
4. Motivation for Our Enhancements
5. System Architecture Overview
6. Live Demo: Match Intelligence Hub (Match 0012)
7. Deep-Dive: AI Reasoning Pipeline
8. Deep-Dive: Retrieval System & Ablation
9. Deep-Dive: Event Graph & Causal Analytics
10. Deep-Dive: Visual Grounding & Tracking
11. Deep-Dive: Factual Verification & Guardrails
12. Deep-Dive: Player Intelligence & MVP
13. Quantitative Benchmark Results
14. Honest Research Limitations
15. Future Work
16. Conclusion & Defense Questions
```

---

### Oral Delivery Scripts for Key Pages

#### PAGE 1 — MATCH INTELLIGENCE
- **30-Second Elevator Pitch:**  
  *"Match Intelligence is our live broadcast command hub. Here you see real match video synchronized with structured event telemetry. When I select this goal by Daley Blind in Match 0012, our system extracts the 4-event sequence, calculates event importance, retrieves Blind's verified Wikipedia biography, checks visual tracking, enforces factual consistency, and generates grounded broadcast commentary with a full audit trail."*
- **1-Minute Summary:**  
  *"On this screen, we unify the entire multimodal intelligence pipeline. At the top, the match header displays competition metadata while honestly reporting that live running scoreboards are unavailable in raw play-by-play data, avoiding score hallucinations. The event timeline lets us select any action. For Daley Blind's strike, our importance scorer rates it 0.87 (Critical) because it represents a sequence climax. The retrieval card shows Daley Blind's biography retrieved via FAISS and entity reranking with a score of 0.85. The factual verifier confirms player, team, and action alignment, while visual grounding verifies 720p frame presence. The resulting commentary is delivered in an Urgent Climax register, supported by a data-derived Player Intelligence table designating Blind as the match MVP."*
- **3-Minute Deep Dive:**  
  *"Let us walk through the architecture live on Match 0012 between Manchester United and Leicester. In traditional automated systems, this event would be treated as an isolated snippet, retrieving background text via unweighted semantic similarity and prompting an LLM that might invent a fictional scoreline.  
  In our system, every single phrase is grounded across four distinct modalities. Notice the Temporal Sequence card: it reconstructs the attacking wave—cross by Bailly, cross by Blind, blocked shot, and culmination into a goal. Because this is an attacking culmination, the system automatically detects a Sequence Climax, boosting event importance to 1.00.  
  Next, look at the Match State card. Live score is flagged as unavailable because our raw play-by-play telemetry only contains full-time metadata. Rather than hallucinating a mid-game score, our system deliberately abstains.  
  In the Retrieved Evidence card, our entity-aware retrieval engine ranks Daley Blind #1 with a final score of 0.8527, applying explicit player and club bonuses over 3,248 Wikipedia biographies.  
  Below, our verification row proves dual grounding: the deterministic factual verifier audits player, club, and action alignment with zero contradictions, while the TrackLab visual verifier queries 720p broadcast frames.  
  Our Multimodal Confidence Scorer aggregates these signals into an explainable 68% score. Notice the honest disclosure: this is an explainable heuristic, not a calibrated probability.  
  Finally, our Player Intelligence module aggregates cumulative event importances across the entire clip, establishing Daley Blind as the data-derived MVP with 2.37 total impact points. Every decision is transparent, explainable, and reproducible."*

---

#### PAGE 3 — RETRIEVAL ANALYSIS
- **30-Second Elevator Pitch:**  
  *"Retrieval Analysis presents a direct side-by-side comparison of our retrieval ablation modes: A0 Semantic Baseline, A1 Entity-Aware Reranking, and A5 Full Context. It proves that entity-aware reranking improves Precision@1 by +43.7%, while also demonstrating our negative finding: dense vector query dilution."*
- **1-Minute Summary:**  
  *"On this page, evaluators can inspect candidate rankings across 3,248 documents. Under A0, pure cosine similarity frequently returns distractors—such as Kevin-Prince Boateng instead of Daley Blind—because of overlapping football vocabulary. Under A1, our additive entity bonus (+0.30 for exact title match, +0.05 for club match) elevates the true player biography to rank 1. Under Column 3, we demonstrate what happens when temporal build-up strings are concatenated into the query: precision slightly decreases. This provides empirical proof that dense encoders suffer from query dilution and that contextual signals should be applied as post-retrieval reranking factors."*

---

#### PAGE 5 — RESEARCH EVALUATION
- **30-Second Elevator Pitch:**  
  *"Research Evaluation provides our formal quantitative benchmark across 17 matches and 139 events. Our entity-aware retrieval increases Precision@1 from 0.1416 to 0.2035 and MRR from 0.1748 to 0.2299, while our deterministic verifier guarantees 100% action consistency on tested contradictions, accompanied by a complete error taxonomy."*
- **1-Minute Summary:**  
  *"This benchmark evaluates all 113 events with ground-truth coverage across 17 matches. In condition A0, unweighted semantic search achieves P@1 of 0.1416. Introducing entity-aware reranking in A1 achieves a peak P@1 of 0.2035—a +43.7% relative improvement. Conditions A2 through A4 illustrate dense query dilution, which we document as an important scientific insight. Finally, Condition A5 introduces our factual verifier, achieving 100% consistency against action contradictions. Below, our error taxonomy transparently categorizes all 139 events, showing that missing live scores and semantic distractors represent the dominant data boundaries in the dataset."*

---

# PART 20 — 50+ FACULTY QUESTIONS & DEFENSE ANSWERS

### Group 1: NLP & RAG Architecture
1. **Q: Why did you use `all-MiniLM-L6-v2` instead of a larger model like BERT-large or OpenAI embeddings?**  
   - *Short:* For local, offline, deterministic inference with zero API costs.  
   - *Detailed:* `all-MiniLM-L6-v2` produces compact 384-dimensional embeddings optimized for cosine similarity. It embeds all 3,248 documents in minutes and searches in milliseconds via FAISS without external network latency or API fees.  
   - *What NOT to say:* "Because it was the easiest thing to copy."
2. **Q: What is the embedding dimension and distance metric?**  
   - *Short:* 384 dimensions; Cosine similarity via FAISS `IndexFlatIP`.  
   - *Detailed:* Vectors are L2-normalized upon extraction. Consequently, the inner product computed by `IndexFlatIP` is mathematically equivalent to cosine similarity.  
   - *What NOT to say:* "We used Euclidean L2 distance."
3. **Q: How large is your retrieval corpus?**  
   - *Short:* 3,248 Wikipedia player biography documents.  
   - *Detailed:* Sourced from `data/addinfo_retrieval/`, covering international and club footballers referenced across SoccerNet matches.  
   - *What NOT to say:* "The entire internet."
4. **Q: Why did adding temporal context in A2 reduce retrieval precision compared to A1?**  
   - *Short:* Dense vector query dilution.  
   - *Detailed:* Appending long build-up strings (e.g., `"CROSS -> SHOT -> GOAL"`) into a dense query dilutes the embedding representation away from the player's biographical entity profile. Context belongs in post-retrieval reranking, not query concatenation.  
   - *What NOT to say:* "A2 is broken."
5. **Q: What is an entity distractor in dense retrieval?**  
   - *Short:* An irrelevant document that shares high topical/lexical overlap with the query.  
   - *Detailed:* Articles about rival players in the same league share terms like "midfielder", "Bundesliga", and "goal", outscoring the true actor in unweighted search.  
   - *What NOT to say:* "A bug in FAISS."

---

### Group 2: Temporal Reasoning & Event Modeling
6. **Q: How is temporal context formally represented?**  
   - *Short:* As an `EventContext` dataclass with a sliding window of prior events.  
   - *Detailed:* It captures event duration, pitch minute, prior sequence path, and sequence climax flags over a sliding window of $N=3$ events.  
   - *What NOT to say:* "Just a string of past text."
7. **Q: What defines a Sequence Climax?**  
   - *Short:* A `GOAL` or `SHOT` culminating after at least one prior build-up action.  
   - *Detailed:* In `src/temporal_context.py:128`, an action is flagged as climax if `action in {'GOAL', 'SHOT'}` and `len(previous_events) >= 1`.  
   - *What NOT to say:* "Any exciting moment."
8. **Q: Why is temporal modeling essential for commentary?**  
   - *Short:* Commentary requires narrative continuity rather than isolated descriptions.  
   - *Detailed:* An isolated goal sounds generic; commentary that notes the preceding cross acknowledges the tactical sequence.  
   - *What NOT to say:* "It makes the prompt longer."

---

### Group 3: Match State & Factual Invariants
9. **Q: Why does the UI state "Score data unavailable"?**  
   - *Short:* Because live running scores are absent from the play-by-play telemetry.  
   - *Detailed:* The PBP JSONL logs contain event timestamps and actions but no running scoreboards. Full-time final results are preserved as historical reference only. Extrapolating a 90-minute score to mid-game would be a factual hallucination.  
   - *What NOT to say:* "The code failed to extract the score."
10. **Q: Why does the system report "Score consequence unavailable"?**  
    - *Short:* Consequence claims require verified live running score state.  
    - *Detailed:* Without running scores, claiming an action "extends the lead" or "equalizes" is ungrounded. The system deliberately abstains.  
    - *What NOT to say:* "We forgot to implement consequence reasoning."
11. **Q: How do you prevent mid-game score hallucinations?**  
    - *Short:* Strict gating via `state.is_score_available`.  
    - *Detailed:* `_score_phrase()` in `commentary_policy.py` checks `state.is_score_available` and returns an empty string if false.  
    - *What NOT to say:* "We instructed the LLM to be careful."

---

### Group 4: Event Importance
12. **Q: What is the range and formula of Event Importance?**  
    - *Short:* Range $[0.0, 1.0]$; Base action score modulated by time, margin, and climax.  
    - *Detailed:* $\text{Score} = (\text{Base} \times M_{\text{time}} \times M_{\text{margin}}) + B_{\text{climax}}$, clamped to $[0.0, 1.0]$.  
    - *What NOT to say:* "It is an official FIFA player rating."
13. **Q: What are the base weights for GOAL, SHOT, and PASS?**  
    - *Short:* GOAL = 1.00, SHOT = 0.70, PASS = 0.25.  
    - *Detailed:* Defined in `src/event_importance.py:DEFAULT_ACTION_WEIGHTS`.  
    - *What NOT to say:* "They were learned by a neural network."
14. **Q: How does late-game timing affect importance?**  
    - *Short:* Events in minute $\ge 75'$ receive a $+0.15$ multiplier boost.  
    - *Detailed:* Reflects heightened tactical tension in the closing 15 minutes of a match.  
    - *What NOT to say:* "It doubles the score."

---

### Group 5: Multimodal Confidence
15. **Q: What are the exact weights in the confidence formula?**  
    - *Short:* Retrieval 35%, Entity 25%, Factual 20%, Visual 20%.  
    - *Detailed:* Defined in `src/confidence_scorer.py:MultimodalConfidenceScorer`.  
    - *What NOT to say:* "The weights are learned through backpropagation."
16. **Q: Why does the UI say "Explainable heuristic - not a calibrated probability"?**  
    - *Short:* To maintain rigorous scientific honesty.  
    - *Detailed:* A calibrated probability requires empirical frequency alignment across held-out test data. This is a transparent composite heuristic.  
    - *What NOT to say:* "Because the math isn't real."
17. **Q: Why is confidence only ~41% in some examples?**  
    - *Short:* When visual grounding is partial (0.50) and semantic retrieval is moderate (~0.35).  
    - *Detailed:* In ungrounded or partially tracked events, confidence honestly drops to the 40–50% range.  
    - *What NOT to say:* "The system is broken."

---

### Group 6: Visual Grounding & Vision Verification
18. **Q: What dataset powers visual grounding?**  
    - *Short:* TrackLab SoccerNet player tracking CSV (`players_in_frames_sn_gamestate.csv`).  
    - *Detailed:* Contains multi-player tracking bounding boxes and jersey numbers across 19 matches.  
    - *What NOT to say:* "A live YOLOv8 model running in real time."
19. **Q: What was the bounding box coordinate bug and how was it solved?**  
    - *Short:* Resolved TrackLab $[x, y, w, h]$ schema where column `x2` stored width.  
    - *Detailed:* Converted to corner coordinates: $x_2 = x_1 + w$ and $y_2 = y_1 + h$, satisfying $x_1 < x_2$ and $y_1 < y_2$.  
    - *What NOT to say:* "We simply swapped x1 and x2."
20. **Q: What happens when the selected player is not visible on screen?**  
    - *Short:* Honestly flagged as "Player not in active camera view".  
    - *Detailed:* If teammates are visible, visual confidence is set to 0.50; if no players are visible, set to 0.00.  
    - *What NOT to say:* "The vision model hallucinates their position."

---

### Group 7: Factual Verification
21. **Q: How does the factual verifier work?**  
    - *Short:* Deterministic rule-based checks validating extracted claims against telemetry.  
    - *Detailed:* Audits player, team, opponent, and action keywords using contradiction dictionaries.  
    - *What NOT to say:* "We prompt GPT-4 to verify itself."
22. **Q: Can the factual verifier auto-correct errors?**  
    - *Short:* Yes, it replaces contradictory text with verified telemetry templates.  
    - *Detailed:* Logged into `VerificationReport` with an audit summary.  
    - *What NOT to say:* "It guesses an alternative."
23. **Q: Does the system guarantee zero hallucinations?**  
    - *Short:* It guarantees zero action contradictions on verified rule dimensions.  
    - *Detailed:* We avoid claiming universal zero hallucinations; it enforces deterministic guardrails on tested attributes.  
    - *What NOT to say:* "Yes, 100% zero hallucinations across the entire universe."

---

### Group 8: Event Graph & Counterfactual Reasoning
24. **Q: What is the Directed Event Graph?**  
    - *Short:* A DAG where nodes are events and edges are tactical transitions.  
    - *Detailed:* Models transitions like `POSSESSION_CONTINUITY`, `TURNOVER`, `CHANCE_CREATION`, and `CULMINATION`.  
    - *What NOT to say:* "A standard Knowledge Graph like Wikidata."
25. **Q: What is a counterfactual intervention in your system?**  
    - *Short:* A structural graph intervention (e.g., event removal) on observed telemetry.  
    - *Detailed:* Evaluates sequence consequences without inventing alternate physical reality.  
    - *What NOT to say:* "We simulate what the goalkeeper would have done."
26. **Q: Why is counterfactual reasoning non-speculative?**  
    - *Short:* To maintain scientific validity.  
    - *Detailed:* Speculating on physical outcomes produces hallucinations; structural graph removal reasons strictly about telemetry dependencies.  
    - *What NOT to say:* "Because we couldn't code physics."

---

### Group 9: Player Intelligence & MVP
27. **Q: How is the match MVP calculated?**  
    - *Short:* By summing event importance scores per player across the match telemetry.  
    - *Detailed:* Ranked by Total Impact Score, key moments count, and total event volume.  
    - *What NOT to say:* "Voted on by users."
28. **Q: Why is Daley Blind the MVP in Match 0012?**  
    - *Short:* He recorded a cross (0.50), a shot (0.87), and a goal (1.00) for a total impact of 2.37.  
    - *Detailed:* Contributed 2 key high-leverage moments, outscoring Bailly (0.50) and Slimani (0.39).  
    - *What NOT to say:* "Because Manchester United won."
29. **Q: Does Player Intelligence claim expected goals (xG)?**  
    - *Short:* No, it derives metrics strictly from observed event counts and importance.  
    - *Detailed:* We avoid ungrounded synthetic metrics like xG without shot-location spatial models.  
    - *What NOT to say:* "Yes, we implemented xG."

---

### Group 10: Evaluation & Benchmark Results
30. **Q: What is the P@1 improvement of A1 over A0?**  
    - *Short:* +43.7% relative improvement (0.1416 to 0.2035).  
    - *Detailed:* Demonstrates the power of additive entity and club reranking bonuses.  
    - *What NOT to say:* "100% accuracy."
31. **Q: What is the MRR improvement of A1 over A0?**  
    - *Short:* +31.5% relative improvement (0.1748 to 0.2299).  
    - *Detailed:* Confirms that true player biographies are ranked substantially higher in the candidate list.  
    - *What NOT to say:* "It doubled."
32. **Q: How many events and matches were evaluated?**  
    - *Short:* 139 events across 17 matches (113 with ground-truth coverage).  
    - *Detailed:* Coverage represents 81.3% of the dataset.  
    - *What NOT to say:* "Thousands of matches."

---

### Group 11: Error Taxonomy & Limitations
33. **Q: What are the main error categories in your taxonomy?**  
    - *Short:* `semantically_similar_distractor` (97), `wrong_player` (94), `missing_match_score` (113), `missing_corpus_entity` (26).  
    - *Detailed:* Extracted via automated audit across all 139 events in `outputs/evaluation/error_analysis.json`.  
    - *What NOT to say:* "We had zero errors."
34. **Q: What does `missing_corpus_entity` mean?**  
    - *Short:* A player in the event telemetry is missing from the 3,248 Wikipedia articles.  
    - *Detailed:* Example: Eric Bailly in Match 0012.  
    - *What NOT to say:* "A corrupted file."
35. **Q: What are the primary documented limitations of the project?**  
    - *Short:* No live running scores, uncalibrated confidence, template generation, and camera boundaries.  
    - *Detailed:* Documented transparently in Section 9 of `IMPROVED_README.md` and Page 6 of the Streamlit dashboard.  
    - *What NOT to say:* "There are no limitations."

---

# PART 21 — "WHY DID YOU DO THIS?" DEFENSE RATIONALE

1. **Why RAG?** Because sports commentary requires factual external background (biographies, records) that language models cannot reliably memorize without hallucinating.
2. **Why FAISS?** Industry-standard C++ vector indexing providing sub-millisecond similarity search over local disk caches.
3. **Why Sentence-Transformers?** Produces state-of-the-art dense semantic sentence embeddings locally without paid cloud API dependencies.
4. **Why `all-MiniLM-L6-v2`?** Optimal tradeoff between embedding quality (384-d), low latency (<15ms), and small memory footprint (<100MB).
5. **Why Entity-Aware Retrieval?** Because sports queries are entity-centric; unweighted dense semantic search frequently retrieves rival players who share similar league vocabulary.
6. **Why Temporal Context?** Because football events occur in structured sequences (`CROSS → SHOT → GOAL`); treating events in isolation destroys narrative momentum.
7. **Why Match-State Reasoning?** Commentary tone and urgency depend entirely on whether an action occurs in the 10th minute or the 89th minute of a tight game.
8. **Why Event Graphs?** Captures tactical possession continuity and chance creation dependencies across the pitch as a directed mathematical structure.
9. **Why Counterfactual Reasoning?** Allows broadcasters to analyze the structural importance of high-leverage moments without making speculative physical guesses.
10. **Why Factual Verification?** Hallucinations in automated sports broadcasting completely destroy audience trust; deterministic post-generation guardrails guarantee alignment.
11. **Why Visual Grounding?** Connects textual event telemetry to physical broadcast video, proving whether actors were actually captured on screen.
12. **Why Multimodal Confidence?** Provides operators with an explainable health score indicating whether an event's commentary is trustworthy.
13. **Why Player Intelligence?** Translates discrete event streams into cumulative player impact metrics and data-derived MVP honors.
14. **Why Streamlit?** Provides a highly responsive, Python-native research dashboard allowing immediate live auditing of every pipeline stage.
15. **Why not an LLM-only approach?** Unconstrained LLMs suffer from severe hallucinations, lack grounding in physical video tracking, and cannot guarantee deterministic reproducibility.
16. **Why not generate commentary directly from raw video pixels?** End-to-end video captioning models frequently miss player identities and fine-grained tactical rules; combining structured telemetry with video grounding achieves far higher factual precision.
17. **Why did retrieval perform worse after adding context (A2–A4)?** Dense vector query dilution: concatenating long heterogeneous strings dilutes the embedding vector away from the target player's entity profile.
18. **Why is confidence only ~41% in some cases?** Because the system honestly lowers its composite score when visual grounding is partial and semantic retrieval is modest.
19. **Why is live score unavailable?** Because raw play-by-play telemetry only contains final full-time scores; the system deliberately abstains rather than hallucinating mid-game scoreboards.
20. **Why is the retrieved player sometimes wrong?** In common surnames (e.g., "Boateng"), dense retrieval can favor an older player article with denser football text.
21. **Why is the system not claiming 100% accuracy?** Because scientific integrity demands reporting true retrieval precision (20.35%) alongside deterministic verification guardrails.

---

# PART 22 — TECHNICAL TERMINOLOGY GLOSSARY

| Term | Simple Meaning | Technical Definition | Why Used in Project | Example in Project | Oral Presentation Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NLP** | Natural Language Processing | Computational techniques for processing and generating human language | Powers text generation and document retrieval | Commentary generation | *"NLP transforms structured football events into natural spoken commentary."* |
| **Multimodal** | Using multiple data types | Combining text, video, tracking, and telemetry signals | Broadcast commentary requires both video and text | Aligning PBP logs with TrackLab video | *"Multimodal means our system fuses video tracking with text telemetry."* |
| **RAG** | Retrieval-Augmented Generation | Enhancing text generation by querying an external knowledge base | Enriches commentary with verified Wikipedia facts | Retrieving Daley Blind's article | *"RAG retrieves verified Wikipedia facts before generating broadcast commentary."* |
| **Dense Embedding** | Numerical representation of text | Continuous vector in $\mathbb{R}^{384}$ capturing semantic meaning | Enables vector similarity search in FAISS | `all-MiniLM-L6-v2` 384-d vectors | *"An embedding represents the meaning of text as a list of numbers."* |
| **FAISS** | Fast Vector Search Library | Facebook AI Similarity Search for dense vectors | Searches 3,248 documents in milliseconds | `faiss.IndexFlatIP` | *"FAISS is our high-speed vector search index."* |
| **Cosine Similarity** | Semantic closeness score | Inner product of normalized vectors: $\mathbf{u} \cdot \mathbf{v} \in [-1, 1]$ | Measures query-document topic alignment | Semantic score: 0.5527 | *"Cosine similarity measures how closely a query matches a document."* |
| **Entity-Aware Retrieval** | Name-prioritizing search | Retrieval incorporating additive bonuses for player and club entities | Prevents distractor articles from outranking target players | $+0.30$ bonus for Daley Blind | *"Entity-aware retrieval ensures the actual player in action ranks first."* |
| **Temporal Context** | Event sequence awareness | Modeling events as sequential chains within a sliding window | Captures attacking momentum | `CROSS -> SHOT -> GOAL` | *"Temporal context understands how an attack was built up over time."* |
| **Sequence Climax** | Attacking culmination | High-leverage event (`GOAL`/`SHOT`) concluding a build-up phase | Highlights decisive scoring attempts | Blind's goal in Match 0012 | *"Sequence climax identifies when an attacking build-up culminates in a shot."* |
| **Match State** | Game situation tracking | Tracking match minute, half, late-game tension, and scoreline | Ensures commentary reflects match drama | Min 14', First Half | *"Match state tracks the clock and score to set the right broadcast tone."* |
| **Event Graph (DAG)** | Flowchart of match events | Directed Acyclic Graph connecting events via tactical transitions | Models tactical possession progression | `CHANCE_CREATION` edge | *"Our event graph models tactical possession flow across the match."* |
| **Counterfactual Reasoning** | "What if" causal analysis | Non-speculative structural graph interventions (event removal) | Analyzes action criticality without hallucinating physics | Excising goal halts sequence | *"Counterfactual reasoning analyzes what happens structurally if an event is removed."* |
| **Factual Verification** | Hallucination guardrail | Deterministic rule-based checks validating claims against telemetry | Prevents action contradictions and team swaps | Checking player and action | *"Factual verification checks every generated phrase against real event telemetry."* |
| **Visual Grounding** | Video evidence alignment | Verifying player presence via TrackLab 720p bounding boxes | Proves whether player was visible on broadcast screen | Box: `[318, 383, 347, 477]` | *"Visual grounding verifies whether the player was physically visible in the camera frame."* |
| **Multimodal Confidence** | Composite reliability score | Explainable 4-signal heuristic ($0.0 - 1.0$) | Informs operators of system reliability | 68% Moderate Confidence | *"Confidence is an explainable composite score combining all 4 signal streams."* |
| **Precision@1** | Top-1 accuracy rate | Proportion of queries where rank-1 candidate is ground truth | Evaluates retrieval effectiveness | P@1 = 0.2035 in A1 | *"Precision@1 measures how often the correct biography is ranked number one."* |
| **MRR** | Mean Reciprocal Rank | Average of reciprocal ranks ($1/\text{rank}$) across queries | Evaluates ranking position of ground truth | MRR = 0.2299 in A1 | *"MRR measures how high up the correct document appears on average."* |
| **PBP** | Play-by-Play | Discrete structured chronological event log | Primary event input stream | `play-by-play-en.jsonl` | *"PBP stands for play-by-play, our structured match event stream."* |
| **Player Impact Score** | Cumulative contribution | Sum of event importance scores for a player across the match | Quantifies player contribution from telemetry | Blind's 2.37 impact score | *"Impact score sums up a player's contributions across all their actions."* |

---

# PART 23 — FINAL TRUTH CHECK: GREEN / YELLOW / RED

```
+-----------------------------------------------------------------------------------+
| GREEN: FULLY IMPLEMENTED & VERIFIED (State with 100% Confidence)                   |
| - Local Sentence-Transformers + FAISS retrieval over 3,248 documents              |
| - Multi-signal entity-aware reranking (+43.7% P@1 gain over baseline)             |
| - Sliding-window temporal context modeling & sequence climax detection            |
| - Match-state reasoning (pitch minute, half, late-game tension flag)               |
| - Deterministic 4-tier explainable event importance scorer                        |
| - Directed Event Graph (DAG) with typed football transitions                      |
| - Non-speculative structural counterfactual interventions (event removal)         |
| - Deterministic rule-based factual verifier with auto-correction                   |
| - TrackLab 720p bounding-box schema conversion ([x1, y1, x1+w, y1+h])             |
| - Player intelligence aggregation & data-derived match MVP selection              |
| - Complete 6-page interactive Streamlit research dashboard                        |
| - Multi-condition ablation benchmark (A0-A5) & empirical error taxonomy          |
+-----------------------------------------------------------------------------------+
| YELLOW: IMPLEMENTED HEURISTICS / KNOWN BOUNDARIES (Disclose Transparently)         |
| - Confidence score is an explainable heuristic, NOT a calibrated probability       |
| - Event importance is an internal metric, NOT an official FIFA/UEFA rating        |
| - Live running scores unavailable in PBP data (honestly reported as unavailable)  |
| - Dense query dilution in A2-A4 (documented as legitimate negative finding)       |
| - 81.3% corpus coverage (26 missing player articles documented in error taxonomy) |
| - Commentary generation uses deterministic templates for verification safety      |
+-----------------------------------------------------------------------------------+
| RED: NOT IMPLEMENTED / DO NOT CLAIM (Forbidden Claims)                            |
| - NEVER claim the system has "100% accuracy" or "zero errors"                     |
| - NEVER claim live minute-by-minute running scores are tracked                    |
| - NEVER claim confidence is a Bayesian calibrated probability                     |
| - NEVER claim counterfactuals simulate alternate physical realities (e.g. saves)  |
| - NEVER claim the video player dynamically renders Streamlit UI cards             |
| - NEVER claim advanced ungrounded metrics like expected goals (fake xG)           |
+-----------------------------------------------------------------------------------+
```

---

# PART 24 — ONE-PAGE PRE-PRESENTATION CHEAT SHEET

```
====================================================================================
               FOOTBALL COMMENTARY INTELLIGENCE: 60-SECOND DEFENSE CHEAT SHEET
====================================================================================
PROJECT: Context-Aware Multimodal Football Commentary Intelligence System
CORE PROBLEM: Automated commentary systems treat events in isolation, suffer from entity
  distractors, hallucinate ungrounded facts, and lack visual and match-state context.

BASE SYSTEM: Japanese open-source prototype (zaemon1251-hesty) using unweighted OpenAI
  embeddings and gpt-4o prompts with zero verification, no temporal context, and no tracking.

OUR KEY ENHANCEMENTS:
  1. Offline Vector RAG: Local all-MiniLM-L6-v2 + FAISS over 3,248 Wikipedia biographies.
  2. Entity Reranking: Player and club bonuses (+43.7% Precision@1 gain: 0.1416 -> 0.2035).
  3. Temporal Context: Sequence build-up modeling (PASS -> CROSS -> GOAL) + Climax detection.
  4. Match State: Minute tracking, late-game tension, and strict score availability gating.
  5. Event Graph: Directed DAG with tactical transitions (TURNOVER, CHANCE_CREATION, CULMINATION).
  6. Counterfactuals: Grounded structural graph interventions without speculative physics.
  7. Factual Verifier: Deterministic rule-based checks eliminating action contradictions.
  8. Visual Grounding: TrackLab 720p bounding box schema resolution (x2 = x1 + w, y2 = y1 + h).
  9. Confidence: Transparent 4-signal heuristic (Retrieval 35%, Entity 25%, Fact 20%, Vis 20%).
 10. Player MVP: Cumulative impact aggregation (Daley Blind MVP in Match 0012 with 2.37 score).

KEY BENCHMARK RESULTS (17 Matches, 139 Events, 113 Gold-Truth Covered):
  - A0 Semantic Baseline: P@1 = 0.1416 | MRR = 0.1748
  - A1 Entity-Aware Reranking: P@1 = 0.2035 (+43.7%) | MRR = 0.2299 (+31.5%)
  - A2-A4 Query Dilution: P@1 = 0.1681 (Proves context must be reranked, not concatenated!)
  - A5 Factual Verification: 100% action consistency on tested contradiction scenarios.

THREE GOLDEN DEFENSE RESPONSES:
  1. "Why is score unavailable?" -> "PBP telemetry only has full-time scores. We deliberately
     abstain rather than hallucinating mid-game scorelines."
  2. "Why is confidence 41%?" -> "It is an uncalibrated explainable heuristic that honestly drops
     when visual grounding is partial and semantic retrieval is moderate."
  3. "Why did A2 perform lower than A1?" -> "Dense query dilution: long context strings dilute
     short dense vector representations. It proves contextual signals belong in post-retrieval."
====================================================================================
```
