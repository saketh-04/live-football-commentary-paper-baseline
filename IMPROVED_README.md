# Context-Aware Multimodal Football Commentary Intelligence System

An integrated, context-aware multimodal intelligence framework for automated sports broadcasting.

---

## 1. Research Overview & Problem Formulation

Sports commentary generation is an intrinsically multimodal and temporal challenge. Traditional automated systems treat events in isolation, retrieving background player biographies using unweighted dense semantic similarity or ungrounded generative language models. In live sports broadcasting, this paradigm suffers from four fundamental failures:

1. **Temporal Disconnection**: Football events unfold as cohesive, multi-actor attacking and defensive sequences (e.g., `PASS → CROSS → SHOT → GOAL`). Treating events in isolation discards narrative momentum, build-up structure, and culmination dynamics.
2. **Entity & Distractor Dilution**: Semantic similarity models frequently suffer from lexical distractors — retrieving rival players or historical articles with spurious textual overlap rather than the true in-action actor.
3. **Absence of Tactical & Visual Grounding**: Systems lack spatial awareness of player presence on screen and lack causal counterfactual understanding of how an individual action altered the match state.
4. **Factual Hallucinations**: Generative models frequently invent live scores, pitch locations, or contradictory actions (e.g., declaring a "brilliant save" on a recorded `GOAL` event).

This project designs, implements, and evaluates a **Context-Aware Multimodal Football Commentary Intelligence System** that grounds every broadcast phrase in real play-by-play telemetry, verified visual tracking, match-state constraints, and explainable multi-signal confidence.

---

## 2. Multimodal System Architecture

```
                       REAL FOOTBALL BROADCAST VIDEO
                                    ↓
                         PLAY-BY-PLAY EVENT STREAM
                                    ↓
                       TEMPORAL CONTEXT WINDOW (V1)
                          (src/temporal_context.py)
                                    ↓
                        ENTITY RESOLUTION ENGINE
                                    ↓
                        MATCH-STATE TRACKER (V1)
                            (src/match_state.py)
                                    ↓
                         DIRECTED EVENT GRAPH (V2)
                            (src/event_graph.py)
                                    ↓
                       EXPLAINABLE EVENT IMPORTANCE (V1)
                         (src/event_importance.py)
                                    ↓
                    CONTEXT-AWARE / ENTITY-AWARE RAG (V1)
                        (src/context_retrieval.py)
                                    ↓
                     COUNTERFACTUAL REASONING ENGINE (V2)
                          (src/counterfactual.py)
                                    ↓
                      ADAPTIVE COMMENTARY POLICY (V2)
                        (src/commentary_policy.py)
                                    ↓
                         GROUNDED COMMENTARY DRAFT
                                    ↓
                   FACTUAL CONSISTENCY VERIFICATION (V1)
                         (src/factual_verifier.py)
                                    ↓
                   VISUAL GROUNDING VERIFICATION (V2)
                          (src/visual_verifier.py)
                                    ↓
                    EXPLAINABLE CONFIDENCE ESTIMATION (V2)
                         (src/confidence_scorer.py)
                                    ↓
                         FINAL VERIFIED COMMENTARY
                                    ↓
                     RESEARCH & BROADCAST DASHBOARD
                          (src/streamlit_app.py)
```

---

## 3. Core System Modules (V1 & V2)

### 3.1. Temporal Match Context (`src/temporal_context.py`)
- Maintains a sliding temporal window of prior events in the sequence.
- Extracts sequence signatures (e.g., `CROSS → SHOT → GOAL`).
- Identifies **Sequence Climax** events where an attacking sequence culminates in a high-leverage action.

### 3.2. Match-State Reasoning (`src/match_state.py`)
- Tracks match minute, half, league, home/away clubs, and score differential.
- **Strict Factual Invariant**: Because live running scores are unavailable in raw play-by-play data, the module reports `"Score information unavailable"` and prevents the generation of score-dependent claims like *"takes the lead"* or *"equalizes"*. Full-time results are preserved as historical context only.

### 3.3. Directed Event Graph (`src/event_graph.py`)
- Constructs a directed acyclic graph (DAG) representing relationships between events in a sequence.
- Semantic edge types include:
  - `BUILD_UP`: Possession-retaining sequence continuation.
  - `TURNOVER`: Possession loss or interception between competing teams.
  - `CREATES_CHANCE`: Action creating a direct shot or penalty opportunity.
  - `CULMINATION`: High-leverage climax of an attacking phase.
  - `SEQUENCE_CONTINUATION`: Generic consecutive event transition.

### 3.4. Explainable Event Importance (`src/event_importance.py`)
- Computes an explainable importance score ($0.0$ to $1.0$) based on base action criticality, late-game tension ($+0.15$ if $\ge 75'$), score margin, and sequence climax bonus ($+0.10$).
- Categorizes events into four tiers: **Critical** ($\ge 0.85$), **High** ($\ge 0.65$), **Moderate** ($\ge 0.45$), and **Routine** ($< 0.45$).

### 3.5. Context-Aware / Entity-Aware Retrieval (`src/context_retrieval.py`)
- Operates over 3,248 Wikipedia player biographies using FAISS cosine similarity on `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- Reranks candidates using explicit additive bonuses:
  $$\text{Score}(d, e) = \left[ \text{Sim}_{\text{cos}}(\mathbf{q}_e, \mathbf{d}) + B_{\text{player}} + B_{\text{team}} + B_{\text{opp}} + B_{\text{temp}} + B_{\text{state}} \right] \times W_{\text{imp}}$$
- Exposes full transparent breakdown: Semantic score, Player bonus, Team bonus, Opponent bonus, Temporal bonus, and Match-state bonus.

### 3.6. Counterfactual Reasoning Engine (`src/counterfactual.py`)
- Implements non-speculative structural graph interventions:
  - `EVENT_REMOVAL`: Evaluates downstream consequences if the current action had not occurred.
  - `SEQUENCE_TRUNCATION`: Truncates build-up to isolate standalone event impact.
- Strictly non-speculative: never hypothesizes alternate real-world results; only reasons over structural dependencies within the observed telemetry.

### 3.7. Adaptive Commentary Policy (`src/commentary_policy.py`)
- Implements a genuine **Speech Gate** deciding *when* to broadcast vs remain *silent*:
  - **Always Speak**: `GOAL`, `PENALTY`, `RED CARD`.
  - **Importance Thresholds**: `CROSS` requires $\ge 0.55$, `PASS`/`DRIVE` requires $\ge 0.65$.
  - **Novelty Gate**: Suppresses immediate duplicate consecutive routine actions to avoid broadcast fatigue.
  - **Confidence Floor**: Suppresses low-importance events with confidence $< 0.35$.
- Dynamic speech registers: *Excited Climax*, *Analytical Tactical*, *Historical Contextual*, *Pacy Play-by-Play*.

### 3.8. Factual Consistency Verification (`src/factual_verifier.py`)
- Rule-based deterministic validator checking:
  - Subject player and club attribution against telemetry.
  - Action semantic alignment (e.g., flagging "save" on a `GOAL` event).
  - Score consequence claims against actual match-state availability.
- Automatically audits contradictions and regenerates verified commentary.

### 3.9. Visual Grounding Verification (`src/visual_verifier.py`)
- Aligns play-by-play events with TrackLab SoccerNet player tracking frames (`data/from_video/players_in_frames_sn_gamestate.csv`).
- **Coordinate Schema Resolution**: Resolves TrackLab $[x_1, y_1, w, h]$ convention where column `x2_720p` stores width and `y2_720p` stores height, correctly computing corner coordinates:
  $$x_2 = x_1 + w, \quad y_2 = y_1 + h \quad (x_1 < x_2, \; y_1 < y_2)$$
- Distinguishes **Confirmed Visual Grounding**, **Teammates Visible in Frame**, and **Camera Coverage Boundary**.

### 3.10. Multimodal Confidence Scorer (`src/confidence_scorer.py`)
- Computes an explainable, transparent composite heuristic:
  $$\text{Conf}(e) = 0.35 \cdot S_{\text{retrieval}} + 0.25 \cdot S_{\text{entity}} + 0.20 \cdot S_{\text{factual}} + 0.20 \cdot S_{\text{visual}}$$
- Explicitly documented as an **explainable heuristic**, not a statistically calibrated probability.

---

## 4. Empirical Modality Coverage Analysis

Empirical audit of the dataset across all 25 clips in `data/demo/sample_metadata.csv` (17 distinct play-by-play matches):

| Match ID | Match Title | Half | PBP Events | PBP | Video | Tracking | GT Biographies | Multimodal Ready |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0008** | Bayern Munich 3 - 0 Bayer Leverkusen | 2 | 3 | YES | YES | YES (6 tracked) | YES | **YES** |
| **0010** | Juventus 3 - 2 Fiorentina | 1 | 4 | YES | YES | YES (4 tracked) | YES | **YES** |
| **0011** | Shakhtar Donetsk 4 - 0 Malmo FF | 1 | 5 | YES | YES | YES (4 tracked) | YES | **YES** |
| **0012** | Manchester United 4 - 1 Leicester | 1 | 5 | YES | YES | YES (11 tracked) | YES | **YES** |
| **0013** | West Brom 2 - 3 Chelsea | 2 | 8 | YES | YES | YES (2 tracked) | YES | **YES** |
| **0014** | West Brom 2 - 3 Chelsea | 1 | 5 | YES | YES | YES (2 tracked) | YES | **YES** |
| **0015** | Hamburger SV 2 - 5 Dortmund | 1 | 13 | YES | YES | YES (3 tracked) | YES | **YES** |
| **0016** | Bayern Munich 3 - 0 Bayer Leverkusen | 1 | 7 | YES | YES | YES (17 tracked) | YES | **YES** |
| **0018** | Bayern Munich 0 - 1 FC Augsburg | 2 | 14 | YES | YES | YES (11 tracked) | YES | **YES** |
| **0020** | Manchester United 1 - 1 Arsenal | 1 | 8 | YES | YES | YES (2 tracked) | YES | **YES** |
| **0022** | Bayern Munich 1 - 2 Real Madrid | 2 | 11 | YES | YES | YES (12 tracked) | YES | **YES** |
| **0023** | Norwich 1 - 2 Chelsea | 1 | 10 | YES | YES | YES (4 tracked) | YES | **YES** |
| **0024** | RB Leipzig 1 - 0 Dortmund | 1 | 9 | YES | YES | YES (4 tracked) | YES | **YES** |
| **0025** | RB Leipzig 1 - 0 Dortmund | 2 | 11 | YES | YES | YES (5 tracked) | YES | **YES** |
| **0026** | Chelsea 3 - 0 Leicester | 2 | 4 | YES | YES | YES (2 tracked) | YES | **YES** |
| **0027** | Real Madrid 3 - 0 Atl. Madrid | 2 | 12 | YES | YES | YES (5 tracked) | YES | **YES** |
| **0028** | Real Madrid 1 - 0 Granada CF | 2 | 10 | YES | YES | YES (2 tracked) | YES | **YES** |

### Modality Summary
- **Total Play-by-Play Matches**: 17 matches (139 total events).
- **Video Coverage**: 20 video clips available in `outputs/demo-step3/`.
- **Tracking Coverage**: 100% of the 17 PBP evaluation matches (19 unique matches) have player tracking frames in `data/from_video/players_in_frames_sn_gamestate.csv`.
- **Fully Multimodal Evaluation Usable**: **17 matches (139 events)** have concurrent PBP, video, player tracking, and gold retrieval documents.

---

## 5. Formal Evaluation & Ablation Benchmark

Evaluated over **113 ground-truth-covered events** across the 17 matches:

| Condition | Description | Precision@1 | Precision@3 | Recall@5 | MRR | Factual Consistency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A0** | Semantic Baseline (`all-MiniLM-L6-v2` + FAISS) | 0.1416 | 0.2124 | 0.2212 | 0.1748 | N/A |
| **A1** | **+ Entity-Aware Retrieval** | **0.2035** | **0.2478** | **0.2743** | **0.2299** | N/A |
| **A2** | + Temporal Context | 0.1681 | 0.2035 | 0.2212 | 0.1903 | N/A |
| **A3** | + Match-State Reasoning | 0.1681 | 0.2035 | 0.2212 | 0.1903 | N/A |
| **A4** | + Event Importance Weighting | 0.1681 | 0.2035 | 0.2212 | 0.1903 | N/A |
| **A5** | **+ Factual Consistency Verification** | 0.1681 | 0.2035 | 0.2212 | 0.1903 | **1.0000 (100%)** |

### Key Scientific Insights
1. **Entity-Aware Reranking ($A_1$) is the dominant retrieval improvement**:
   - P@1 increases by **+43.7%** (0.1416 → 0.2035).
   - MRR increases by **+31.5%** (0.1748 → 0.2299).
2. **Dense Vector Query Dilution ($A_2 - A_4$)**:
   - Concatenating long temporal sequences and match-state context strings directly into dense queries slightly dilutes embedding vectors compared to concise entity targeting. Contextual reasoning should therefore be applied as **post-retrieval reranking factors**, not naive query concatenation.
3. **Factual Verification ($A_5$) Eliminates Contradictions**:
   - Deterministic rule-based checks completely prevent hallucinated action attributions.

---

## 6. End-to-End Pipeline Integration Results (Match 0008)

Full integration test on real match data (Bayern Munich vs Bayer Leverkusen, clip 0008):

```
============================================================
  INTEGRATION TEST SUMMARY (Match 0008)
============================================================
  Total Events Processed:         3
  Events with SPEAK Decision:     2 (SHOT, GOAL)
  Events with SILENT Decision:    1 (CROSS: importance 0.495 < threshold 0.55)
  Events Verified Factual:        2
  Events Auto-Corrected:          1 (SHOT: removed spurious 'goal' verb)
  Events Visually Grounded:       2 (Kresic D. #25 confirmed at 871s, bbox: [318, 383, 347, 477])
  Events with Teammates Visible:  1 (Boateng not in frame at 871s; 3 teammates visible)
  Average Composite Confidence:   0.6451 (Moderate Confidence)
  Status:                         PASS (All 10 core modules verified)
============================================================
```

---

## 7. How to Run & Reproduce

All commands use `uv run` to guarantee isolated execution within the project environment.

### 1. Run Comprehensive Test Suite (All 10 Core Modules)
```bash
uv run python src/run_all_tests.py
```
*Expected: `ALL 10 CORE MODULE TESTS PASSED PERFECTLY!`*

### 2. Run End-to-End Pipeline Integration Test
```bash
uv run python src/pipeline_integration.py
```
*Executes the complete 10-stage pipeline on Match 0008 and saves `outputs/integration/integration_test_result.json`.*

### 3. Generate Empirical Modality Coverage Matrix
```bash
uv run python src/generate_modality_coverage.py
```
*Audits all 25 clips and saves `outputs/research/modality_coverage.csv`.*

### 4. Run Evaluation, Ablation & Error Analysis
```bash
uv run python src/evaluation.py
uv run python src/ablation.py
uv run python src/error_analysis.py
```

### 5. Launch the Streamlit Research Dashboard
```bash
uv run streamlit run src/streamlit_app.py
```
Navigate to **`http://localhost:8501`** to access the 6-page interactive research dashboard.

---

## 8. Human Evaluation Protocol

The evaluation protocol is defined in `outputs/research/human_evaluation_template.csv`:
- **Dimensions**: Factual Correctness, Relevance, Context Awareness, Fluency, Naturalness, Timing, Overall Quality.
- **Rating Scale**: 1 (Very Poor) to 5 (Excellent).
- **Conditions Compared**: $A_0$ (Baseline), $A_1$ (Entity-Aware), $A_5$ (Full Pipeline with Verification).

---

## 9. Documented Limitations & Scientific Integrity

| Limitation | Technical Detail |
| :--- | :--- |
| **No Live Running Scores** | Raw play-by-play telemetry contains only full-time final scores. The system deliberately abstains from claiming running score changes. |
| **Broadcast Frame Boundaries** | TrackLab tracking data accurately captures players within the 720p broadcast camera view; off-screen players are honestly reported as not captured. |
| **Uncalibrated Confidence** | Multimodal confidence is an explainable heuristic, not a calibrated Bayesian probability. Platt scaling remains future work. |
| **Deterministic Generation** | Rule-based generation ensures strict verification reproducibility; neural LLM integration with post-hoc verification is future work. |
