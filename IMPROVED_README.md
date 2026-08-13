# Improved Local Retrieval Prototype (src/improved_rag.py + src/improved_demo.py)

This is a small, self-contained addition to the original soccer-bg-commentary
repository. It does NOT modify, replace, or hide any part of the original
paper implementation in src/soccer_bg_commentary/. It sits beside it as an
independent local prototype.

## 1. What the original system does

The original pipeline (src/soccer_bg_commentary/) is:

event/action spotting -> additional information retrieval
-> OpenAI embeddings -> FAISS -> OpenAI LLM -> generated football commentary

Background documents (player biographies, ~3,248 files under
data/addinfo_retrieval/) are embedded with the OpenAI Embeddings API,
indexed with FAISS via LangChain, retrieved for a given play-by-play event,
and passed to an OpenAI LLM (via langchain-openai) which writes the final
commentary text.

## 2. Reproduction status (honest, as required)

Original repository successfully installed and original pipeline was
executed through the OpenAI embedding stage. Full original generation could
not be completed because the configured OpenAI API account has insufficient
quota (HTTP 429 / "no credits remaining"). No original results are faked or
invented anywhere in this repository.

Command attempted (for the record):

uv run python src/soccer_bg_commentary/main.py --game "england_epl/2015-2016/2015-08-23 - 15-30 West Brom 2 - 3 Chelsea" --half 2 --start 474 --end 504 --save_jsonl "outputs/demo-step2/13.jsonl" --save_srt "outputs/demo-step2/13.srt" --mode run

This failed at the OpenAI LLM call with an insufficient-quota error. The
retrieval side of the original pipeline (embeddings + FAISS) was reached and
is architecturally understood, but could not run to completion for the same
reason (OpenAI Embeddings API also requires quota).

## 3. What our improvement does

Because the OpenAI API is unavailable, this prototype replaces only the
embedding/retrieval/generation stages, entirely locally, with no paid API and
no large model download:

real play-by-play event -> entity extraction (player / team / opponent)
-> Sentence-Transformers (all-MiniLM-L6-v2) embeddings -> FAISS
-> entity-aware reranking -> local/deterministic commentary generation

### Models used (full names)

- sentence-transformers/all-MiniLM-L6-v2 -- a small (~80MB) local
  sentence embedding model from the Sentence-Transformers library, used to
  embed both the 3,248 background documents and the query built from the
  event/entities. Chosen because it was already confirmed working in this
  environment and requires no download beyond what was already fetched.
- FAISS (faiss-cpu), specifically IndexFlatIP (flat inner-product
  index) -- since embeddings are L2-normalized, inner product is equivalent
  to cosine similarity. Chosen for simplicity and because it matches what
  the original paper also uses for indexing (just with different
  embeddings).
- No generative LLM is used for commentary. See Section 5.

### Why this change was made

The original pipeline's retrieval and generation stages both depend on the
OpenAI API, which currently has no usable quota in this environment. Rather
than fake OpenAI output or silently skip the demo, this prototype
demonstrates the same architectural shape (embed -> retrieve -> generate)
using components that run entirely offline, so the retrieval logic and
overall pipeline can still be inspected and discussed.

## 4. Entity-aware retrieval

src/improved_rag.py provides ImprovedFootballRAG, which:

- Loads all files under data/addinfo_retrieval/ (unchanged from the
  original script's document set).
- Embeds them with all-MiniLM-L6-v2 and builds a FAISS IndexFlatIP.
- Caches the embeddings, FAISS index, and document/filename list to
  cache/improved_rag/, keyed to a fingerprint (file count + total byte
  size) of data/addinfo_retrieval/. If the folder hasn't changed, the
  ~10 minute embedding step is skipped entirely on subsequent runs.

src/improved_demo.py adds entity-aware reranking on top of that baseline
retrieval:

final_score = semantic_score + player_bonus + team_bonus + opponent_bonus

Where:

Bonus | Value | Condition
Player (filename match) | +0.30 | Player's name appears in the retrieved document's filename
Player (text match, fallback) | +0.10 | Player's name appears in the document body, but not the filename
Team | +0.05 | The event's team name appears in the document body
Opponent | +0.03 | The event's opponent team name appears in the document body

These weights are hand-picked for this demonstration only. They are not
the result of any tuning, grid search, or optimization process, and no claim
is made that they are scientifically optimal. They exist to show,
qualitatively, that folding in known entities changes the ranking in a
sensible direction compared to semantic similarity alone.

Reranking is done over a candidate pool of the top 30 semantically-similar
documents (cheap: one FAISS search, then a rerank pass), not a second FAISS
query per entity.

## 5. Commentary generation

Because no local generative LLM was already available in this environment
and downloading one (several GB) was explicitly out of scope for tonight's
deadline, commentary is generated with a local, deterministic,
template-based generator (generate_local_commentary in
src/improved_demo.py). It is not an LLM and does not call any API. It only
uses facts already present in the real event record (player, team,
opponent, action type) -- no invented facts.

This is a deliberate, explicit trade-off, not something concealed: the
retrieval improvement is the actual contribution being demonstrated; the
commentary line is a simple, honest way to show the retrieved context being
used in a final output.

## 6. How to run it

One-time setup (already done in this environment):

uv sync

Build (or reuse the cached) FAISS index and confirm retrieval works standalone:

uv run python src/improved_rag.py

Run the full demo (event -> entities -> baseline vs. entity-aware retrieval
-> local commentary -> saved report):

uv run python src/improved_demo.py

Outputs are written to outputs/improved-demo/:

- demo_report.txt -- full human-readable report (console output).
- commentary.json -- structured event, retrieval, scores, and commentary.

## 7. Example output (actually run, not fabricated)

Match: data/demo/pbp/0008/play-by-play-en.jsonl (Bayer Leverkusen vs
Bayern Munich). Event: real GOAL entry, player "Kresic", team "Bayer
Leverkusen".

Baseline (semantic similarity only) top-3:

1. Fabian_Giefer.txt   semantic_score=0.5275  final_score=0.5275
2. Simon_Rolfes.txt    semantic_score=0.5022  final_score=0.5022
3. Kevin_Kampl.txt     semantic_score=0.4981  final_score=0.4981

None of these is the actual scorer.

Entity-aware retrieval top-3:

1. Dario_Kresic.txt    semantic_score=0.4097  entity_bonus=0.35  final_score=0.7597
2. Fabian_Giefer.txt   semantic_score=0.5275  entity_bonus=0.08  final_score=0.6075
3. Kevin_Kampl.txt     semantic_score=0.4981  entity_bonus=0.08  final_score=0.5781

Dario_Kresic.txt -- the actual goalscorer's biography -- is correctly pulled
to #1, despite having a lower raw semantic score than the other
candidates. This is the concrete, observable effect of the entity bonus.

Generated commentary:

GOAL! Kresic scores for Bayer Leverkusen against Bayern Munich!

## 8. Comparison with the original approach

Stage | Original paper | Our local improvement
Embeddings | OpenAI Embeddings API (paid, requires quota) | sentence-transformers/all-MiniLM-L6-v2 (local, free)
Vector index | FAISS (via LangChain) | FAISS (IndexFlatIP, direct)
Retrieval ranking | Semantic similarity only | Semantic similarity + entity-aware bonus (player/team/opponent)
Generation | OpenAI LLM (via langchain-openai) | Local deterministic template generator
Requires API key | Yes | No
Requires internet at run time | Yes | No (after initial model download)

What changed: the embedding provider, the ranking function, and the
generation step. What did not change: the underlying documents
(data/addinfo_retrieval/), the general pipeline shape (retrieve then
generate), and the original repository's own implementation of that
pipeline, which remains untouched.

## 9. Evaluation

No formal retrieval evaluation (precision/recall, MRR, etc.) was performed.
Doing so honestly would require labeled ground truth pairing events with the
"correct" background document, which was not available within tonight's
scope. The single worked example above is illustrative, not a benchmark. No
accuracy numbers, evaluation metrics, or improvement percentages are claimed
anywhere in this project.

## 10. Limitations

- Only one worked example (Bayer Leverkusen vs Bayern Munich, GOAL event) is
  demonstrated end-to-end. The entity-aware logic has not been validated
  across many events or matches.
- Entity matching is simple substring matching on player/team names, not a
  proper named-entity-recognition or coreference system. It can miss
  aliases, nicknames, or transliteration differences.
- Team/opponent bonuses rely on the team name literally appearing in a
  player biography's text, which is common but not guaranteed for every
  player and every club.
- The commentary generator is template-based, not a language model. It
  produces a single, fairly rigid sentence structure per action type.
- No quantitative retrieval evaluation was performed (see Section 9).
- This prototype only demonstrates the retrieval + generation portion of
  the pipeline; it does not re-implement event/action spotting, which is
  taken as given from the existing data/demo/pbp/ files.
