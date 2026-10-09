# Independent knowledge graph

Kennisgraaf reads active indexed source passages directly. It never reads or
calls Feitenoverzicht or entity extraction. The shared source viewer and local
Ollama transport remain available. Graph data uses separate `kg_` SQLite tables
in each investigation and travels with ordinary `.bosuil` exports/imports.

## Research basis

[Extract, Define, Canonicalize (Zhang and Soh, EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.548/)
separates extraction, schema definitions and canonicalisation. Its discussion
of redundant relation labels and large schemas informs our compact, selectable
vocabulary with directional definitions. Our adaptation uses exact source-local
name resolution and explicit manual identity merges. It does not reproduce
their open-schema pipeline or trained schema retriever.

[Automated Knowledge Graph Construction using Large Language Models and Sentence Complexity Modelling (Anuyah et al., EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.783/)
informs atomic extraction and example-based prompting. Table 9 reports strong
Llama 3 8B results on a Boolean relationship task with 100 sentences per model;
Table 7 shows much weaker end-to-end results. That historical result cannot rank
current Qwen/Gemma releases or predict Dutch investigation quality. We keep
original source passages and request atomic claims, without LLM sentence
rewriting or an extended chain of thought. This is an adaptation, not a
reproduction of the paper's full method.

[Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
constrain response shape; evidence validation still needs separate checks.
[Thinking controls](https://docs.ollama.com/capabilities/thinking) vary by model.
Supported controls are checked through model metadata when available.

The broader vocabulary also draws on the distinction between entities and
literal values in [RDF Concepts](https://www.w3.org/TR/rdf11-concepts/#section-Graph-Literal)
and temporal relations in [OWL-Time](https://www.w3.org/TR/owl-time/).
These are design references; the stored graph is not an RDF/OWL implementation.

## Pipeline

1. Select indexed documents and a generation model. The tab starts with
   `OLLAMA_LLM_MODEL`, falling back to the editable name `qwen3.6:latest`.
   Refresh models lists installed models. Nothing is downloaded automatically.
2. Process adjacent passages in windows of at most 4,000 source characters,
   with bounded overlap. Files and identified attachments stay separate.
3. English instructions, definitions, positive/negative examples and a JSON
   schema request up to 24 concrete, atomic relationships per window. Compact
   field names and numbered sentence/line IDs avoid repeating evidence quotes
   in the generated output. The app reconstructs contiguous quotes directly
   from the original source, preserving whitespace. Evidence IDs must be
   consecutive and within one source fragment; models never generate offsets. Cover
   time, location, facts/properties, processes/effects and people/organizations
   without prioritizing human actions. Keep
   original names and exact source-language quotes. Preserve polarity,
   attribution and explicit time wording.
4. Deterministically check fields, relationship/type vocabulary, passage IDs,
   exact quotes, exact endpoint mentions, and qualifiers within the quote.
5. By default, a second call to the selected model checks semantics, direction,
   ambiguity and qualifications. An accurately represented denial/allegation
   can be supported as a qualified source claim without proving the event.
   Parallel arrays return a brief source-based reason (up to 120 characters) and
   a boolean for each candidate. Reasons precede decisions to retain deliberate
   semantic checking while avoiding repeated indices/keys. Invalid or incomplete verdicts prevent publication;
   successful extraction remains cached for another verification attempt.
   This model check is not independent ground truth.
6. Save complete batches atomically as suggestions. Human review is explicit.

Defaults: 8,192 context tokens, 4,096 output tokens per call, temperature zero,
thinking off, 180 seconds per request, verification on. Settings persist per
investigation. Conservative byte-based prompt estimation includes output
headroom; model-advertised context limits are checked when available. This is
not the model's tokenizer and does not guarantee fit for every model. An untouched
saved 2,048-token default upgrades once to 4,096; other saved budgets remain.
An intentional 2,048-token choice made after this upgrade also remains. Older
settings cannot distinguish an explicit 2,048 choice from the previous default;
both upgrade once.

## Review and identity

Exact normalised names/types resolve within one source. Names across different
documents/attachments remain separate until explicitly merged. Namesakes within
one source can still be conflated; mention-level disambiguation and split-node
operations are outside this initial implementation.
Literal identities preserve casing, including case-sensitive units such as
`mW` versus `MW`; they are not lowercased like entity aliases.

Accept, reject or reset claims; correct relationship/qualifiers; rename/retype
nodes; merge identities explicitly. Source aliases and reviews survive reruns.
Claim/node edits reset affected accepted claims to suggested and mark edited
verification as `user_edited`. Source quotes remain unchanged. Acceptance is
review, not proof. Persistent merges currently have no undo.

Each claim retains the quote, chunk ID, source hash, model, original run,
verification state and review note. Contradictory and multiple-source claims
coexist. Explicitly described events, objects, processes, conditions and concepts
can be endpoints even without a proper name. Invented event labels and topic-only
concepts are excluded. Dates, times, periods, durations, quantities (with units or
currencies) and states can be literal endpoints, copied verbatim from the evidence.
Property/measurement names are retained on claims, so equal height and width values
remain distinct facts. For example, `Bridge C --has_measurement (height)--> 15 m`.
The schema covers occurrence/start/end, explicit order/overlap, event location,
spatial containment/direction, classification, attributes, quantities, durations
and explicit dependencies/causes/effects. Chronology alone never establishes a
causal relation, and dates/durations are not guessed or calculated.

## Incremental processing and diagnostics

Extraction and verification are independently cached by input IDs/text/hashes,
source identity, protocol version, model name/metadata, thinking mode and selected
relations. Verification also includes the exact candidate list. Output budget,
context size and request timeout are operational limits and do not invalidate
completed work. Changing verification reuses extraction while applying the chosen
check. Model/source/vocabulary changes invalidate the corresponding cache.
Compatible completed v2 windows are adopted using their original cache signatures,
preserving existing suggestions, corrections and review decisions. Their prompts
are not rerun unless **Reprocess cached windows** is selected.
Failed verification retries reuse extraction. Failed/cancelled recovery sections
resume individually, without repeating completed siblings or a truncated parent.
Reprocess bypasses cache and retains review decisions/corrections. Different
settings can add suggestions; older claims remain for review. Duplicate claims
retain original provenance, while per-run raw outputs support comparisons.
The compact protocol uses a new prompt/cache version. Investigations saved
with all of the original default relations automatically gain the expanded
defaults; custom selections are preserved. Empty new property fields do not
duplicate otherwise identical pre-upgrade claims. Stored property annotations
travel in the same database/archive as the claims.

Stopping discards incomplete streamed JSON and retains completed source sections
and cached extraction/verification stages.
Cancellation and wall-clock limits are checked between stream updates. During
network silence the HTTP inactivity timeout applies, so the wall-clock check
is cooperative rather than an exact hard deadline. Closing requests cancellation
and keeps the window open until the worker stops; close again afterward.
Switching investigations cancels the old worker and ignores its late UI signals.
The worker retains its original database, preventing writes into the new case.

Malformed/disconnected output is discarded. Token truncation, the 24-claim cap,
or an estimated context overflow trigger two smaller source sections with bounded
overlap. Subdivision is limited to five levels and a minimum source size; failures
after this limit remain explicit rather than publishing incomplete JSON. Successful
recovery plans and child sections persist. Increasing context/output budget allows
a terminal failed section to be attempted again. Verifier truncation subdivides
the candidate list, caching completed groups. Recovery keeps verification enabled.
Run diagnostics record all completed and token-truncated calls, including token
counts, timing and discarded output; recovery costs remain visible. A successful
build does not imply exhaustive extraction. Inactive, disappeared or changed
passages are hidden from the current graph. JSON export includes rejected/stale
claims with review and `source_current` flags.

Feitenoverzicht already sends `think=False`, but permits 16,384 output tokens
and makes repeated calls. Those are plausible latency contributors; Dutch
prompting alone is not an established cause. KG diagnostics distinguish model
loading, input processing and token generation using Ollama timing/token fields,
wall time and emitted thinking-character counts. Smaller windows/output budgets
bound work; verification normally adds another call for nonempty windows. Requests
remain sequential to avoid multiplying GPU context memory. The larger output
budget is a ceiling, not a requirement to generate that many tokens.

Diagnostics retain raw extraction/verifier output locally, including rejected
claims and failure details. Some models cannot disable reasoning: choose a
supported thinking mode or another model. No source text goes to an external
service under the app's default local-only configuration.

## Model assessment and interface

An earlier v2 synthetic Dutch smoke test, after correcting verifier treatment of qualified
claims, retained all 3 expected relationships with local Qwen 3.6 35B and 2 with
Gemma 4 12B. Gemma missed a denial. Neither emitted thinking characters with
thinking off. This tiny example checks compatibility and caught a prompt bug;
it is not a representative model-quality/speed benchmark or a general ranking.

The compact v3 protocol has automated checks for evidence reconstruction,
literal/property fidelity, resumable stage caches and bounded truncation recovery.
Its final live model comparison could not finish because the local Ollama service
became unavailable. The historical smoke results above do not validate v3
quality or establish a throughput estimate for a large investigation.

Compare current models on a small representative document set, reviewing
unsupported/missing relationships, incorrect identities, lost qualifiers and
duration. Edge counts, verifier rejection rates and model self-confidence do
not establish quality. The tab supplies per-run diagnostics and source review;
automatic gold-set precision/recall scoring is not included.

Documents, Graph, Relationships and Build log are subtabs; evidence is on the
right. Drag nodes, click for one-hop focus, use search, wheel zoom, background
pan and Fit graph. **Nodes shown** selects a numeric display limit or **All**
(the default), subject to the search/review/focus filters. The maximum numeric
choice follows the available matching nodes, with no fixed 60-node ceiling.
The graph groups identical labelled edges; the table retains every filtered
source claim regardless of the display limit. Dashed labels distinguish negated/uncertain claims.
Rejected claims are hidden by default. Node positions persist within the open
panel session when dragged, not as saved views. **Rearrange** discards manual
positions; layout and display-limit preferences persist per investigation.
**Connected groups** uses a bounded spring layout within smaller components,
aligns their major axes horizontally, and packs their rectangles using a skyline
so smaller groups can fill space beside larger ones. Components larger than 160
nodes use a compact traversal grid; this is a layout-computation bound, not a
display limit. **Compact grid** packs the complete
matching node set into a single grid. Traversal rows alternate direction so
successive nodes stay close at row boundaries. These deterministic layouts avoid
an empty outer circle. Pairwise spring matrices are bounded per small component;
large graphs retain linear storage for nodes/connections. NumPy is already an
application dependency.
No graph server or new dependency is needed. Diagnostics are in **Build log**.

## JSON import/export

**Import graph** accepts v1 and v2 Bosuil graph JSON. V2 exports add stable graph,
node and claim identities; earlier exports derive an identity from their source
provenance. Imported IDs are remapped into the receiving investigation. Separate
namesakes and explicitly merged identities are preserved. Reimporting does not
overwrite local names, corrections, reviews or notes. Imports into the original
investigation reuse matching existing claims. Invalid references/metadata or
conflicting identity mappings abort the transaction; cancellation also rolls
back the entire import. Import works in a background worker without Ollama.

Full source documents are absent from graph JSON. Imported claims therefore
retain their exact quotes, source paths/hashes, properties, polarity, attribution,
time, original model and review state in separate portable records. They never
create synthetic indexed documents or extraction cache entries. A local passage
is linked only when its identity (or unambiguous path/index/attachment), source
hash and quote agree. Evidence currency is calculated from the current local
source rather than trusted from the JSON flag. Unavailable/changed sources show
an explicit imported-quote label and a plain quote preview, without enabling
source opening. Existing review status remains review of the original claim,
not evidence that the receiving investigation has verified it. Reimporting after
matching documents are indexed can attach local evidence while preserving edits.

Imported quotes remain available if a linked local passage disappears. Reexport
retains portable identities and isolated nodes as well as claims, allowing
imports through another investigation without multiplying the same relationships.
All imported records and identity mappings travel with ordinary `.bosuil` archives.
