# Prompt token-efficiency audit

An analysis of every prompt the bot sends, looking for wasted tokens:
duplicated text, tools that re-deliver information already in the prompt, and
prefix instability that defeats provider prompt-caching.

Scope of files reviewed:

- `./prompts/researcher.j2`, `./agents/researcher.py`
- `./prompts/writer.system.j2`, `./agents/writer.py`
- `./prompts/setup_assistant.j2`, `./agents/setup_assistant.py`
- `./prompts/prompt_refiner.j2`, `./agents/prompt_refiner.py`
- `./prompts/news_qa.j2`, `./agents/news_qa.py`
- `./prompts/shared/do_not_repeat.j2`, `./prompts/formats/*.j2`
- `./agents/history.py`, `./orchestrator/prompt_builder.py`

Findings are ordered roughly by token impact. Each item lists the **cause**,
**evidence**, and a **suggested fix**.

---

## 1. Tools that re-return data already injected into the prompt

When a tool's whole job is to hand back data that is already inside the system
prompt, the model pays for it three times: the tool schema/docstring in the
request, the tool-call tokens, and the tool-result tokens echoed back into
context — for zero new information.

### 1a. Setup Assistant — `list_existing_channels` duplicates the prompt
- **Evidence:** `./prompts/setup_assistant.j2:78-84` already renders every
  existing channel (`id — display_name (mode)`). The tool
  `list_existing_channels` (`./agents/setup_assistant.py:81-86`) returns the
  same list (id/display_name/mode/hashtag).
- **Impact:** A full extra copy of the channel list per call if the model calls
  it (and the docstring/schema cost even if it does not).
- **Fix:** Drop the tool; the data is static for the session and already in the
  prompt. If hashtag matters, add it to the prompt line instead.

### 1b. Setup Assistant — `list_format_templates` duplicates the prompt
- **Evidence:** `./prompts/setup_assistant.j2:43` already lists the available
  template names. `list_format_templates` (`./agents/setup_assistant.py:89-91`)
  returns the full template dicts.
- **Fix:** Either remove the tool (names in prompt are enough to choose a slug),
  or remove the names from the prompt and keep only the tool — do not ship both.

### 1c. Prompt Refiner — `recent_accepted_feedback` duplicates the prompt
- **Evidence:** `./prompts/prompt_refiner.j2:64-69` already renders the recently
  accepted edits. The tool `recent_accepted_feedback`
  (`./agents/prompt_refiner.py:112-114`) returns the identical `recent_accepted`
  list from deps.
- **Fix:** Remove the tool; the section is already in the system prompt.

---

## 2. The same instruction stated 2–3× inside the Researcher prompt

The researcher describes each tool's semantics in **three** places, and
pydantic-ai sends the tool docstrings to the model as part of the tool schema,
so the duplication is real request tokens on every tick.

### 2a. `fetch_url` described twice (prompt + docstring)
- **Evidence:** `./prompts/researcher.j2:34-40` explains fetch_url (capped,
  stored under an `id`, only fetched sources can be published). The docstring
  `./agents/researcher.py:77-85` says the same thing again ("register it as a
  pickable source", id `s1`, "You publish by returning that id as `picked_id`").
- **Fix:** Keep the operational detail in the docstring (the model always sees
  it via the schema) and reduce the prompt bullet to one line, or vice-versa —
  not both at full length.

### 2b. `check_relevance` described twice, at length (prompt + docstring)
- **Evidence:** `./prompts/researcher.j2:44-65` is a ~20-line description of the
  two deterministic checks. The docstring `./agents/researcher.py:156-178`
  repeats the same two checks ("Date relevance", "Duplicate / collision",
  `is_relevant` = fresh AND not duplicate, "call it ONCE"). This is the single
  largest duplication in the codebase.
- **Fix:** Pick one home. The docstring is the cheaper place (always sent, and
  closest to the call site); trim the prompt section to a one-line pointer.

### 2c. The "Loop" section restates the tools a third time
- **Evidence:** `./prompts/researcher.j2:67-97` walks the same fetch → check →
  commit flow already covered by both the Tools section and the docstrings
  (step 5 = fetch_url, step 6 = check_relevance ONCE, step 7 = return id).
- **Fix:** Collapse the loop into a short ordered checklist that references the
  tools by name without re-explaining what they do.

### 2d. `picked_id` / id-echo explained in many places
- **Evidence:** the "only fetched sources can be picked, echoed `source_id`"
  idea appears in `./prompts/researcher.j2:38-39,61-64,81-82,88-90`, in the
  `picked_id` field description (`./agents/researcher.py:30-38`), and in
  `RelevanceCandidate.id` (`./agents/researcher.py:101-108`). Five+ restatements.
- **Fix:** State it once authoritatively (the field description) and remove the
  repeated reminders.

---

## 3. Repeated values / concepts that don't need repeating

### 3a. `today` interpolated twice in the Researcher prompt
- **Evidence:** `{{ today }}` at `./prompts/researcher.j2:8` and again at
  `./prompts/researcher.j2:50`. The date is also recomputed and returned by
  `check_relevance` in its result, so the model gets it a third time at call
  time.
- **Fix:** Keep the single authoritative mention at the top; drop the inline
  copy in the check_relevance description.

### 3b. `freshness_days` re-explained repeatedly
- **Evidence:** `./prompts/researcher.j2:17-18`, `:51-54`, and `:85-92` all
  re-litigate "default N days unless the topic says tighter, then that wins."
- **Fix:** Define the window once; later mentions can just say "the freshness
  window".

### 3c. Writer — "no hashtags in body" said three times
- **Evidence:** `./prompts/writer.system.j2:55-56` (body spec), `:72` (assembly
  block), and `:81-82` ("Do NOT write any hashtags inside `body`").
- **Fix:** Once is enough; keep it in the assembly block and delete the others.

### 3d. Writer — title attention/disclosure idea spread across three sections
- **Evidence:** the "title must hook AND disclose" theme appears in Voice
  (`:16-17`), Attention+disclosure (`:28-36`), and Format (`:51-53`), with the
  NVIDIA example only in one. Lots of overlapping prose.
- **Fix:** Consolidate into the Attention+disclosure section; make Format a
  terse field spec (length, no trailing punctuation) that points back to it.

### 3e. Writer — `sources_used` / "put the url in sources_used" repeated
- **Evidence:** `./prompts/writer.system.j2:59` and again `:120-121`.
- **Fix:** State once.

---

## 4. Prefix instability that defeats prompt caching

Provider prompt-caches key on the **longest stable prefix**. Putting volatile
values early in a prompt invalidates the cache for everything after them.

### 4a. Researcher puts per-channel/volatile data *before* the static rules
- **Evidence:** `./prompts/researcher.j2:6-19` (channel id, display_name,
  `topic_prompt_active`, **`today`**, freshness, search mode) comes first; the
  large, channel-independent instruction blocks (Tools, Loop, Selection
  priority — `:21-111`) come after.
- **Consequence:** No two channels share a cacheable prefix, and because
  `today` sits at line 8, the *entire* researcher prompt cache busts every day
  for every channel.
- **Fix:** Reorder to: static rules first (Tools/Loop/Selection), then a
  per-channel block, then the dynamic `do_not_repeat` window last. Compare with
  `./prompts/writer.system.j2`, which already does static-first (`:1-85`) then
  channel context (`:86+`) — that ordering is the target.

### 4b. Channel-specific values baked into otherwise-static instruction text
- **Evidence:** `./prompts/researcher.j2:29-30` interpolates
  `{{ channel.search_topic }}` and `{{ channel.display_name }}` into the example
  search queries, which are otherwise identical across channels. Likewise
  `{{ today }}`/`{{ freshness_days }}` are spliced into the check_relevance
  description (`:50-54`).
- **Consequence:** Even the "static" instruction region varies per channel/day,
  so it can't be cached across runs.
- **Fix:** Phrase the examples generically (e.g. "biggest <topic> news today")
  and move the channel's actual topic/freshness into the per-channel block at
  the end. Keep variables out of the shared instruction body.

### 4c. Consistent system-prompt construction (good — keep it)
- **Note:** `./agents/history.py:26-28` correctly places the stable system
  prompt as the first message so it forms a cache-friendly prefix for the
  conversational agents (news_qa, prompt_refiner, setup_assistant). The only
  caveat is §4a/§4b: the *content* of those system prompts must itself be
  prefix-stable to benefit.

---

## 5. Smaller items

### 5a. Researcher user prompt restates the system prompt
- **Evidence:** `./orchestrator/orchestrator.py:108` sends "Find one item to
  cover this tick. Follow the loop in the system prompt." — harmless but pure
  restatement.
- **Fix:** Optional; could be shortened to "Begin." since the loop is already in
  the system prompt.

### 5b. Writer assembly block partially re-derives the Format spec
- **Evidence:** `./prompts/writer.system.j2:62-84` ("How your fields are
  assembled … do not duplicate") overlaps with the Format section `:49-60`.
- **Fix:** Keep the literal layout skeleton (useful) but drop the prose that
  repeats field rules already in Format.

### 5c. Cross-prompt voice rules duplicated (low priority, separate agents)
- **Evidence:** "plain language / no filler / concrete nouns over adjectives /
  don't fabricate" appears in both `./prompts/writer.system.j2:18-22` and
  `./prompts/news_qa.j2:5-9`.
- **Note:** These are different agents/calls, so this is not duplication within a
  single request. Could be factored into a shared partial for maintainability,
  but it is **not** a per-call token cost.

---

## Priority summary

| # | Issue | Type | Est. impact |
|---|-------|------|-------------|
| 2b | `check_relevance` described twice at length | duplication | high |
| 4a | Researcher volatile data before static rules (`today` at top) | caching | high |
| 2a/2c/2d | fetch_url + loop + id-echo restated 2–3× | duplication | high |
| 4b | channel/date values inside static instructions | caching | medium |
| 1a/1b/1c | tools returning data already in prompt | redundant tools | medium |
| 3a–3e | repeated values (`today`, freshness, hashtags, sources_used) | duplication | medium |
| 5a/5b/5c | minor restatements | duplication | low |
</content>
