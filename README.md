# News Bot — Usage Guide

A **personal Telegram news assistant**. You describe what you want to read about —
AI breakthroughs, geopolitics, your favourite football club, a niche research
area — and the bot researches the live web on a schedule you control and drops
short, well-sourced briefings into a single Telegram chat with you. Every news
post comes with a **🔗 Sources** button (no fabricated links), a
**✏️ Feedback** button (tell it _"shorter"_ / _"more technical"_ / _"skip
funding rounds"_ and it refines the channel's prompt for you), and a
**💬 Ask** button (ask a follow-up question about that specific post).

The same engine doubles as a **daily-drip generator** for anything else you'd
like recurring: a *Kazakh word a day*, a *daily interesting fact*, a *math
concept of the morning*, an *English idiom* — if it can be described, it can
be scheduled.

**Single-owner by design.** The bot only responds to the Telegram chat id you
put in `owner_chat_id`. Strangers who find it are silently ignored — your
OpenAI bill is safe.

---

## 1. Prerequisites

You need four things before the bot can run.

### 1.1 A Telegram bot

1. Open Telegram, talk to **[@BotFather](https://t.me/BotFather)**.
2. `/newbot`, pick a name and a username ending in `bot`.
3. Save the **bot token** BotFather returns — it goes into `telegram.bot_token`.
4. Send a message to your new bot from your own account (this is the chat the bot
   will post into).

### 1.2 Telegram `api_id` / `api_hash`

1. Sign in at **[my.telegram.org](https://my.telegram.org)** with the phone number
   of your Telegram account.
2. Go to **API development tools**, create an application (any name / shortname).
3. Save **`App api_id`** and **`App api_hash`** — they go into `telegram.api_id`
   and `telegram.api_hash`.

### 1.3 Your `owner_chat_id`

This is the numeric chat id of you talking to your bot. Easiest way:

1. Send any message to your bot.
2. Open `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates` in a browser.
3. Look for `"chat":{"id": 123456789, ...}` — that integer is your
   `owner_chat_id`.

### 1.4 LLM key

- **OpenAI API key** — [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
  Goes into `openai.api_key`.

That's it. Sourced channels use OpenAI's **native** `web_search_preview` tool
(exposed through PydanticAI's `WebSearchTool` built-in) — no separate search
API key required. Web-search calls are billed by OpenAI alongside the LLM call.

---

## 2. Install & run

### 2.1 Local (Python 3.11+)

```bash
git clone <this repo>
cd news-bot
cp config.example.yaml config.yaml
$EDITOR config.yaml                 # fill in the four sets of credentials above

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_bot.py
```

On first run the bot:
- creates `./data/news-bot.sqlite` (state + history),
- loads every `./channels/*.yaml` into the DB,
- starts APScheduler and Telegram long-polling.

Send `/start` to your bot in Telegram to confirm it is alive — you should get
the welcome message back. Then `/create_channel` to set up your first feed.

### 2.2 Docker

```bash
cp config.example.yaml config.yaml  # fill in
mkdir -p data
docker compose up --build -d
docker compose logs -f
```

The compose file mounts `./config.yaml`, `./channels/`, `./prompts/` (read-only)
and `./data/` (read-write) into the container.

---

## 3. `config.yaml` walkthrough

All infra / secrets live here. Prompts and per-channel knobs live in
`./channels/*.yaml` (and can also be edited live via the bot — see §6).

```yaml
telegram:
  api_id: 12345                  # from my.telegram.org
  api_hash: "REPLACE_ME"
  bot_token: "REPLACE_ME"        # from @BotFather
  owner_chat_id: 111222333       # see §1.3

openai:
  api_key: "sk-REPLACE_ME"
  default_writer_model: gpt-5.4-mini     # used unless overridden in a channel
  default_researcher_model: gpt-5.4-mini
  default_refiner_model: gpt-5.4-mini
  default_setup_model: gpt-5.4-mini
  embedding_model: text-embedding-3-small   # used for duplicate/collision detection
  flex_mode: false               # use OpenAI flex service tier (lower cost, slower)

researcher:
  per_tick_fetch_budget: 8       # hard cap on fetch_url calls per channel tick
                                 # (web search itself is run by the LLM provider
                                 # via WebSearchTool — billed per-call by OpenAI,
                                 # capped indirectly by cost_control below)
  per_tick_check_budget: 3       # hard cap on check_relevance (collision-check) rounds per tick
  temperature: null              # researcher sampling temperature; raise (e.g. 0.8) to
                                 # diversify queries. Leave null for OpenAI reasoning
                                 # models (gpt-5.x) — they reject a non-default temperature.

dedup:
  similarity_threshold: 0.75     # cosine sim above which two posts are the same story
  lookback_days: 14              # only compare against posts from the last N days
  log_candidates: false          # append every candidate + its top similarity scores to a JSONL file
  candidate_log_path: data/dedup_candidates.jsonl   # where those score logs go when enabled

cost_control:
  global_daily_usd: 1.50         # combined cap across all channels & agents
  per_channel_default_daily_usd: 0.50
  on_threshold: pause            # warn | downgrade | pause
  downgrade_to: gpt-5.4-mini     # used when on_threshold == downgrade

scheduling:
  probabilistic_jitter_seconds: 1800   # ±30 min jitter on probabilistic firings

storage:
  db_path: data/news-bot.sqlite

observability:
  lmnr_enabled: false            # see §9
  lmnr_project_api_key: ""

logging:
  level: INFO                    # DEBUG | INFO | WARNING | ERROR
```

---

## 4. Channels

A **channel** is one stream of posts: a topic prompt + a schedule + a format.
Every channel lands in the same Telegram chat with you, tagged by its hashtag.

### 4.1 Starter channels

The repo ships with an **empty `./channels/` directory** so you start from a
clean slate. The recommended way to add your first channel is to send
`/create_channel` to the bot once it's running and describe — in your own words —
what you want to read and how often. The Setup Agent will draft the
YAML, show it to you, let you iterate, and save it on approval.

If you'd rather hand-author one, drop a file in `./channels/<id>.yaml` using
the schema below and restart.

A few channels people commonly set up first:

- **AI news** — `sourced`, weekdays 09:00 & 18:00 — new models, capability jumps.
- **Football club** — `sourced`, daily 08:00 — transfers, match results, injuries.
- **Geopolitics digest** — `sourced`, daily 19:00 — major moves in a region you follow.
- **Kazakh word a day** — `llm_only`, every 6h — vocabulary drip.
- **Math concept** — `llm_only`, daily 10:00 — one ML-interview-relevant concept.
- **Daily interesting fact** — `llm_only`, daily 09:00 — one short, surprising fact.

### 4.2 Channel YAML schema

```yaml
id: my_channel                   # unique, snake_case; used in /now my_channel
display_name: My Channel
hashtag: "#my_channel"           # appears at the top of every post
mode: sourced                    # sourced | llm_only
model_writer: gpt-5.4            # optional; overrides openai.default_writer_model
model_researcher: gpt-5.4-mini   # optional; sourced channels only
format: |                        # optional; free-text formatting instructions
  one bold word, then its meaning, then a short example sentence

topic_prompt: |
  Free-form description of what this channel should post about, in what tone,
  with what to skip. This is what the Writer sees as per-channel context and
  what the Feedback flow refines.

schedule:
  kind: cron                     # cron | interval | probabilistic
  spec: "0 9,18 * * 1-5"         # see §4.4

# sourced channels only
search:
  freshness_days: 7              # restrict web_search to last N days
  topic: news                    # news | general

dedup_window_n: 7                # how many recent items to put in the "do not repeat" list

images:
  enabled: false                 # image generation is a stub in v1
```

### 4.3 Adding a new channel

Two ways:

**A. From Telegram (recommended).** Send `/create_channel` to the bot, describe
what you want in natural language (topic, frequency, tone, things to skip),
and the Setup Agent drafts a channel YAML for you. You can iterate
("more technical", "twice a day instead of daily", "no funding rounds")
before approving. On approval the YAML is written to `./channels/<id>.yaml`
and the channel is scheduled immediately — no restart needed.

**B. Hand-author the YAML.**

1. Drop a new `./channels/<id>.yaml` file in.
2. Restart the bot (YAML files are loaded on startup).
3. The channel is upserted into SQLite and scheduled automatically.

Per-channel formatting is free text: set the optional `format` field to
plain-language instructions for how each post should be structured/styled
(e.g. "three short bullet points, no emoji"). Leave it out and the Writer
uses sensible defaults. The Setup Agent sets it for you only when you ask
for a specific format, and the Feedback flow can edit it later.

### 4.4 Schedule kinds

```yaml
# Cron — APScheduler's crontab syntax
schedule: { kind: cron, spec: "0 9,18 * * 1-5" }   # 09:00 & 18:00 on weekdays

# Interval — every N hours / minutes / seconds
schedule: { kind: interval, spec: { hours: 6 } }
schedule: { kind: interval, spec: { minutes: 90 } }

# Probabilistic — roughly N firings per day, jittered, inside a window
schedule:
  kind: probabilistic
  spec:
    times_per_day: 2
    start_hour: 10
    end_hour: 22
```

Probabilistic firings are jittered by `scheduling.probabilistic_jitter_seconds`
from `config.yaml`.

---

## 5. Commands

Send these to your bot in Telegram:

| Command | What it does |
|---|---|
| `/start` | Welcome message + overview of what the bot does. |
| `/create_channel` | Guided setup — describe a channel and the bot drafts the YAML. (alias: `/addchannel`) |
| `/change_channel [channel_id]` | Edit an existing channel's prompt/schedule via the same guided flow. No arg → picker. |
| `/channels` | List all channels with id, mode, enabled state + action buttons. |
| `/now [channel_id]` | Fire that channel right now (bypasses schedule). No arg → picker. |
| `/pause [channel_id]` | Stop scheduling that channel (in-DB toggle). No arg → picker. |
| `/resume [channel_id]` | Re-enable a paused channel. No arg → picker. |
| `/del_channel [channel_id]` | Permanently delete a channel. No arg → picker. (alias: `/delchannel`) |
| `/usage` | Per-agent / per-channel token + USD spend in the last 24h. |
| `/cancel` | Abort an in-progress onboarding flow. |
| `/help` | List of commands. |

Example:

```
/create_channel
/now ai_news
/pause kazakh
/usage
```

---

## 6. Feedback flow (multi-shot prompt refinement)

Every delivered post has an **`✏️ Feedback`** inline button.

1. Tap **Feedback**.
2. Bot replies *"What should I change for `<channel>`? Reply to this message."*
3. Type what you want changed: *"too long, max 3 bullets"* / *"more code examples"*
   / *"translate to Russian"*.
4. The **Prompt Refiner** agent already has the current channel prompt, the
   message you reacted to and the session history in its instructions (it
   spends no tool calls fetching them). It edits the prompt **in place**,
   snippet by snippet (via its `edit_prompt` tool) rather than rewriting it
   wholesale, and returns the proposed prompt with a one-line change summary.
   For `sourced` channels it can also adjust the content-recency window
   (`search.freshness_days`) when you ask for fresher/older items.
5. You see three buttons:
   - **✅ Approve** — the new prompt is saved to the channel and used from the
     next tick on.
   - **❌ Cancel** — drop the proposal, keep the current prompt.
   - **✏️ More feedback** — keeps the same session open and re-runs the
     refiner with your additional instruction layered on top of its last
     proposal. You can iterate any number of times.

A new feedback session for the same channel **supersedes** any still-pending
proposal so you never accidentally approve a stale one.

---

## 7. Sources & Ask buttons

For `mode: sourced` channels, every post has a **`🔗 Sources`** inline button.
Tap it and the bot sends a follow-up message with the canonical URLs the LLM
picked when researching. The Researcher is forbidden from fabricating URLs:
every link it emits must come from a real `web_search` hit it actually saw.

Every post also has a **`💬 Ask`** button. Tap it, type a question, and the
**News Q&A** agent answers using that post plus the running conversation — it
calls no tools, just answers. You can keep asking follow-ups until you tap
**Done**.

If a sourced channel finds nothing new at firing time, you get an explicit
*"nothing new for `<channel>`"* message instead of silence. This is intentional
for v1 so scheduling and source fetching are easy to debug.

---

## 8. Cost controls

Configured in `config.yaml > cost_control`. The Cost Controller is consulted
**before** each channel tick and **after** each LLM call (it accumulates token
usage in a `usage_ledger` SQLite table).

When the global daily USD cap or the per-channel cap is hit:

- `on_threshold: warn` — Telegram notification, but the bot keeps firing.
- `on_threshold: downgrade` — switches the channel's model to
  `cost_control.downgrade_to` for the rest of the day.
- `on_threshold: pause` — disables the channel for the rest of the day.

Token counts come from the OpenAI API's `usage` field; USD is computed against
a small per-model price table in `./llm/cost.py`. Edit that table if you use
a model it doesn't price.

---

## 9. Laminar observability (optional, but recommended)

[Laminar (lmnr.ai)](https://lmnr.ai) traces every PydanticAI agent run end to
end, with full prompt / tool-call / output visibility.

To enable:

```yaml
observability:
  lmnr_enabled: true
  lmnr_project_api_key: "lmnr-prj-..."
```

Then restart. Each agent (`researcher` / `writer` / `prompt_refiner` /
`setup_agent` / `news_qa`) shows up as a separate span; tools
(`web_search`, `fetch_url`, `check_relevance`, `edit_prompt`,
`set_freshness_days`, `set_schedule`, `set_format`, …) are nested under
their agent.

---
