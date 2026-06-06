# Agentic Flow — User Journeys

How real user actions flow through the agents and the tools each agent uses.
Read each diagram top‑to‑bottom: **user action → agent (LLM call) → tools → result back to the user.**

Legend:
- 🧑 user action / Telegram tap
- 🤖 agent (an LLM call)
- 🔧 tool the agent can call
- 📤 what the user gets back

---

## Journey 1 — "Give me the news" (post a new item)

Triggered by the schedule firing, or the user tapping **/now**.
For a `sourced` channel the **Researcher** finds a story, then the **Writer** turns it into a post.

```mermaid
flowchart TD
    U["🧑 Schedule tick / user taps /now"] --> O{Channel mode?}

    O -->|sourced| RA["🤖 Researcher agent"]
    O -->|llm_only| WA

    RA --> T1["🔧 Web Search<br/>(live web)"]
    RA --> T2["🔧 fetch_url<br/>(read the article)"]
    RA --> T3["🔧 check_relevance<br/>(fresh? already posted?)"]
    T1 --> RA
    T2 --> RA
    T3 --> RA

    RA -->|nothing good| NN["📤 'Nothing new today'"]
    RA -->|picked a story| WA["🤖 Writer agent<br/>(channel format injected into its prompt)"]

    WA --> DUP{Duplicate of<br/>a past post?}
    DUP -->|yes| WA
    DUP -->|no| POST["📤 News post in Telegram<br/>with Sources / Feedback / Ask buttons"]
```

---

## Journey 2 — "I have a question about this post" (Ask)

User taps **💬 Ask** under a post and types a question.
The **News Q&A** agent answers using the post + the running conversation. No tools — it just answers.

```mermaid
flowchart TD
    U["🧑 Taps 💬 Ask, types a question"] --> QA["🤖 News Q&A agent"]
    QA --> A["📤 Answer in Telegram<br/>with 'Ask again' / 'Done'"]
    A -->|asks again| QA
```

---

## Journey 3 — "Change what this channel posts" (Feedback)

User taps **✏️ Feedback** and says what to change.
The **Prompt Refiner** edits the channel's prompt in place (snippet by snippet,
not a full rewrite) and can also change deterministic settings — content
freshness, the posting schedule (timing + timezone), and the per-channel
format — then shows the result for approval. The current prompt, schedule,
format and session history are already in its instructions, so it doesn't
spend tool calls fetching them.

```mermaid
flowchart TD
    U["🧑 Taps ✏️ Feedback, describes the change"] --> PR["🤖 Prompt Refiner agent"]

    PR --> T1["🔧 edit_prompt<br/>(replace just the snippet that changes)"]
    PR --> T2["🔧 recent_accepted_feedback<br/>(past accepted changes)"]
    PR --> T3["🔧 set_freshness_days<br/>(content recency)"]
    PR --> T4["🔧 set_schedule<br/>(timing + timezone)"]
    PR --> T5["🔧 set_format<br/>(post structure / style)"]
    T1 --> PR
    T2 --> PR
    T3 --> PR
    T4 --> PR
    T5 --> PR

    PR --> P["📤 Proposed changes<br/>Approve / More feedback / Cancel"]
    P -->|more feedback| PR
    P -->|approve| SAVE["✅ Changes saved for the channel"]
```

---

## Journey 4 — "Set up a new channel" (/addchannel or edit)

User runs **/addchannel** (or edits a channel) and describes, in plain words, what they want.
The **Setup Agent** asks clarifying questions and proposes a full channel spec.

```mermaid
flowchart TD
    U["🧑 /addchannel — describes the feed they want"] --> SA["🤖 Setup Agent"]

    SA --> T1["🔧 list_existing_channels<br/>(avoid clashes)"]
    T1 --> SA

    SA --> D{Enough info?}
    D -->|no| Q["📤 Clarifying questions"]
    Q --> U
    D -->|yes| PROP["📤 Proposed channel<br/>Create / Change / Cancel"]
    PROP -->|change| U
    PROP -->|create| SAVE["✅ Channel created & scheduled"]
```

---

## All agents at a glance

```mermaid
flowchart LR
    subgraph Users["🧑 What the user does"]
        A1["Schedule / /now"]
        A2["💬 Ask"]
        A3["✏️ Feedback"]
        A4["/addchannel"]
    end

    A1 --> R["🤖 Researcher"] --> W["🤖 Writer"]
    A2 --> Q["🤖 News Q&A"]
    A3 --> P["🤖 Prompt Refiner"]
    A4 --> S["🤖 Setup Agent"]

    R --- RT["🔧 Web Search · fetch_url · check_relevance"]
    W --- WT["🔧 fetch_url"]
    Q --- QT["🔧 (no tools)"]
    P --- PT["🔧 edit_prompt · recent_accepted_feedback · set_freshness_days · set_schedule · set_format"]
    S --- ST["🔧 list_existing_channels"]
```

> Behind the scenes, both the Researcher's **check_relevance** tool and the
> Writer's duplicate check call an **embedding model** (`text-embedding-3-small`)
> to tell whether a story was already posted. It runs automatically — the user
> never triggers it directly.
