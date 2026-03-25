# ftl-expert-system

Library for building self-improving expert systems. Provides the fast-path/slow-path pattern, automatic belief extraction, and hit-rate tracking. Designed to be imported by [expert-service](https://github.com/benthomasson/expert-service) and similar applications.

Uses [ftl-reasons](https://github.com/benthomasson/ftl-reasons) (RMS) for truth-maintained reasoning, with LLMs solving the knowledge acquisition bottleneck that killed expert systems in the 1980s.

## Architecture

```
Query arrives
    |
    v
+------------------+
|  Fast path:      |--hit--> Return belief + justification chain
|  search beliefs  |         (no LLM call, no cost, instant)
+------------------+
    | miss
    v
+------------------+
|  Slow path:      |--> LLM reasons from sources
|  LLM inference   |    (code, docs, state, observations)
+------------------+
    |
    v
+------------------+
|  Learn:          |--> Extract belief -> add to RMS
|  Self-improve    |    (next time this hits the fast path)
+------------------+
```

The fast path gets faster over time. The slow path feeds the fast path. AI cost converges toward zero as the knowledge base grows.

## Install

```bash
pip install ftl-expert-system
```

## What this library provides

### 1. SelfImprovingAgent

Wraps any LLM agent with fast-path-first routing and automatic belief extraction:

```python
from ftl_expert_system import ExpertSystem, Answer

expert = ExpertSystem.load(expert_dir)
answer = await expert.ask("Does the SSH layer validate host keys?")

if answer.source == "fast_path":
    # Answered from beliefs, no LLM call
    print(f"Belief: {answer.belief_id}")
else:
    # LLM reasoned from scratch, new belief extracted for next time
    print("Learned new belief")
```

### 2. Multi-expert parallel search

Search multiple knowledge bases simultaneously. Grep is orders of magnitude cheaper than LLM inference, so searching 10 experts in parallel is still faster than a single LLM call:

```python
from ftl_expert_system import MultiExpertSearch

search = MultiExpertSearch([expert_a, expert_b, expert_c])
matches = search.search("host key verification")
# Returns ranked results across all experts
```

### 3. FastPathMetrics

Track hit rate to measure whether the expert system is self-improving:

```python
from ftl_expert_system import FastPathMetrics

metrics = FastPathMetrics()
# ... after each query ...
print(f"Hit rate: {metrics.hit_rate:.1%}")
print(f"Beliefs extracted: {metrics.beliefs_extracted}")
```

### 4. BeliefExtractor

Extract structured beliefs from LLM answers:

```python
from ftl_expert_system import BeliefExtractor

extractor = BeliefExtractor()
belief = await extractor.extract(question, llm_answer)
# {"id": "ssh-validates-host-keys", "text": "SSH layer validates host keys..."}
```

## Integration with expert-service

expert-service already has multi-project knowledge bases, ftl-reasons RMS with PostgreSQL, 20 chat tools, meta-expert routing, and eval framework. This library adds the self-improving pattern:

```python
# In expert-service's chat agent setup:
from ftl_expert_system import ExpertSystem

# Wrap the existing agent with fast-path-first routing
expert = ExpertSystem.load(project_dir)

# Before calling the LLM agent, check beliefs
matches = expert.search_beliefs(user_question)
if matches:
    return matches[0]  # Fast path, no LLM needed

# Otherwise, let the agent reason, then extract a belief
answer = await agent.invoke(user_question)
belief = await expert.extract_belief(user_question, answer)
expert.add_belief(belief)
```

## Why not Drools / CLIPS / PyKE?

Those are **production rule systems** (OPS5 lineage). This is a **truth maintenance system** (Doyle/de Kleer lineage).

| Capability | Drools/CLIPS/PyKE | ftl-expert-system |
|---|---|---|
| Dependency-directed retraction | No | Yes (RMS) |
| Non-monotonic reasoning (UNLESS) | No | Yes (outlist) |
| Automatic restoration | No | Yes |
| LLM-friendly export | No | Yes (markdown) |
| Self-improving from LLM | No | Yes |
| Knowledge acquisition | Manual DSL | Automated via LLM |

Production rule systems embed business logic in enterprise apps. This library enables LLM-powered knowledge acquisition with truth-maintained reasoning.

## Self-improvement modes

**Declarative (code-expert style):** LLM reads artifacts, extracts factual beliefs, adds to RMS with dependencies. Best for understanding domains.

**Imperative (ftl2-ai-loop style):** LLM solves a problem, writes a deterministic rule, rule fires next time without LLM. Best for automation tasks.

Both feed the fast path. Both reduce AI cost over time toward zero.

## Ecosystem

| Tool | Role |
|------|------|
| **ftl-expert-system** | Self-improving expert system library (this repo) |
| [expert-service](https://github.com/benthomasson/expert-service) | Web service that uses this library |
| [ftl-reasons](https://github.com/benthomasson/ftl-reasons) | RMS -- dependency tracking, retraction cascades |
| [ftl-beliefs](https://github.com/benthomasson/ftl-beliefs) | Belief registry -- provenance, staleness detection |
| [code-expert](https://github.com/benthomasson/code-expert) | Deep code analysis -- scan, explore, extract, derive |

## Prior Art

- **Doyle (1979)** -- Truth Maintenance Systems: dependency-directed backtracking
- **de Kleer (1986)** -- ATMS: assumption-based reasoning, nogoods
- **Feigenbaum** -- identified the knowledge acquisition bottleneck
- **MYCIN, R1/XCON, DENDRAL** -- classical expert systems that worked but couldn't scale knowledge acquisition
