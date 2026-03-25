"""Core expert system: fast path (beliefs) + slow path (LLM) + self-improvement."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Answer:
    """An answer from the expert system."""

    text: str
    justification: str | None = None
    source: str = "unknown"  # "fast_path" or "slow_path"
    belief_id: str | None = None


@dataclass
class ExpertSystem:
    """LLM-powered expert system with truth-maintained reasoning.

    Fast path: search beliefs.md for existing knowledge.
    Slow path: LLM reasons from sources, extracts new beliefs.
    Self-improvement: slow path feeds fast path automatically.
    """

    domain: str
    beliefs_path: Path
    reasons_db: Path
    repo_path: Path | None = None
    model: str = "claude"
    _beliefs_text: str | None = field(default=None, repr=False)

    @classmethod
    def load(cls, expert_dir: Path) -> ExpertSystem:
        """Load an expert system from a directory."""
        config_path = expert_dir / ".expert" / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No expert system config at {config_path}")
        config = json.loads(config_path.read_text())
        return cls(
            domain=config["domain"],
            beliefs_path=expert_dir / "beliefs.md",
            reasons_db=expert_dir / "reasons.db",
            repo_path=Path(config["repo"]) if config.get("repo") else None,
        )

    @classmethod
    def init(cls, expert_dir: Path, domain: str, repo: Path | None = None) -> ExpertSystem:
        """Initialize a new expert system."""
        expert_dir.mkdir(parents=True, exist_ok=True)
        config_dir = expert_dir / ".expert"
        config_dir.mkdir(exist_ok=True)

        config = {"domain": domain}
        if repo:
            config["repo"] = str(repo)

        (config_dir / "config.json").write_text(json.dumps(config, indent=2))

        # Initialize beliefs and reasons
        beliefs_path = expert_dir / "beliefs.md"
        if not beliefs_path.exists():
            beliefs_path.write_text(f"# Beliefs: {domain}\n\n")

        reasons_db = expert_dir / "reasons.db"

        return cls(
            domain=domain,
            beliefs_path=beliefs_path,
            reasons_db=reasons_db,
            repo_path=repo,
        )

    @property
    def beliefs_text(self) -> str:
        """Load and cache beliefs.md content."""
        if self._beliefs_text is None:
            if self.beliefs_path.exists():
                self._beliefs_text = self.beliefs_path.read_text()
            else:
                self._beliefs_text = ""
        return self._beliefs_text

    def invalidate_cache(self) -> None:
        """Clear cached beliefs text (call after RMS changes)."""
        self._beliefs_text = None

    def search_beliefs(self, query: str) -> list[dict]:
        """Fast path: search beliefs.md for matches.

        Uses keyword matching against the flat markdown file.
        Returns matching belief blocks with their IDs and text.
        """
        keywords = _extract_keywords(query)
        if not keywords:
            return []

        matches = []
        current_belief = None
        current_lines: list[str] = []

        for line in self.beliefs_text.splitlines():
            if line.startswith("### "):
                # Save previous belief if it matched
                if current_belief and current_lines:
                    block = "\n".join(current_lines)
                    score = sum(1 for kw in keywords if kw in block.lower())
                    if score > 0:
                        matches.append({
                            "id": current_belief,
                            "text": block,
                            "score": score,
                        })
                # Start new belief
                current_belief = line.replace("### ", "").strip()
                current_lines = [line]
            elif current_belief:
                current_lines.append(line)

        # Don't forget the last belief
        if current_belief and current_lines:
            block = "\n".join(current_lines)
            score = sum(1 for kw in keywords if kw in block.lower())
            if score > 0:
                matches.append({
                    "id": current_belief,
                    "text": block,
                    "score": score,
                })

        matches.sort(key=lambda m: m["score"], reverse=True)
        return matches

    def explain(self, belief_id: str) -> str | None:
        """Get the justification chain for a belief via RMS."""
        try:
            result = subprocess.run(
                ["reasons", "explain", belief_id],
                capture_output=True,
                text=True,
                cwd=self.beliefs_path.parent,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except FileNotFoundError:
            pass
        return None

    async def ask(self, question: str) -> Answer:
        """Ask the expert system a question.

        Fast path: search beliefs for existing knowledge.
        Slow path: LLM reasons, extracts new belief.
        """
        # Fast path
        matches = self.search_beliefs(question)
        if matches and matches[0]["score"] >= 2:
            best = matches[0]
            justification = self.explain(best["id"])
            return Answer(
                text=best["text"],
                justification=justification,
                source="fast_path",
                belief_id=best["id"],
            )

        # Slow path
        answer = await self._llm_reason(question)

        # Self-improve: extract and store belief
        belief = await self._extract_belief(question, answer)
        if belief:
            self._add_belief(belief)
            self.invalidate_cache()

        return Answer(
            text=answer,
            source="slow_path",
        )

    async def _llm_reason(self, question: str) -> str:
        """Slow path: invoke LLM to reason about the question."""
        prompt = self._build_prompt(question)
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"LLM inference failed: {result.stderr}")
        return result.stdout.strip()

    async def _extract_belief(self, question: str, answer: str) -> dict | None:
        """Extract a belief from an LLM answer for future fast-path use."""
        prompt = (
            f"Extract a single factual belief from this Q&A.\n\n"
            f"Question: {question}\n\n"
            f"Answer: {answer}\n\n"
            f"Respond with JSON: {{\"id\": \"kebab-case-id\", \"text\": \"factual claim\"}}\n"
            f"If no clear factual belief can be extracted, respond with null."
        )
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        try:
            # Extract JSON from response
            text = result.stdout.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return None

    def _add_belief(self, belief: dict) -> None:
        """Add a belief to the RMS and re-export markdown."""
        subprocess.run(
            ["reasons", "assert", belief["id"], belief["text"]],
            capture_output=True,
            text=True,
            cwd=self.beliefs_path.parent,
        )
        subprocess.run(
            ["reasons", "export-markdown"],
            capture_output=True,
            text=True,
            cwd=self.beliefs_path.parent,
        )

    def _build_prompt(self, question: str) -> str:
        """Build LLM prompt with domain context and existing beliefs."""
        parts = [f"You are an expert on: {self.domain}\n"]

        # Include relevant beliefs as context
        matches = self.search_beliefs(question)
        if matches:
            parts.append("Relevant existing knowledge:\n")
            for m in matches[:5]:
                parts.append(m["text"])
                parts.append("")

        parts.append(f"Question: {question}\n")
        parts.append("Answer the question based on your expertise and the knowledge above.")

        return "\n".join(parts)


def _extract_keywords(query: str) -> list[str]:
    """Extract search keywords from a natural language query."""
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "do", "does", "did", "have", "has", "had", "will", "would",
        "can", "could", "should", "may", "might", "shall",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "it", "its", "this", "that", "these", "those",
        "and", "or", "but", "not", "no", "nor",
        "what", "how", "why", "when", "where", "which", "who",
    }
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_-]*", query.lower())
    return [w for w in words if w not in stop_words and len(w) > 2]
