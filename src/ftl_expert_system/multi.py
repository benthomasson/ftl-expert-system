"""Parallel search across multiple expert knowledge bases."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ftl_expert_system.expert import ExpertSystem


@dataclass
class MultiExpertMatch:
    """A match from a specific expert."""

    expert_domain: str
    belief_id: str
    text: str
    score: int
    beliefs_path: str


class MultiExpertSearch:
    """Search multiple expert knowledge bases in parallel.

    Grep is orders of magnitude cheaper than LLM inference,
    so searching 10 experts in parallel is still faster than
    a single LLM call.
    """

    def __init__(self, experts: list[ExpertSystem]) -> None:
        self.experts = experts

    def search(self, query: str, limit: int = 10) -> list[MultiExpertMatch]:
        """Search all experts synchronously and merge results."""
        all_matches: list[MultiExpertMatch] = []
        for expert in self.experts:
            matches = expert.search_beliefs(query)
            for m in matches:
                all_matches.append(MultiExpertMatch(
                    expert_domain=expert.domain,
                    belief_id=m["id"],
                    text=m["text"],
                    score=m["score"],
                    beliefs_path=str(expert.beliefs_path),
                ))
        all_matches.sort(key=lambda m: m.score, reverse=True)
        return all_matches[:limit]

    async def search_async(self, query: str, limit: int = 10) -> list[MultiExpertMatch]:
        """Search all experts in parallel using asyncio."""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, expert.search_beliefs, query)
            for expert in self.experts
        ]
        results = await asyncio.gather(*tasks)

        all_matches: list[MultiExpertMatch] = []
        for expert, matches in zip(self.experts, results):
            for m in matches:
                all_matches.append(MultiExpertMatch(
                    expert_domain=expert.domain,
                    belief_id=m["id"],
                    text=m["text"],
                    score=m["score"],
                    beliefs_path=str(expert.beliefs_path),
                ))
        all_matches.sort(key=lambda m: m.score, reverse=True)
        return all_matches[:limit]
