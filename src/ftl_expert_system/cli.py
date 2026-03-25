"""CLI for ftl-expert-system."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from ftl_expert_system.expert import ExpertSystem


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize an expert system."""
    expert_dir = Path(args.dir or ".").resolve()
    repo = Path(args.repo).resolve() if args.repo else None
    expert = ExpertSystem.init(expert_dir, domain=args.domain, repo=repo)
    print(f"Initialized expert system: {expert.domain}")
    print(f"  Directory: {expert_dir}")
    if repo:
        print(f"  Repository: {repo}")
    print(f"  Beliefs: {expert.beliefs_path}")


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask the expert system a question."""
    expert = ExpertSystem.load(Path(args.dir or ".").resolve())
    answer = asyncio.run(expert.ask(args.question))

    if answer.source == "fast_path":
        print(f"[fast path] belief: {answer.belief_id}\n")
    else:
        print("[slow path] reasoned from sources\n")

    print(answer.text)

    if answer.justification:
        print(f"\n--- Justification ---\n{answer.justification}")


def cmd_search(args: argparse.Namespace) -> None:
    """Search the knowledge base."""
    expert = ExpertSystem.load(Path(args.dir or ".").resolve())
    matches = expert.search_beliefs(args.query)

    if not matches:
        print("No matching beliefs found.")
        return

    for m in matches[:args.limit]:
        print(f"[score={m['score']}] {m['id']}")
        # Show first line of text
        lines = m["text"].strip().splitlines()
        if len(lines) > 1:
            print(f"  {lines[1].strip()}")
        print()


def cmd_status(args: argparse.Namespace) -> None:
    """Show expert system status."""
    expert_dir = Path(args.dir or ".").resolve()
    expert = ExpertSystem.load(expert_dir)

    # Count beliefs
    in_count = 0
    out_count = 0
    for line in expert.beliefs_text.splitlines():
        if line.startswith("### "):
            in_count += 1
        if "- Status: OUT" in line:
            out_count += 1
            in_count -= 1

    print(f"Expert System: {expert.domain}")
    print(f"  Directory: {expert_dir}")
    if expert.repo_path:
        print(f"  Repository: {expert.repo_path}")
    print(f"  Beliefs: {in_count} IN, {out_count} OUT")
    print(f"  Database: {expert.reasons_db}")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        prog="expert",
        description="LLM-powered expert system with truth-maintained reasoning",
    )
    parser.add_argument("--dir", "-d", help="Expert system directory (default: cwd)")
    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Initialize an expert system")
    p_init.add_argument("domain", help="Domain description")
    p_init.add_argument("--repo", "-r", help="Repository to analyze")

    # ask
    p_ask = sub.add_parser("ask", help="Ask the expert system a question")
    p_ask.add_argument("question", help="Question to ask")

    # search
    p_search = sub.add_parser("search", help="Search the knowledge base")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", "-n", type=int, default=10, help="Max results")

    # status
    sub.add_parser("status", help="Show expert system status")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "init": cmd_init,
        "ask": cmd_ask,
        "search": cmd_search,
        "status": cmd_status,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
