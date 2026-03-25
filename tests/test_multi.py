"""Tests for multi-expert parallel search."""

import pytest

from ftl_expert_system.expert import ExpertSystem
from ftl_expert_system.multi import MultiExpertSearch


class TestMultiExpertSearch:
    def test_search_across_experts(self, tmp_path):
        # Create two experts with different beliefs
        expert_a = ExpertSystem.init(tmp_path / "expert-a", domain="SSH Security")
        expert_a.beliefs_path.write_text(
            "# Beliefs\n\n"
            "### ssh-host-key-verification\n"
            "SSH host key verification is enabled by default.\n"
            "- Status: IN\n"
        )
        expert_a.invalidate_cache()

        expert_b = ExpertSystem.init(tmp_path / "expert-b", domain="Policy Engine")
        expert_b.beliefs_path.write_text(
            "# Beliefs\n\n"
            "### policy-validates-fields\n"
            "Policy engine validates rule fields at construction time.\n"
            "- Status: IN\n"
        )
        expert_b.invalidate_cache()

        search = MultiExpertSearch([expert_a, expert_b])

        # Search for SSH - should find expert_a's belief
        matches = search.search("SSH host key")
        assert len(matches) >= 1
        assert matches[0].expert_domain == "SSH Security"
        assert matches[0].belief_id == "ssh-host-key-verification"

        # Search for policy - should find expert_b's belief
        matches = search.search("policy validates fields")
        assert len(matches) >= 1
        assert matches[0].expert_domain == "Policy Engine"

    def test_search_merges_and_ranks(self, tmp_path):
        expert_a = ExpertSystem.init(tmp_path / "expert-a", domain="Domain A")
        expert_a.beliefs_path.write_text(
            "# Beliefs\n\n"
            "### shared-topic-a\n"
            "SSH security is important for production.\n"
            "- Status: IN\n"
        )
        expert_a.invalidate_cache()

        expert_b = ExpertSystem.init(tmp_path / "expert-b", domain="Domain B")
        expert_b.beliefs_path.write_text(
            "# Beliefs\n\n"
            "### shared-topic-b\n"
            "SSH security requires host key verification and command injection prevention.\n"
            "- Status: IN\n"
        )
        expert_b.invalidate_cache()

        search = MultiExpertSearch([expert_a, expert_b])
        matches = search.search("SSH security host key verification")
        # expert_b should rank higher (more keyword matches)
        assert len(matches) == 2
        assert matches[0].expert_domain == "Domain B"

    def test_search_empty_experts(self, tmp_path):
        expert = ExpertSystem.init(tmp_path / "empty", domain="Empty")
        expert.invalidate_cache()
        search = MultiExpertSearch([expert])
        matches = search.search("anything")
        assert matches == []

    def test_search_limit(self, tmp_path):
        expert = ExpertSystem.init(tmp_path / "expert", domain="Test")
        expert.beliefs_path.write_text(
            "# Beliefs\n\n"
            "### belief-1\nSSH topic one.\n\n"
            "### belief-2\nSSH topic two.\n\n"
            "### belief-3\nSSH topic three.\n"
        )
        expert.invalidate_cache()
        search = MultiExpertSearch([expert])
        matches = search.search("SSH topic", limit=2)
        assert len(matches) == 2
