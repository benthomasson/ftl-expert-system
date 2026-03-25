"""Tests for the expert system core."""

from pathlib import Path

import pytest

from ftl_expert_system.expert import ExpertSystem, _extract_keywords


class TestExtractKeywords:
    def test_removes_stop_words(self):
        kws = _extract_keywords("Does the SSH layer validate host keys?")
        assert "does" not in kws
        assert "the" not in kws
        assert "ssh" in kws
        assert "layer" in kws
        assert "validate" in kws
        assert "host" in kws
        assert "keys" in kws

    def test_short_words_removed(self):
        kws = _extract_keywords("is it ok")
        assert kws == []

    def test_kebab_case(self):
        kws = _extract_keywords("host-key-verification")
        assert "host-key-verification" in kws


class TestExpertSystemInit:
    def test_init_creates_structure(self, tmp_path):
        expert = ExpertSystem.init(tmp_path / "test-expert", domain="Testing")
        assert expert.domain == "Testing"
        assert expert.beliefs_path.exists()
        assert (tmp_path / "test-expert" / ".expert" / "config.json").exists()

    def test_init_with_repo(self, tmp_path):
        repo = tmp_path / "my-repo"
        repo.mkdir()
        expert = ExpertSystem.init(tmp_path / "test-expert", domain="Testing", repo=repo)
        assert expert.repo_path == repo

    def test_load_roundtrip(self, tmp_path):
        expert_dir = tmp_path / "test-expert"
        ExpertSystem.init(expert_dir, domain="Testing")
        loaded = ExpertSystem.load(expert_dir)
        assert loaded.domain == "Testing"


class TestSearchBeliefs:
    def test_search_finds_matching_beliefs(self, tmp_path):
        expert = ExpertSystem.init(tmp_path / "test-expert", domain="Testing")
        expert.beliefs_path.write_text(
            "# Beliefs\n\n"
            "### ssh-host-key-verification\n"
            "SSH host key verification is enabled by default using system known_hosts.\n"
            "- Status: IN\n\n"
            "### policy-engine-active\n"
            "The policy engine validates rules at construction time.\n"
            "- Status: IN\n"
        )
        expert.invalidate_cache()
        matches = expert.search_beliefs("SSH host key verification")
        assert len(matches) >= 1
        assert matches[0]["id"] == "ssh-host-key-verification"

    def test_search_no_matches(self, tmp_path):
        expert = ExpertSystem.init(tmp_path / "test-expert", domain="Testing")
        expert.invalidate_cache()
        matches = expert.search_beliefs("quantum entanglement")
        assert matches == []

    def test_search_ranks_by_score(self, tmp_path):
        expert = ExpertSystem.init(tmp_path / "test-expert", domain="Testing")
        expert.beliefs_path.write_text(
            "# Beliefs\n\n"
            "### ssh-security-full\n"
            "SSH security includes host key verification and command injection prevention.\n"
            "- Status: IN\n\n"
            "### unrelated-belief\n"
            "The module cache stores entries by name.\n"
            "- Status: IN\n\n"
            "### ssh-basics\n"
            "SSH connections use port 22.\n"
            "- Status: IN\n"
        )
        expert.invalidate_cache()
        matches = expert.search_beliefs("SSH host key security verification")
        assert len(matches) >= 2
        # ssh-security-full should score higher (more keyword matches)
        assert matches[0]["id"] == "ssh-security-full"
