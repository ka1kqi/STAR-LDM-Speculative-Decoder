"""Unit tests for structure extraction and canonical serialization."""

import pytest
from src.data.structure import extract_structure, canonical_serialize, StructureFeatures


class TestExtractStructure:
    def test_short_text(self):
        feats = extract_structure("Hello world.")
        assert feats.length_bucket == "short"
        assert feats.token_count == 2
        assert not feats.has_list
        assert not feats.has_question

    def test_list_detection(self):
        text = "Items:\n- apples\n- bananas\n- cherries"
        feats = extract_structure(text)
        assert feats.has_list

    def test_question_detection(self):
        text = "What is the meaning of life?"
        feats = extract_structure(text)
        assert feats.has_question

    def test_code_block(self):
        text = "Here is code:\n```python\nprint('hello')\n```\nEnd."
        feats = extract_structure(text)
        assert feats.has_code_block

    def test_length_buckets(self):
        short = extract_structure("one two")
        assert short.length_bucket == "short"

        medium = extract_structure(" ".join(["word"] * 50))
        assert medium.length_bucket == "medium"

        long_text = extract_structure(" ".join(["word"] * 150))
        assert long_text.length_bucket == "long"

        very_long = extract_structure(" ".join(["word"] * 300))
        assert very_long.length_bucket == "very_long"

    def test_section_tags(self):
        text = "# Introduction\nSome text.\n## Methods\nMore text."
        feats = extract_structure(text)
        assert "Introduction" in feats.section_tags
        assert "Methods" in feats.section_tags


class TestCanonicalSerialize:
    def test_basic_serialization(self):
        feats = StructureFeatures(token_count=10, sentence_count=2, length_bucket="short")
        s = canonical_serialize(feats)
        assert "length=short" in s
        assert "tokens=10" in s
        assert "sentences=2" in s

    def test_with_flags(self):
        feats = StructureFeatures(has_list=True, has_question=True, length_bucket="medium")
        s = canonical_serialize(feats)
        assert "list=yes" in s
        assert "question=yes" in s

    def test_with_sections(self):
        feats = StructureFeatures(section_tags=["Intro", "Methods"])
        s = canonical_serialize(feats)
        assert "sections=Intro|Methods" in s

    def test_label_dict(self):
        feats = StructureFeatures(has_list=True, has_question=False, length_bucket="long")
        labels = feats.to_label_dict()
        assert labels["length_bucket"] == 2
        assert labels["has_list"] == 1
        assert labels["has_question"] == 0
