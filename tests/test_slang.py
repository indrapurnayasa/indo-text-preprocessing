"""Tests untuk slang.py."""

from indo_text_preprocessing.slang import get_slang_dict, replace_slang


def test_slang_loaded():
    d = get_slang_dict()
    assert len(d) > 3000
    assert d["gak"] == "enggak"


def test_replace_basic():
    assert replace_slang("gak tau emangnya kenapa") == "enggak tahu memangnya kenapa"


def test_no_slang_in_text():
    assert replace_slang("kata baku semua") == "kata baku semua"


def test_custom_mapping_overrides():
    out = replace_slang("gak jadi", mapping={"gak": "tidak"})
    assert out == "tidak jadi"


def test_custom_mapping_adds():
    out = replace_slang("w kalem", mapping={"w": "saya", "kalem": "santai"})
    assert out == "saya santai"


def test_multiword_target():
    d = get_slang_dict()
    multi = [k for k, v in d.items() if " " in v]
    assert len(multi) > 0  # corpus memang ada mapping multi-kata


def test_empty():
    assert replace_slang("") == ""