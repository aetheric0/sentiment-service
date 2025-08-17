import pytest
from app.spacy_preprocessor import SpacyPreprocessor


@pytest.fixture(scope="module")
def preproc():
    proc = SpacyPreprocessor()
    proc.fit([])  # load spaCy model once
    return proc


def test_basic_transformation(preproc):
    raw = ["This is a TEST!"]
    cleaned = preproc.transform(raw)
    assert isinstance(cleaned, list)
    assert cleaned[0] == "this test" or "test" in cleaned[0]


def test_removes_stopwords(preproc):
    raw = ["I am very happy with this product"]
    cleaned = preproc.transform(raw)[0]
    assert "am" not in cleaned
    assert "very" not in cleaned


def test_handles_non_ascii(preproc):
    raw = ["El niño comió piña."]
    cleaned = preproc.transform(raw)[0]
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0


def test_empty_string(preproc):
    raw = [""]
    cleaned = preproc.transform(raw)[0]
    assert cleaned == ""
