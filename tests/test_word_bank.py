from orchestrator.word_bank import sample_words


def test_sample_words_returns_requested_count():
    words = sample_words(20)
    assert len(words) == 20


def test_sample_words_are_unique_and_have_no_example_field():
    words = sample_words(20)
    seen = {w["word"] for w in words}
    assert len(seen) == len(words)
    for w in words:
        assert set(w.keys()) == {"word", "definition"}


def test_sample_words_varies_across_calls():
    first = {w["word"] for w in sample_words(20)}
    second = {w["word"] for w in sample_words(20)}
    assert first != second
