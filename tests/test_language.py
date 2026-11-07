import pytest

from voice_reader.language import resolve_language


@pytest.mark.parametrize("text,expected", [
    ("Hello, this is a technical presentation.", "en"),
    ("这是中文。", "zh"),
    ("这是一个 API demo。", "zh"),
    ("これは日本語です。", "ja"),
    ("Hola, buenos días. Muchas gracias por su ayuda.", "es"),
    ("مرحبا بكم", "ar"),
])
def test_auto_language_matches_text_without_translation(text, expected):
    assert resolve_language(text, "auto") == expected


@pytest.mark.parametrize("language", ["en", "zh", "ja", "es", "ar"])
def test_explicit_language_overrides_text_detection(language):
    assert resolve_language("Hello，这是混合文本。", language) == language


def test_unsupported_language_is_rejected():
    with pytest.raises(ValueError, match="language"):
        resolve_language("Hello.", "invalid")


def test_untrained_mixed_language_code_is_rejected():
    with pytest.raises(ValueError, match="language"):
        resolve_language("Hello，这是中文。", "zhen")
