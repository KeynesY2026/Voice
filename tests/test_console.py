import io

from voice_reader import console


def test_console_can_log_all_supported_language_scripts(monkeypatch):
    data = io.BytesIO()
    stream = io.TextIOWrapper(data, encoding="cp1252")
    monkeypatch.setattr(console.sys, "stdout", stream)
    console.configure_utf8_output()
    text = "中文、日本語、العربية"
    print(text, end="")
    stream.flush()
    assert data.getvalue().decode("utf-8") == text
