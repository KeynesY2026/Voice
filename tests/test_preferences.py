import pytest

from voice_reader.preferences import DEFAULT_PREFERENCES, load_browser_secret, restore_preferences


def test_preferences_restore_text_and_all_controls_without_normalizing_spaces():
    saved = dict(DEFAULT_PREFERENCES, text="第一句。   Second.\nThird.", profile="custom",
                 mode="manual", category="happy", strength=0.25, language="en", device="cpu")
    restored, warning = restore_preferences(saved, {"custom"}, {"auto", "cpu"})
    assert restored == saved
    assert warning == ""


def test_stale_or_invalid_preferences_are_reported_and_use_valid_defaults():
    saved = dict(DEFAULT_PREFERENCES, profile="deleted", strength=float("nan"), language="xx",
                 device="cuda")
    restored, warning = restore_preferences(saved, {"builtin-male"}, {"auto", "cpu"})
    assert restored == DEFAULT_PREFERENCES
    assert warning


def test_browser_secret_persists_across_restarts(tmp_path):
    path = tmp_path / "state.key"
    assert load_browser_secret(path) == load_browser_secret(path)
    assert len(path.read_text()) >= 32


def test_corrupt_browser_secret_errors_explicitly(tmp_path):
    path = tmp_path / "state.key"
    path.write_text("bad")
    with pytest.raises(ValueError, match="key"):
        load_browser_secret(path)


def test_legacy_preferences_never_enable_advanced_mode():
    saved = dict(text="Hello.\nNext.", mode="manual", category="happy", strength=0.8)
    restored, _ = restore_preferences(saved, {"builtin-male"}, {"auto", "cpu"})
    assert restored["advanced"] is False
    assert restored["strength"] == 0.8


def test_advanced_preference_requires_boolean():
    restored, warning = restore_preferences(
        dict(DEFAULT_PREFERENCES, advanced="false"), {"builtin-male"}, {"auto", "cpu"},
    )
    assert restored["advanced"] is False
    assert "advanced" in warning
