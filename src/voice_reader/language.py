import re
from functools import lru_cache

LANGUAGE_CHOICES = [
    ("Auto / 自动", "auto"),
    ("English / 英文", "en"),
    ("中文", "zh"),
    ("日本語 / 日文", "ja"),
    ("Español / 西班牙文", "es"),
    ("العربية / 阿拉伯文", "ar"),
]


@lru_cache(maxsize=1)
def _identifier():
    from langid.langid import LanguageIdentifier, model

    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    identifier.set_languages(["en", "es"])
    return identifier


def resolve_language(text: str, selected: str) -> str:
    if selected not in {code for _, code in LANGUAGE_CHOICES}:
        raise ValueError("Unsupported language selection.")
    if selected != "auto":
        return selected
    if re.search(r"[\u0600-\u06ff]", text):
        return "ar"
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text):
        return "zh"
    return _identifier().classify(text)[0]
