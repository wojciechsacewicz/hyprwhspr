"""Language prompt ids for multilingual Nemotron streaming ASR."""

import re
from typing import Optional

# Canonical prompt_dictionary ids from NVIDIA's multilingual model config.
LANGUAGE_IDS = {
    "en": 0, "en-US": 0, "en-GB": 1,
    "es-ES": 2, "es": 3, "es-US": 3,
    "zh": 4, "zh-CN": 4,
    "hi": 6, "hi-IN": 6,
    "ar": 7, "ar-AR": 7,
    "fr": 8, "fr-FR": 8,
    "de": 9, "de-DE": 9,
    "ja": 10, "ja-JP": 10,
    "ru": 11, "ru-RU": 11,
    "pt-BR": 12, "pt": 13, "pt-PT": 13,
    "ko": 14, "ko-KR": 14,
    "it": 15, "it-IT": 15,
    "nl": 16, "nl-NL": 16,
    "pl": 17, "pl-PL": 17,
    "tr": 18, "tr-TR": 18,
    "uk": 19, "uk-UA": 19,
    "ro": 20, "ro-RO": 20,
    "el": 21, "el-GR": 21,
    "cs": 22, "cs-CZ": 22,
    "hu": 23, "hu-HU": 23,
    "sv": 24, "sv-SE": 24,
    "da": 25, "da-DK": 25,
    "fi": 26, "fi-FI": 26,
    "sk": 28, "sk-SK": 28,
    "hr": 29, "hr-HR": 29,
    "bg": 30, "bg-BG": 30,
    "lt": 31, "lt-LT": 31,
    "th": 32, "th-TH": 32,
    "vi": 33, "vi-VN": 33,
    "et": 60, "et-EE": 60,
    "lv": 61, "lv-LV": 61,
    "sl": 62, "sl-SI": 62,
    "he": 64, "he-IL": 64,
    "fr-CA": 100,
    "auto": 101,
    "mt": 102, "mt-MT": 102,
    "nb": 103, "nb-NO": 103,
    "nn": 104, "nn-NO": 104,
}

_LANGUAGE_TAG_RE = re.compile(r"<[a-z]{2}(?:-[A-Z]{2})?>")


def language_id_for(language: Optional[str]) -> int:
    """Return the model language id, using multilingual auto-detection by default."""
    if not language:
        return LANGUAGE_IDS["auto"]
    normalized = language.strip()
    if normalized in LANGUAGE_IDS:
        return LANGUAGE_IDS[normalized]
    lowered = normalized.lower()
    for candidate, lang_id in LANGUAGE_IDS.items():
        if candidate.lower() == lowered:
            return lang_id
    base = lowered.split("-", 1)[0]
    return LANGUAGE_IDS.get(base, LANGUAGE_IDS["auto"])
