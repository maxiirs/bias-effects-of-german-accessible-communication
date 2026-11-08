"""Syllable count of the readability formulas, shared by notebooks 01 and 02."""

_DIC = None


def syllable_count(word: str) -> int:
    """Syllables via pyphen de_DE hyphenation (standard approximation, cf. textstat)."""
    global _DIC
    if _DIC is None:
        import pyphen
        _DIC = pyphen.Pyphen(lang="de_DE")
    stripped = "".join(c for c in word if c.isalpha())
    if not stripped:
        return 0
    return _DIC.inserted(stripped).count("-") + 1
