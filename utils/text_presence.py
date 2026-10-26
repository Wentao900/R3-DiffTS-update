"""Whether a text string counts as a present source.

The dataset's missing-text sentinel is the token ``NA``. Exact ``NA`` is
always absent. A retrieval placeholder is that same sentinel followed by an
explanation, for example ``NA (No relevant information found ...)``. A word
that merely begins with those letters, such as ``Napa``, is not the sentinel.
"""


def text_is_absent(text, placeholder_absent=False):
    if text is None:
        return True
    if not isinstance(text, str):
        text = str(text)
    stripped = text.strip()
    if stripped == "" or stripped.upper() == "NA":
        return True
    if not placeholder_absent:
        return False
    upper = stripped.upper()
    return upper.startswith("NA") and len(upper) > 2 and not upper[2].isalnum()
