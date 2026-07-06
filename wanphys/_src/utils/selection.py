import functools
from fnmatch import fnmatch
from types import NoneType
from typing import TYPE_CHECKING, Any

import warp as wp
from warp.types import is_array

def match_labels(labels: list[str], pattern: str | list[str] | list[int]) -> list[int]:
    """Find indices of elements in ``labels`` that match ``pattern``.

    See :ref:`label-matching` for the pattern syntax accepted across Newton APIs.

    Args:
        labels: List of label strings to match against.
        pattern: A ``str`` is matched via :func:`fnmatch.fnmatch` against each label.
            A ``list[str]`` matches any pattern.
            A ``list[int]`` is returned as-is (indices used directly).
            Mixing ``str`` and ``int`` in the same list is not allowed.

    Returns:
        Unique list of matching indices, or ``pattern`` itself for ``list[int]``.

    Raises:
        TypeError: If list elements are not all ``str`` or all ``int``.
    """
    if isinstance(pattern, str):
        return [idx for idx, label in enumerate(labels) if fnmatch(label, pattern)]

    if all(isinstance(item, int) for item in pattern):
        return pattern
    if all(isinstance(item, str) for item in pattern):
        return [idx for idx, label in enumerate(labels) if any(fnmatch(label, p) for p in pattern)]
    types = {type(item).__name__ for item in pattern}
    raise TypeError(f"Expected a list of str patterns or a list of int indices, got: {', '.join(sorted(types))}")

