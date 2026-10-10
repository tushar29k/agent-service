"""Prompt versioning: the ReAct instruction text lives here, one file per
version. `load_prompt("v1")` returns the template with {tool_list} and
{transcript} slots; agent.py's _react_prompt fills them in per think step.

Old versions stay frozen — new prompt ideas get new files, so A/B runs are
reproducible and the eval history stays comparable across versions.
"""
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
_cache = {}  # prompts are frozen once shipped — cache the read


def list_versions():
    """Every prompt version on disk (the .txt files, minus extension)."""
    return sorted(f[:-4] for f in os.listdir(_DIR)
                  if f.endswith(".txt"))


def load_prompt(version="v1"):
    """The template for one version. Raises on unknown versions and on
    templates missing a slot — a broken prompt should fail loudly, not
    silently render half a system prompt."""
    if version in _cache:
        return _cache[version]
    path = os.path.join(_DIR, f"{version}.txt")
    if not os.path.isfile(path):
        raise ValueError(f"unknown prompt version {version!r} — "
                         f"want one of {list_versions()}")
    with open(path) as f:
        tmpl = f.read()
    for slot in ("{tool_list}", "{transcript}"):
        if slot not in tmpl:
            raise ValueError(f"prompt {version!r} is missing its "
                             f"{slot} slot")
    _cache[version] = tmpl
    return tmpl
