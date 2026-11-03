"""Real-model backend for the adversarial evals, on the stored
Google AI Studio credential.

Why this exists: the adversarial suites grade the detector against
mock brains, and the mocks cannot decode base64/ROT13/leetspeak —
so every encoded variant lands in scoring's "evaded" bucket and the
one question that matters (does a REAL model act on what evaded?)
stays open. This adapter closes it: same agent loop, same ReAct
prompt, same _parse_react_step — only the brain is a live Gemini
model, reached through the connector surrogate so no raw key ever
touches this process, an env var, or a log.

Two deliberate differences from agent.FreeBackend:
  - NO mock fallback. FreeBackend swaps in the mock's rules when the
    API call fails, which is right for a demo and wrong for an eval:
    a failed call must surface as a failed run, never as a mock run
    wearing a real label.
  - A 429 is fatal, not retried. Free-tier quota is a hard wall;
    retry-looping against it burns the quota and lies about the run.
    generate() raises RateLimitStop and the runner stops, keeping
    whatever partial results it already has.

Auth notes (connector: custom.google-ai-studio): the credential is
attached per request by dynamic_credentials.url_with_surrogate_
query_param — Gemini reads its key from the `key` query param. A
request built without the helper carries nothing and looks exactly
like a bad key, so AuthRejected (401/403 WITH the helper's surrogate
attached) is reported as its own class: it means the stored
credential itself was refused, which is a different problem.

    python3 -c "import sys; sys.path.insert(0, 'evals'); \
        import real_backend; print(real_backend.smoke())"
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import read_json_response, \
    url_with_surrogate_query_param  # noqa: E402

sys.path.insert(0, ".")
from agent import ModelBackend, _parse_react_step, _react_prompt  # noqa: E402
from llm_client import FreeLLMError  # noqa: E402

CREDENTIAL = "custom.google-ai-studio"
ALLOWED_HOSTS = ["generativelanguage.googleapis.com"]
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
# llm_client's free-tier default — checked against the live model
# list on 2026-11-03 before wiring this up; it is available.
DEFAULT_MODEL = "gemini-3.8-flash"

_TIMEOUT = 60
_MIN_INTERVAL = 4.0  # seconds between calls — free-tier pacing


class RateLimitStop(FreeLLMError):
    """429 / quota exhausted: the run stops here, partials stand."""


class AuthRejected(FreeLLMError):
    """401/403 with the surrogate attached: the stored credential
    itself was refused (distinct from never attaching it)."""


class SurrogateGeminiClient:
    """FreeLLMClient's Gemini request shape, credential attached via
    the connector surrogate instead of an env-var key."""

    def __init__(self, model=None, min_interval=_MIN_INTERVAL):
        self.model = model or DEFAULT_MODEL
        self.min_interval = min_interval
        self.calls = 0
        self._last_call = 0.0

    @property
    def label(self):
        return f"gemini:{self.model} (stored credential)"

    def _pace(self):
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def generate(self, prompt, max_tokens=512, temperature=0):
        base = f"{API_BASE}/{self.model}:generateContent"
        url = url_with_surrogate_query_param(
            base, CREDENTIAL, allowed_hosts=ALLOWED_HOSTS)
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens,
                                     "temperature": temperature}}
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        # 5xx and dropped connections get real retries with backoff —
        # gemini-3.8-flash sheds load with 503s and outright
        # disconnects when demand spikes, and one retry wasn't
        # enough. 429 and 401/403 never retry (quota wall / auth
        # verdict — see the class docstrings).
        backoff = (5, 15, 30)
        last = None
        for attempt in range(len(backoff) + 1):
            self._pace()
            try:
                self.calls += 1
                self._last_call = time.monotonic()
                with urllib.request.urlopen(req, timeout=_TIMEOUT) \
                        as resp:
                    return self._parse(read_json_response(resp))
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode()[:300]
                except Exception:
                    pass
                if e.code == 429 or "RESOURCE_EXHAUSTED" in detail:
                    raise RateLimitStop(
                        f"gemini quota/rate limit (HTTP {e.code}) — "
                        "run stopped, partial results stand") from None
                if e.code in (401, 403):
                    raise AuthRejected(
                        f"stored credential refused (HTTP {e.code}): "
                        f"{detail[:160]}") from None
                last = e
                if e.code in (500, 502, 503, 504) \
                        and attempt < len(backoff):
                    time.sleep(backoff[attempt])
                    continue
                raise FreeLLMError(
                    f"gemini rejected the request (HTTP {e.code}): "
                    f"{detail[:160]}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = e
                if attempt < len(backoff):
                    time.sleep(backoff[attempt])
                    continue
                raise FreeLLMError(
                    f"gemini unreachable ({type(e).__name__})") from None
        raise FreeLLMError(f"gemini call failed ({last})") from None

    @staticmethod
    def _parse(payload):
        try:
            parts = payload["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError):
            blocked = (payload.get("promptFeedback") or {}).get(
                "blockReason")
            if blocked:
                raise FreeLLMError(
                    f"gemini blocked the prompt ({blocked})") from None
            raise FreeLLMError(
                "gemini returned an unexpected response shape") from None


class GeminiRealBackend(ModelBackend):
    """The free backend's ReAct text loop on the surrogate client —
    same prompt, same parser, real model, no mock fallback."""

    def __init__(self, client=None, prompt_version=None):
        self.client = client or SurrogateGeminiClient()
        # the cost meter prices by this name; estimates only
        self.model = self.client.model
        self.last_error = None
        self.prompt_version = (prompt_version or
                               os.environ.get("AGENT_PROMPT_VERSION",
                                              "v1"))

    @property
    def label(self):
        return f"gemini-real:{self.model}"

    def think(self, messages, tools):
        text = self.client.generate(
            _react_prompt(messages, tools, version=self.prompt_version),
            max_tokens=512, temperature=0)
        return _parse_react_step(text or "", tools)


def smoke():
    """One tiny live call — proves the credential attaches and the
    default model answers, before an eval run spends real quota."""
    client = SurrogateGeminiClient()
    out = client.generate("Reply with exactly: ok", max_tokens=16,
                          temperature=0)
    return f"{client.label} -> {out.strip()[:40]!r}"
