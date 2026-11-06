#!/usr/bin/env python3
"""decision_client.py — a client for decision models served at ``POST /v1/systemone``.

Owner: atrium-page-classification (issue #58; umbrella ufal/atrium-project#83). Copies in
atrium-ocr-postprocess, atrium-digital-convert and atrium-keyword-extract are pinned by SHA-256
(``tests/test_vendored_decision_parity.py``, ``VENDORED`` + ``ADOPTED``); change it here and
re-vendor (atrium-project ``docs/shared_components.md``).

A decision model (Clef, Clef-Flash, Strands Decider 2B) reads a *state* (text or JSON, optionally
images) and a set of typed *questions*, and returns a probability for every allowed option of every
question in one forward pass. It generates no text. The wire format, as the Clef model card and its
``joint_schema_model.systemone`` give it::

    request  {"model": str, "state": <str | JSON>, "questions": {id: question}, "images"?: [...]}
    question {"type": "choice", "instructions"?: str, "criteria": {option_id: description | null}}
             {"type": "score",  "instructions"?: str, "criteria": [description, ...]}
             {"type": "noul",   "instructions"?: str, "criteria"?: {"true": str, "false": str}}
    response {"model": str, "answers": {id: answer}, "usage": {"input_tokens": int, ...}}
    answer   {"type": "choice", "choice": id, "confidence": p, "probabilities": {id: p}}
             {"type": "score", "score": E, "confidence": p, "legend": {...}, "probabilities": {"0": p, ...}}
             {"type": "noul", "noul": p_true}

How images travel over HTTP is the server's choice and is not in the card; ``IMAGE_STYLES`` lists the
forms this client can send, and ``--probe`` finds out which one a server reads.

What the client guarantees:

* ``DecisionClient.decide()`` never raises. It returns a ``Decision`` whose ``status`` is one of
  ``ok``, ``invalid``, ``timeout``, ``error``, ``rejected`` or ``not_configured`` (the status set of
  atrium-digital-convert's stage connector, ``service/stages.py:_post``).
* Every answer is validated against its question. An answer outside the options, probabilities that
  do not cover the options or do not sum to about 1, or a missing answer makes the whole decision
  ``invalid``. Nothing is repaired.
* Each call records its latency, ``usage.input_tokens``, the HTTP status, the ``Server`` header and
  the question set's digest (``schema_digest``), which serves as the prompt version.

It needs only the standard library; Pillow is imported only to encode images. The command line::

    python3 decision_client.py --selftest
    python3 decision_client.py --probe --model clef-oQ8e --image a.png --image b.png --out probe.json

``--url``, ``--model`` and ``--timeout`` default to ``DECISION_MODEL_URL``, ``DECISION_MODEL_NAME`` and
``DECISION_TIMEOUT_S``. The probe report names the endpoint as ``$DECISION_MODEL_URL``, never the host.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__version__ = "0.1.0"

#: Limits of the Jev/SystemOne interface, checked before a request is sent.
MIN_OPTIONS = 2
MAX_OPTIONS = 255
MAX_QUESTIONS = 64
MAX_IMAGES = 4

DEFAULT_PATH = "/v1/systemone"
DEFAULT_TIMEOUT_S = 60.0
QUESTION_TYPES = ("noul", "choice", "score")
STATUSES = ("ok", "invalid", "timeout", "error", "rejected", "not_configured")

#: The wire forms of one image: a ``data:`` URI string, its bare base64, ``{"url": <data URI>}``, or
#: the OpenAI-style ``{"type": "image_url", "image_url": {"url": <data URI>}}``.
IMAGE_STYLES = ("data_uri", "base64", "url", "image_url")

USER_AGENT = f"atrium-decision-client/{__version__}"


class DecisionInputError(ValueError):
    """A question set or an image that the interface cannot carry (raised by the builders only)."""


# ── question builders ─────────────────────────────────────────────────────────────────────────


def choice(instructions: Optional[str], criteria: Mapping[str, Optional[str]]) -> Dict[str, Any]:
    """A ``choice`` question: one of 2–255 named options, each with a description or ``None``."""
    question: Dict[str, Any] = {"type": "choice", "criteria": {str(k): v for k, v in criteria.items()}}
    if instructions:
        question["instructions"] = instructions
    _check_question("choice", question)
    return question


def score(instructions: Optional[str], levels: Sequence[str]) -> Dict[str, Any]:
    """A ``score`` question: an ordered list of 2–255 level descriptions, indexed from 0."""
    question: Dict[str, Any] = {"type": "score", "criteria": [str(level) for level in levels]}
    if instructions:
        question["instructions"] = instructions
    _check_question("score", question)
    return question


def noul(instructions: Optional[str], true: Optional[str] = None, false: Optional[str] = None) -> Dict[str, Any]:
    """A ``noul`` (yes/no) question, optionally with descriptions of what true and false mean."""
    question: Dict[str, Any] = {"type": "noul"}
    if instructions:
        question["instructions"] = instructions
    criteria = {key: value for key, value in (("true", true), ("false", false)) if value}
    if criteria:
        question["criteria"] = criteria
    return question


def questions(items: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Check a whole question set and return it as a plain dict (insertion order kept)."""
    out = {str(qid): dict(question) for qid, question in items.items()}
    validate_questions(out)
    return out


def options_of(question: Mapping[str, Any]) -> List[str]:
    """The option ids a question allows, in the order the request lists them."""
    qtype = question.get("type")
    if qtype == "noul":
        return ["true", "false"]
    if qtype == "choice":
        return [str(key) for key in question["criteria"]]
    return [str(index) for index in range(len(question["criteria"]))]


def _check_question(qid: str, question: Mapping[str, Any]) -> None:
    qtype = question.get("type")
    if qtype not in QUESTION_TYPES:
        raise DecisionInputError(f"{qid}: type must be one of {QUESTION_TYPES}, not {qtype!r}")
    if qtype == "noul":
        extra = set(question.get("criteria") or {}) - {"true", "false"}
        if extra:
            raise DecisionInputError(
                f"{qid}: a noul question takes criteria for 'true'/'false' only, not {sorted(extra)}"
            )
        return
    criteria = question.get("criteria")
    if qtype == "choice" and not isinstance(criteria, Mapping):
        raise DecisionInputError(f"{qid}: a choice question needs criteria as {{option_id: description}}")
    if qtype == "score" and not isinstance(criteria, (list, tuple)):
        raise DecisionInputError(f"{qid}: a score question needs criteria as a list of level descriptions")
    n = len(criteria)
    if not MIN_OPTIONS <= n <= MAX_OPTIONS:
        raise DecisionInputError(f"{qid}: {n} options; a {qtype} question takes {MIN_OPTIONS}–{MAX_OPTIONS}")
    if qtype == "choice" and any(not str(key) for key in criteria):
        raise DecisionInputError(f"{qid}: an option id is empty")


def validate_questions(items: Mapping[str, Mapping[str, Any]]) -> None:
    """Raise ``DecisionInputError`` unless the set has 1–64 well-formed questions."""
    if not isinstance(items, Mapping) or not items:
        raise DecisionInputError("at least one question is required")
    if len(items) > MAX_QUESTIONS:
        raise DecisionInputError(f"{len(items)} questions; a request takes at most {MAX_QUESTIONS}")
    for qid, question in items.items():
        if not str(qid):
            raise DecisionInputError("a question id is empty")
        _check_question(str(qid), question)


def schema_digest(items: Mapping[str, Mapping[str, Any]]) -> str:
    """SHA-256 of the canonical JSON of a question set: the version of the prompt it is."""
    canonical = json.dumps(items, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── images ────────────────────────────────────────────────────────────────────────────────────


def image_data_uri(image: Any, max_side: int = 1024, quality: int = 90) -> str:
    """A JPEG ``data:`` URI of *image* (a path, bytes or a PIL image), its long side at most *max_side*.

    A page scan at full resolution fills the model's context with image tokens, so every image is
    downscaled (never upscaled) before it is sent. Pillow is imported only here."""
    from PIL import Image  # noqa: PLC0415 — only callers that send images need Pillow

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            picture = opened.convert("RGB")
    elif isinstance(image, (bytes, bytearray)):
        with Image.open(io.BytesIO(image)) as opened:
            picture = opened.convert("RGB")
    elif hasattr(image, "convert"):
        picture = image.convert("RGB")
    else:
        raise DecisionInputError(f"cannot encode an image of type {type(image).__name__}")
    if max_side and max(picture.size) > max_side:
        picture.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def image_on_wire(data_uri: str, style: str = "data_uri") -> Any:
    """One image in the wire form *style* (see ``IMAGE_STYLES``)."""
    if style == "data_uri":
        return data_uri
    if style == "base64":
        return data_uri.split(",", 1)[1]
    if style == "url":
        return {"url": data_uri}
    if style == "image_url":
        return {"type": "image_url", "image_url": {"url": data_uri}}
    raise DecisionInputError(f"unknown image style {style!r}; one of {IMAGE_STYLES}")


# ── results ───────────────────────────────────────────────────────────────────────────────────


@dataclass
class Answer:
    """One validated answer. ``probabilities`` covers every option; for ``noul`` they are true/false."""

    qid: str
    type: str
    probabilities: Dict[str, float]
    choice: Optional[str] = None
    confidence: Optional[float] = None
    p_true: Optional[float] = None
    score: Optional[float] = None

    def best(self) -> Tuple[str, float]:
        """The most probable option and its probability."""
        option = max(self.probabilities, key=self.probabilities.__getitem__)
        return option, self.probabilities[option]


@dataclass
class Decision:
    """The outcome of one request. ``answers`` is filled only when ``status == "ok"``."""

    status: str
    schema_digest: str
    answers: Dict[str, Answer] = field(default_factory=dict)
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    total_ms: Optional[float] = None
    attempts: int = 0
    http_status: Optional[int] = None
    input_tokens: Optional[int] = None
    server: Optional[str] = None
    error: Optional[str] = None
    raw: Any = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self, with_raw: bool = False) -> Dict[str, Any]:
        out = asdict(self)
        if not with_raw:
            out.pop("raw", None)
        return out


def _probability(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _sum_tolerance(n_options: int) -> float:
    # Servers round each probability to 4 decimals (Clef's systemone_answer), so the sum of n
    # rounded values may drift by up to n * 0.5e-4.
    return 1e-3 + n_options * 5e-5


def validate_answers(items: Mapping[str, Mapping[str, Any]], body: Any) -> Tuple[Dict[str, Answer], Optional[str]]:
    """Check a response body against its questions; returns ``(answers, None)`` or ``({}, reason)``."""
    if not isinstance(body, Mapping):
        return {}, f"the response is a JSON {type(body).__name__}, not an object"
    raw_answers = body.get("answers")
    if not isinstance(raw_answers, Mapping):
        return {}, "the response has no 'answers' object"
    missing = sorted(set(items) - set(raw_answers))
    unexpected = sorted(set(raw_answers) - set(items))
    if missing or unexpected:
        return {}, f"answers do not match the questions (missing {missing}, unexpected {unexpected})"
    answers: Dict[str, Answer] = {}
    for qid, question in items.items():
        answer, reason = _validate_one(qid, question, raw_answers[qid])
        if reason:
            return {}, f"{qid}: {reason}"
        answers[qid] = answer
    return answers, None


def _validate_one(qid: str, question: Mapping[str, Any], raw: Any) -> Tuple[Optional[Answer], Optional[str]]:
    qtype = question["type"]
    if isinstance(raw, Mapping) and raw.get("type") not in (None, qtype):
        return None, f"answer type {raw.get('type')!r} for a {qtype} question"
    if qtype == "noul":
        value = raw.get("noul", raw.get("probability")) if isinstance(raw, Mapping) else raw
        p_true = _probability(value)
        if p_true is None:
            return None, f"P(true) {value!r} is not a probability"
        return Answer(qid, "noul", {"true": p_true, "false": 1.0 - p_true}, p_true=p_true), None
    if not isinstance(raw, Mapping):
        return None, f"a {qtype} answer must be an object"
    options = options_of(question)
    probabilities = raw.get("probabilities")
    if not isinstance(probabilities, Mapping):
        return None, "no probabilities"
    keys = {str(key) for key in probabilities}
    if keys != set(options):
        return None, f"probabilities cover {sorted(keys)}, not the options {sorted(options)}"
    by_key = {str(key): value for key, value in probabilities.items()}
    clean: Dict[str, float] = {}
    for option in options:
        p = _probability(by_key.get(option))
        if p is None:
            return None, f"P({option}) is not a probability"
        clean[option] = p
    total = sum(clean.values())
    if abs(total - 1.0) > _sum_tolerance(len(options)):
        return None, f"probabilities sum to {total:.4f}, not 1"
    confidence = _probability(raw.get("confidence")) if raw.get("confidence") is not None else None
    if qtype == "choice":
        picked = raw.get("choice")
        if picked is None or str(picked) not in clean:
            return None, f"choice {picked!r} is not one of the options"
        picked = str(picked)
        if clean[picked] < max(clean.values()) - 1e-3:
            return None, f"choice {picked!r} is not the most probable option"
        return Answer(qid, "choice", clean, choice=picked, confidence=confidence or clean[picked]), None
    try:
        expected = float(raw.get("score"))
    except (TypeError, ValueError):
        return None, f"score {raw.get('score')!r} is not a number"
    if not -1e-6 <= expected <= len(options) - 1 + 1e-6:
        return None, f"score {expected} is outside 0–{len(options) - 1}"
    return Answer(qid, "score", clean, confidence=confidence or max(clean.values()), score=expected), None


# ── the client ────────────────────────────────────────────────────────────────────────────────


class DecisionClient:
    """Sends decision requests to one server and model. ``decide()`` never raises."""

    def __init__(
        self,
        url: Optional[str],
        model: Optional[str],
        timeout_s: float = DEFAULT_TIMEOUT_S,
        retries: int = 2,
        path: str = DEFAULT_PATH,
        image_style: str = "data_uri",
        use_env_proxy: bool = True,
        backoff_s: float = 0.5,
    ) -> None:
        self.url = (url or "").strip() or None
        self.model = (model or "").strip() or None
        self.timeout_s = float(timeout_s)
        self.retries = max(0, int(retries))
        self.path = path if path.startswith("/") else "/" + path
        self.image_style = image_style
        self.backoff_s = backoff_s
        handlers = [] if use_env_proxy else [urllib.request.ProxyHandler({})]
        self._opener = urllib.request.build_opener(*handlers)

    @property
    def endpoint(self) -> Optional[str]:
        if not self.url:
            return None
        base = self.url.rstrip("/")
        return base if base.endswith(self.path.rstrip("/")) else base + self.path

    def request_body(
        self,
        state: Any,
        items: Mapping[str, Mapping[str, Any]],
        images: Optional[Sequence[Any]] = None,
        max_side: int = 1024,
    ) -> Dict[str, Any]:
        """The JSON body of one request (raises ``DecisionInputError`` on a malformed set)."""
        validate_questions(items)
        body: Dict[str, Any] = {"model": self.model or "", "state": state, "questions": dict(items)}
        if images:
            if len(images) > MAX_IMAGES:
                raise DecisionInputError(f"{len(images)} images; a request takes at most {MAX_IMAGES}")
            encoded = []
            for image in images:
                uri = image if isinstance(image, str) and image.startswith("data:") else image_data_uri(image, max_side)
                encoded.append(image_on_wire(uri, self.image_style))
            body["images"] = encoded
        return body

    def decide(
        self,
        state: Any,
        items: Mapping[str, Mapping[str, Any]],
        images: Optional[Sequence[Any]] = None,
        max_side: int = 1024,
    ) -> Decision:
        digest = schema_digest(items) if isinstance(items, Mapping) else ""
        if not self.endpoint or not self.model:
            missing = "a URL" if not self.endpoint else "a model name"
            return Decision("not_configured", digest, error=f"no decision server: {missing} is not set")
        if not self.endpoint.startswith(("http://", "https://")):
            return Decision("not_configured", digest, model=self.model, error="the URL must start with http(s)://")
        try:
            body = self.request_body(state, items, images, max_side)
        except (DecisionInputError, OSError) as exc:
            return Decision("rejected", digest, model=self.model, error=f"client: {exc}")
        except Exception as exc:  # an image Pillow cannot read, an unserialisable state, ...
            return Decision("rejected", digest, model=self.model, error=f"client: {type(exc).__name__}: {exc}")
        try:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            return Decision("rejected", digest, model=self.model, error=f"client: the state is not JSON: {exc}")
        return self._send(payload, items, digest)

    def _send(self, payload: bytes, items: Mapping[str, Mapping[str, Any]], digest: str) -> Decision:
        started = time.perf_counter()
        attempts = 0
        last = Decision("error", digest, model=self.model, error="no attempt made")
        while attempts <= self.retries:
            attempts += 1
            last = self._attempt(payload, items, digest)
            last.attempts = attempts
            last.total_ms = round((time.perf_counter() - started) * 1000.0, 1)
            if not getattr(last, "_retry", False) or attempts > self.retries:
                break
            time.sleep(min(self.backoff_s * (2 ** (attempts - 1)), 8.0))
        last.__dict__.pop("_retry", None)
        return last

    def _attempt(self, payload: bytes, items: Mapping[str, Mapping[str, Any]], digest: str) -> Decision:
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT},
        )
        t0 = time.perf_counter()
        decision = Decision("error", digest, model=self.model)
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                status, text, server = response.status, response.read(), response.headers.get("Server")
        except urllib.error.HTTPError as exc:
            decision.latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            decision.http_status = exc.code
            decision.server = exc.headers.get("Server") if exc.headers else None
            detail = _detail(exc)
            if exc.code in (404, 405):
                decision.error = f"the server has no {self.path} route ({exc.code}): {detail}"
            elif exc.code == 429 or exc.code >= 500:
                decision.error = f"the server answered {exc.code}: {detail}"
                decision._retry = True  # type: ignore[attr-defined]
            else:
                decision.status = "rejected"
                decision.error = f"the server refused the request ({exc.code}): {detail}"
            return decision
        except (socket.timeout, TimeoutError) as exc:
            decision.status = "timeout"
            decision.error = f"no answer within {self.timeout_s:g} s ({exc})"
            return decision
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, (socket.timeout, TimeoutError)):
                decision.status = "timeout"
                decision.error = f"no answer within {self.timeout_s:g} s"
            else:
                decision.error = f"cannot reach the server: {reason}"
                decision._retry = True  # type: ignore[attr-defined]
            return decision
        except (OSError, ValueError) as exc:  # a reset connection, a malformed URL, ...
            decision.error = f"cannot reach the server: {type(exc).__name__}: {exc}"
            decision._retry = True  # type: ignore[attr-defined]
            return decision

        decision.latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        decision.http_status = status
        decision.server = server
        try:
            body = json.loads(text.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            decision.status = "invalid"
            decision.error = f"the server answered {status} with a body that is not JSON"
            decision.raw = text[:2000].decode("utf-8", "replace")
            return decision
        decision.raw = body
        usage = body.get("usage") if isinstance(body, Mapping) else None
        if isinstance(usage, Mapping) and isinstance(usage.get("input_tokens"), (int, float)):
            decision.input_tokens = int(usage["input_tokens"])
        answers, reason = validate_answers(items, body)
        if reason:
            decision.status = "invalid"
            decision.error = reason
            return decision
        decision.status = "ok"
        decision.answers = answers
        if isinstance(body, Mapping) and isinstance(body.get("model"), str):
            decision.model = body["model"]
        return decision


def _detail(exc: urllib.error.HTTPError) -> str:
    try:
        text = exc.read().decode("utf-8", "replace").strip()
    except Exception:  # noqa: BLE001 — the detail is best effort
        return "(no body)"
    return (text[:300] + "…") if len(text) > 300 else (text or "(no body)")


# ── lanes and metrics ─────────────────────────────────────────────────────────────────────────


def lane(probability: float, accept_at: float, review_below: float) -> str:
    """``accept`` at or above *accept_at*, ``review`` below *review_below*, ``escalate`` in between.

    The thresholds belong to one model revision and one question wording: set them on dev data."""
    if review_below > accept_at:
        raise ValueError(f"review_below ({review_below}) must not exceed accept_at ({accept_at})")
    if probability >= accept_at:
        return "accept"
    if probability < review_below:
        return "review"
    return "escalate"


def brier(probabilities: Sequence[Mapping[str, float]], golds: Sequence[str]) -> float:
    """Mean multi-class Brier score: Σ over options of (p − 1[option is gold])², averaged over items."""
    if len(probabilities) != len(golds) or not golds:
        raise ValueError("brier needs one gold per probability map, and at least one item")
    total = 0.0
    for probs, gold in zip(probabilities, golds):
        total += sum((p - (1.0 if option == gold else 0.0)) ** 2 for option, p in probs.items())
    return total / len(golds)


def ece(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    """Expected calibration error over *bins* equal-width bins of (0, 1] (0 falls in the first)."""
    if len(confidences) != len(correct) or not correct:
        raise ValueError("ece needs one correctness flag per confidence, and at least one item")
    sums = [[0, 0.0, 0.0] for _ in range(bins)]  # count, Σ confidence, Σ correct
    for conf, hit in zip(confidences, correct):
        index = min(bins - 1, max(0, math.ceil(conf * bins) - 1))
        sums[index][0] += 1
        sums[index][1] += conf
        sums[index][2] += 1.0 if hit else 0.0
    n = len(correct)
    return sum(abs(s[2] / s[0] - s[1] / s[0]) * s[0] / n for s in sums if s[0])


def coverage_curve(confidences: Sequence[float], correct: Sequence[bool]) -> List[Tuple[float, float, float]]:
    """``(threshold, coverage, error)`` for accepting every item with confidence ≥ threshold."""
    if len(confidences) != len(correct):
        raise ValueError("one correctness flag per confidence")
    pairs = sorted(zip(confidences, correct), key=lambda pair: -pair[0])
    curve: List[Tuple[float, float, float]] = []
    wrong = 0
    n = len(pairs)
    for index, (conf, hit) in enumerate(pairs, start=1):
        wrong += 0 if hit else 1
        if index == n or pairs[index][0] != conf:
            curve.append((conf, index / n, wrong / index))
    return curve


def pick_threshold(
    confidences: Sequence[float], correct: Sequence[bool], target_error: float
) -> Optional[Tuple[float, float, float]]:
    """The lowest threshold whose accepted items have error ≤ *target_error* (the widest coverage)."""
    best = None
    for threshold, coverage, error in coverage_curve(confidences, correct):
        if error <= target_error:
            best = (threshold, coverage, error)
    return best


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for *b* and *c* discordant pairs (binomial, p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


# ── a fake server, for the selftest and for tests ─────────────────────────────────────────────


def _stable_unit(*parts: Any) -> float:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def reference_answers(request: Mapping[str, Any]) -> Dict[str, Any]:
    """Answers in the shapes of Clef's ``systemone_answer``, from deterministic pseudo-probabilities.

    They depend on the state, the images (as sent) and the question, nothing else."""
    state_key = json.dumps(request.get("state"), sort_keys=True, ensure_ascii=False)
    images_key = hashlib.sha256(json.dumps(request.get("images") or [], sort_keys=True).encode("utf-8")).hexdigest()
    answers: Dict[str, Any] = {}
    for qid, question in request["questions"].items():
        options = options_of(question)
        weights = [math.exp(3.0 * _stable_unit(state_key, images_key, qid, option)) for option in options]
        total = sum(weights)
        probs = {option: w / total for option, w in zip(options, weights)}
        if question["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": round(probs["true"], 4)}
        elif question["type"] == "choice":
            picked = max(options, key=probs.__getitem__)
            answers[qid] = {
                "type": "choice",
                "choice": picked,
                "confidence": round(probs[picked], 4),
                "probabilities": {option: round(probs[option], 4) for option in options},
            }
        else:
            answers[qid] = {
                "type": "score",
                "score": round(sum(i * probs[o] for i, o in enumerate(options)), 4),
                "confidence": round(max(probs.values()), 4),
                "legend": dict(zip(options, question["criteria"])),
                "probabilities": {option: round(probs[option], 4) for option in options},
            }
    return answers


Responder = Callable[[Dict[str, Any]], Tuple[int, Any]]


def default_responder(request: Dict[str, Any]) -> Tuple[int, Any]:
    """A well-behaved ``/v1/systemone``: the reference checks, then ``reference_answers``."""
    items = request.get("questions")
    if not isinstance(request.get("model"), str) or "state" not in request:
        return 400, {"error": "model and state are required"}
    if not isinstance(items, dict) or not items:
        return 400, {"error": "at least one question is required"}
    try:
        validate_questions(items)
    except DecisionInputError as exc:
        return 400, {"error": str(exc)}
    tokens = len(json.dumps(request, ensure_ascii=False)) // 4
    return 200, {
        "model": request["model"],
        "answers": reference_answers(request),
        "usage": {"input_tokens": tokens, "output_tokens": 0},
    }


class FakeSystemOne:
    """An in-process ``POST /v1/systemone`` on 127.0.0.1 (a free port), for selftests and tests.

    ``responder(request) -> (status, body)``; a ``bytes`` body is sent as is, anything else as JSON.
    Every request body received is kept in ``requests``."""

    def __init__(self, responder: Responder = default_responder, path: str = DEFAULT_PATH) -> None:
        self.responder = responder
        self.path = path
        self.requests: List[Dict[str, Any]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "FakeSystemOne/1"

            def log_message(self, *args: Any) -> None:  # quiet
                return

            def do_POST(self) -> None:  # noqa: N802 — the http.server API
                if self.path != fake.path:
                    self._reply(404, {"error": "no such route"})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    request = json.loads(self.rfile.read(length).decode("utf-8"))
                except ValueError:
                    self._reply(400, {"error": "the body is not JSON"})
                    return
                fake.requests.append(request)
                status, body = fake.responder(request)
                self._reply(status, body)

            def _reply(self, status: int, body: Any) -> None:
                data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):  # the client gave up (a timeout test)
                    pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "FakeSystemOne":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()


# ── selftest ──────────────────────────────────────────────────────────────────────────────────

_CARD_STATE = "Our checkout started returning errors and orders are blocked."


def card_questions() -> Dict[str, Dict[str, Any]]:
    """The three questions of the Clef model card's ``systemone`` example."""
    return questions(
        {
            "department": choice(
                "Which team should handle the message?",
                {"billing": "Payments or invoices", "technical": "Bugs or outages"},
            ),
            "urgency": score(None, ["Can wait", "This week", "Today"]),
            "outage": noul("Is a service down?"),
        }
    )


def selftest() -> int:
    """Every status against ``FakeSystemOne``; prints one line per case, returns the exit code."""
    items = card_questions()
    failures = 0

    def check(name: str, condition: bool, decision: Optional[Decision] = None) -> None:
        nonlocal failures
        failures += 0 if condition else 1
        extra = f" — {decision.status}: {decision.error}" if decision is not None and not condition else ""
        print(f"{'PASS' if condition else 'FAIL'} {name}{extra}")

    with FakeSystemOne() as fake:
        client = DecisionClient(fake.url, "fake", timeout_s=5, retries=0)
        d = client.decide(_CARD_STATE, items)
        check("ok: the card's request is answered and validated", d.ok and set(d.answers) == set(items), d)
        check("ok: usage.input_tokens is recorded", d.input_tokens is not None and d.input_tokens > 0, d)
        check("ok: the digest is the question set's", d.schema_digest == schema_digest(items), d)

    def bad_choice(request: Dict[str, Any]) -> Tuple[int, Any]:
        status, body = default_responder(request)
        body["answers"]["department"]["choice"] = "sales"
        return status, body

    def bad_sum(request: Dict[str, Any]) -> Tuple[int, Any]:
        status, body = default_responder(request)
        body["answers"]["department"]["probabilities"] = {"billing": 0.7, "technical": 0.7}
        return status, body

    def missing(request: Dict[str, Any]) -> Tuple[int, Any]:
        status, body = default_responder(request)
        del body["answers"]["outage"]
        return status, body

    for name, responder in (
        ("a choice outside the options", bad_choice),
        ("probabilities not summing to 1", bad_sum),
        ("a missing answer", missing),
        ("a body that is not JSON", lambda r: (200, b"<html>")),
    ):
        with FakeSystemOne(responder) as fake:
            d = DecisionClient(fake.url, "fake", timeout_s=5, retries=0).decide(_CARD_STATE, items)
            check(f"invalid: {name}", d.status == "invalid", d)

    with FakeSystemOne(lambda r: (400, {"error": "bad request"})) as fake:
        d = DecisionClient(fake.url, "fake", timeout_s=5, retries=0).decide(_CARD_STATE, items)
        check("rejected: a 400 from the server", d.status == "rejected" and d.http_status == 400, d)

    calls = {"n": 0}

    def flaky(request: Dict[str, Any]) -> Tuple[int, Any]:
        calls["n"] += 1
        return (503, {"error": "busy"}) if calls["n"] == 1 else default_responder(request)

    with FakeSystemOne(flaky) as fake:
        d = DecisionClient(fake.url, "fake", timeout_s=5, retries=2, backoff_s=0.01).decide(_CARD_STATE, items)
        check("ok after a retry: one 503, then an answer", d.ok and d.attempts == 2, d)

    with FakeSystemOne(lambda r: (503, {"error": "down"})) as fake:
        d = DecisionClient(fake.url, "fake", timeout_s=5, retries=1, backoff_s=0.01).decide(_CARD_STATE, items)
        check("error: 503 on every attempt", d.status == "error" and d.attempts == 2, d)

    with FakeSystemOne(path="/v2/other") as fake:
        d = DecisionClient(fake.url, "fake", timeout_s=5, retries=2).decide(_CARD_STATE, items)
        check("error: no /v1/systemone route, not retried", d.status == "error" and d.attempts == 1, d)

    def slow(request: Dict[str, Any]) -> Tuple[int, Any]:
        time.sleep(1.5)
        return default_responder(request)

    with FakeSystemOne(slow) as fake:
        d = DecisionClient(fake.url, "fake", timeout_s=0.3, retries=0).decide(_CARD_STATE, items)
        check("timeout: no answer within the timeout", d.status == "timeout", d)

    d = DecisionClient(None, "fake").decide(_CARD_STATE, items)
    check("not_configured: no URL", d.status == "not_configured", d)
    d = DecisionClient("http://127.0.0.1:9", "fake", timeout_s=2, retries=0).decide(_CARD_STATE, {"q": {"type": "x"}})
    check("rejected (client): a malformed question set is never sent", d.status == "rejected", d)
    d = DecisionClient("http://127.0.0.1:9", "fake", timeout_s=2, retries=0).decide(_CARD_STATE, items)
    check("error: nothing listens on the port", d.status == "error", d)

    print(f"{'OK' if not failures else 'FAILED'}: {failures} failure(s)")
    return 0 if not failures else 1


# ── probe ─────────────────────────────────────────────────────────────────────────────────────


def _summary(decision: Decision) -> Dict[str, Any]:
    out = decision.to_dict()
    out["answers"] = {
        qid: {"choice": a.choice, "p_true": a.p_true, "score": a.score, "best": list(a.best())}
        for qid, a in decision.answers.items()
    }
    return out


def probe(client: DecisionClient, images: Sequence[str], max_side: int = 1024) -> Dict[str, Any]:
    """Four checks of a live server; returns a JSON-ready report without the server's address.

    1. ``card_noul``: the model card's curl request (one ``noul``).
    2. ``card_mixed``: the card's choice + score + noul example.
    3. ``images``: one image-dependent question per image and image style, and once without an
       image. An image style "is read" when it changes P(true) by more than 0.05 for some image.
    4. ``limits``: one request at the interface's limits (a 255-option choice and 63 yes/no questions).
    """
    steps: Dict[str, Any] = {}
    noul_only = questions({"urgency": noul("Does this message express urgency?")})
    steps["card_noul"] = _summary(client.decide("My Stripe integration keeps failing. Please help ASAP.", noul_only))
    steps["card_mixed"] = _summary(client.decide(_CARD_STATE, card_questions()))

    image_report: Dict[str, Any] = {"skipped": not images}
    if images:
        page_q = questions({"photo_main": noul("Is a photograph the main content of the attached page image?")})
        state = {"task": "Look at the attached page image and answer the question."}
        baseline = client.decide(state, page_q)
        image_report["without_image"] = _summary(baseline)
        p0 = baseline.answers["photo_main"].p_true if baseline.ok else None
        image_report["styles"] = {}
        original_style = client.image_style
        for style in IMAGE_STYLES:
            client.image_style = style
            rows = []
            for path in images:
                d = client.decide(state, page_q, images=[path], max_side=max_side)
                p = d.answers["photo_main"].p_true if d.ok else None
                rows.append(
                    {
                        "image": Path(path).name,
                        "status": d.status,
                        "http_status": d.http_status,
                        "p_true": p,
                        "latency_ms": d.latency_ms,
                        "input_tokens": d.input_tokens,
                        "error": d.error,
                    }
                )
            changed = [r for r in rows if r["p_true"] is not None and p0 is not None and abs(r["p_true"] - p0) > 0.05]
            spread = [r["p_true"] for r in rows if r["p_true"] is not None]
            image_report["styles"][style] = {
                "rows": rows,
                "read": bool(changed) or (len(spread) > 1 and max(spread) - min(spread) > 0.05),
            }
        client.image_style = original_style

    big = {
        "big_choice": choice("Which option is it?", {f"opt{i:03d}": f"option number {i}" for i in range(MAX_OPTIONS)})
    }
    for i in range(MAX_QUESTIONS - 1):
        big[f"flag{i:02d}"] = noul(f"Is statement {i} true?")
    steps["limits"] = _summary(client.decide({"statements": "none of them"}, questions(big)))

    servers = {s.get("server") for s in steps.values() if s.get("server")}
    return {
        "client_version": __version__,
        "endpoint": "$DECISION_MODEL_URL" + client.path,
        "model": client.model,
        "server_header": sorted(servers),
        "probed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "steps": steps,
        "images": image_report,
        "image_styles_read": sorted(s for s, r in image_report.get("styles", {}).items() if r["read"]),
    }


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true", help="every status against an in-process fake server")
    mode.add_argument("--probe", action="store_true", help="four checks of a live server (see probe())")
    parser.add_argument(
        "--url", default=os.environ.get("DECISION_MODEL_URL"), help="server base URL ($DECISION_MODEL_URL)"
    )
    parser.add_argument(
        "--model", default=os.environ.get("DECISION_MODEL_NAME"), help="model name ($DECISION_MODEL_NAME)"
    )
    parser.add_argument("--timeout", type=float, default=_env_float("DECISION_TIMEOUT_S", DEFAULT_TIMEOUT_S))
    parser.add_argument("--path", default=DEFAULT_PATH, help=f"the route (default {DEFAULT_PATH})")
    parser.add_argument("--image", action="append", default=[], help="a page image for the image check (repeatable)")
    parser.add_argument("--max-side", type=int, default=1024, help="long side of sent images, in pixels")
    parser.add_argument("--no-proxy", action="store_true", help="ignore HTTP(S)_PROXY for this server")
    parser.add_argument("--out", help="write the probe report (JSON) here")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()
    client = DecisionClient(
        args.url, args.model, timeout_s=args.timeout, retries=1, path=args.path, use_env_proxy=not args.no_proxy
    )
    if not client.endpoint or not client.model:
        parser.error("--probe needs --url (or DECISION_MODEL_URL) and --model (or DECISION_MODEL_NAME)")
    report = probe(client, args.image, args.max_side)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    for name, step in report["steps"].items():
        print(
            f"{name:<11} {step['status']:<8} http={step['http_status']} {step['latency_ms']} ms  {step['error'] or ''}"
        )
    if not report["images"]["skipped"]:
        print(f"image styles the server reads: {report['image_styles_read'] or 'none'}")
    return 0 if all(step["status"] == "ok" for step in report["steps"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
