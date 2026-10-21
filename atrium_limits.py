"""atrium_limits.py — every service limit as a setting (atrium-project#53, factor III).

Canonical copy lives in the hub at ``docs/templates/shared/atrium_limits.py`` and is
mirrored **byte-identically** to the root of every tool repo (``MANIFEST.json``, enforced
by ``para-drift.reusable.yml``). It imports only the standard library: the command-line
tools import it too, and the nlp-enrich and llm-enrich CLI images install no web
framework. The HTTP side (the harmonised error body, ``/info``) is in
``service/atrium_service.py``, which imports this module, never the other way round.

What a limit is (``docs/agent_skill_strategy.md`` §4.5): a value that refuses an input;
cuts, samples or splits an input or a prompt; caps an output; or bounds the time or the
retries of one request. AMČR's pipeline splits large documents itself and calls the
services from its own workers, so every limit must be a setting it can read and change,
and no limit may cut an input without saying so.

Four pieces:

* :class:`LimitSpec`, built with :func:`limit` — one limit: the environment variable
  that sets it, its default, unit and bounds. The environment value is checked when the
  spec is built, so a malformed setting stops the process at startup and names the
  variable (:class:`LimitConfigError`) instead of being ignored. :meth:`LimitSpec.get`
  re-reads the environment on every call, so a test (or a long-running process) that
  changes the variable sees the new value.
* :class:`LimitSet` — the limits one tool declares, in one place (the repo's
  ``tool_limits.py``). :meth:`LimitSet.values` is the flat ``/info`` ``limits`` map;
  :meth:`LimitSet.meta` says which variable sets each one, its unit, default and source.
* :class:`LimitExceeded` — raised when an input is over a limit. The service maps it to
  the harmonised error ``{status, reason: "limit_exceeded", detail, limit}``. It is
  deliberately **not** a ``ValueError``, so an ``except ValueError`` that turns bad
  input into a 422 cannot swallow it.
* :class:`LimitNotes` — the per-request record of every limit that shaped a result
  without refusing it (a sampled language window, a trimmed prompt, a split page). It is
  passed down the call chain explicitly (a context variable would not cross
  ``run_in_executor`` or a subprocess), written into the paradata as ``limits_applied``
  and echoed in the service response.

No example in this file names a variable literally: ``tests/test_env_contract.py`` reads
source text, and would count one as a variable this module reads in all five repos.

Run: ``python atrium_limits.py --selftest``
"""

from __future__ import annotations

import math
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

Number = Union[int, float]

#: What a limit did to a result that it shaped without refusing it. Published in the
#: paradata and in every response, so a value here is never renamed or removed.
#:
#: * ``sampled`` — a decision was made on a part of the input (a language-id window).
#: * ``split``   — the input was processed in full, in pieces, and the pieces shaped the
#:   result (wrapped lines, a page chunked for the reading-order model).
#: * ``trimmed`` — a prompt, a list or an embedding window was cut (vocabulary terms left
#:   out of a prompt, the top-N entities kept).
#: * ``skipped`` — a unit of the input got no result (a segment left untranslated after
#:   the retries ran out, a line whose reply was cut).
#: * ``stopped`` — processing ended early (too many consecutive errors).
EFFECTS: Tuple[str, ...] = ("sampled", "split", "trimmed", "skipped", "stopped")

#: HTTP statuses a :class:`LimitExceeded` may carry (atrium-project#53, D7). The reason
#: is always ``limit_exceeded``; the status follows the cause: 413 when the request
#: content is too large, 422 when a parameter or a per-input processing budget is
#: exceeded, 504 when a time limit that depends on an upstream service expires.
LIMIT_STATUSES: Tuple[int, ...] = (413, 422, 504)

_MIB = 1024 * 1024


class LimitConfigError(ValueError):
    """A limit's environment (or config-file) value is malformed or out of range.

    Raised when the spec is built, i.e. at import or startup, so a mis-set limit stops
    the process with a message naming the variable instead of being silently ignored.
    """


class LimitExceeded(Exception):
    """An input is over a limit. Mapped to ``{status, reason: "limit_exceeded", ...}``.

    ``key`` is the ``/info`` key, ``value`` the limit's effective value, ``observed``
    what the input measured (``None`` when it is not known exactly, e.g. a stream cut
    off at the limit), ``unit`` the unit of both. ``http_status`` is one of
    :data:`LIMIT_STATUSES`.
    """

    def __init__(
        self,
        key: str,
        value: Number,
        observed: Optional[Number],
        /,
        *,
        unit: str,
        env: Optional[str] = None,
        http_status: int = 413,
        detail: Optional[str] = None,
    ) -> None:
        if http_status not in LIMIT_STATUSES:
            raise ValueError(f"LimitExceeded http_status must be one of {LIMIT_STATUSES}, not {http_status}")
        self.key = key
        self.value = value
        self.observed = observed
        self.unit = unit
        self.env = env
        self.http_status = http_status
        self.detail = detail or _default_detail(key, value, observed, unit, env)
        super().__init__(self.detail)

    def to_dict(self) -> Dict[str, Any]:
        """The ``limit`` member of the harmonised error body."""
        return {
            "key": self.key,
            "env": self.env,
            "value": self.value,
            "observed": self.observed,
            "unit": self.unit,
        }


def _default_detail(key: str, value: Number, observed: Optional[Number], unit: str, env: Optional[str]) -> str:
    what = f"{_fmt(observed)} {unit}" if observed is not None else f"more than {_fmt(value)} {unit}"
    setting = f" (setting: {env})" if env else ""
    return f"Over the {key} limit: {what}; the limit is {_fmt(value)} {unit}{setting}."


def _fmt(n: Any) -> str:
    if isinstance(n, float) and n.is_integer():
        return str(int(n))
    if isinstance(n, float):
        return f"{n:g}"
    return str(n)


@dataclass(frozen=True)
class LimitSpec:
    """One limit. Build it with :func:`limit`, not directly."""

    env: str
    key: str
    default: Number
    unit: str
    kind: type = int
    minimum: Optional[Number] = None
    maximum: Optional[Number] = None
    zero_means_unlimited: bool = False
    http_status: int = 413
    legacy_env: Optional[str] = None
    legacy_divisor: Number = 1

    # -- reading -------------------------------------------------------------------

    def _parse(self, raw: Any, where: str) -> Number:
        text = str(raw).strip()
        try:
            number = float(text)
        except ValueError:
            raise LimitConfigError(f"{where} is {text!r}, which is not a number ({self.unit}).") from None
        if not math.isfinite(number):
            raise LimitConfigError(f"{where} is {text!r}; a limit must be a finite number.")
        if self.kind is int:
            if not number.is_integer():
                raise LimitConfigError(f"{where} is {text!r}; {self.env} takes a whole number of {self.unit}.")
            value: Number = int(number)
        else:
            value = float(number)
        return self._bounded(value, where)

    def _bounded(self, value: Number, where: str) -> Number:
        if value < 0:
            raise LimitConfigError(f"{where} is {_fmt(value)}; a limit cannot be negative.")
        if self.zero_means_unlimited and value == 0:
            return value
        if self.minimum is not None and value < self.minimum:
            raise LimitConfigError(f"{where} is {_fmt(value)}; the smallest allowed value is {_fmt(self.minimum)}.")
        if self.maximum is not None and value > self.maximum:
            raise LimitConfigError(f"{where} is {_fmt(value)}; the largest allowed value is {_fmt(self.maximum)}.")
        return value

    def source(self, *, config: Any = None) -> str:
        """Where the effective value comes from: ``env``, ``config`` or ``default``."""
        if _present(os.environ.get(self.env)):
            return "env"
        if self.legacy_env and _present(os.environ.get(self.legacy_env)):
            return "env"
        if _present(config):
            return "config"
        return "default"

    def get(self, *, config: Any = None) -> Number:
        """The effective value: environment, then the config-file value, then the default.

        Read on every call. A blank environment value counts as unset
        (``docs/templates/env.example.template`` Rule 2).
        """
        raw = os.environ.get(self.env)
        if _present(raw):
            return self._parse(raw, f"The environment variable {self.env}")
        if self.legacy_env:
            legacy = os.environ.get(self.legacy_env)
            if _present(legacy):
                parsed = self._parse(legacy, f"The environment variable {self.legacy_env}")
                return self._bounded(parsed / self.legacy_divisor, f"{self.legacy_env} (converted to {self.unit})")
        if _present(config):
            return self._parse(config, f"The config value for {self.env}")
        return self.default

    # -- enforcing -----------------------------------------------------------------

    def is_unlimited(self, value: Optional[Number] = None) -> bool:
        v = self.get() if value is None else value
        return self.zero_means_unlimited and v == 0

    def exceeded(self, observed: Optional[Number], *, value: Optional[Number] = None, detail: Optional[str] = None) -> LimitExceeded:
        """Build (not raise) the :class:`LimitExceeded` for ``observed``."""
        v = self.get() if value is None else value
        return LimitExceeded(
            self.key, v, observed, unit=self.unit, env=self.env, http_status=self.http_status, detail=detail
        )

    def check(self, observed: Number, *, value: Optional[Number] = None, detail: Optional[str] = None) -> None:
        """Raise :class:`LimitExceeded` when ``observed`` is over the limit.

        ``value`` overrides the effective value (for a limit read once per request, or
        one whose value came from a config file). A limit set to 0 with
        ``zero_means_unlimited`` never refuses.
        """
        v = self.get() if value is None else value
        if self.zero_means_unlimited and v == 0:
            return
        if observed > v:
            raise self.exceeded(observed, value=v, detail=detail)

    def meta(self, *, config: Any = None) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "env": self.env,
            "unit": self.unit,
            "default": self.default,
            "source": self.source(config=config),
        }
        if self.zero_means_unlimited:
            entry["zero_means_unlimited"] = True
        return entry


def _present(raw: Any) -> bool:
    return raw is not None and str(raw).strip() != ""


def limit(
    env: str,
    default: Number,
    /,
    *,
    unit: str,
    kind: type = int,
    key: Optional[str] = None,
    minimum: Optional[Number] = None,
    maximum: Optional[Number] = None,
    zero_means_unlimited: bool = False,
    status: int = 413,
    legacy_env: Optional[str] = None,
    legacy_divisor: Number = 1,
) -> LimitSpec:
    """Declare one limit and check its current environment value.

    ``env`` and ``default`` are positional-only and must be literals at the call site,
    so ``tests/test_env_contract.py`` finds the variable and its default in the source.
    ``key`` defaults to ``env`` in lower case; pass it to keep an ``/info`` key that
    predates this module. ``status`` is the HTTP status its refusal carries (see
    :data:`LIMIT_STATUSES`). Raises :class:`LimitConfigError` if the variable is set to a
    malformed or out-of-range value.
    """
    if not env or env.upper() != env or not env.replace("_", "").isalnum():
        raise ValueError(f"limit(): {env!r} is not an environment variable name")
    if kind not in (int, float):
        raise ValueError(f"limit(): kind must be int or float, not {kind!r}")
    if not unit:
        raise ValueError(f"limit(): {env} needs a unit")
    if status not in LIMIT_STATUSES:
        raise ValueError(f"limit(): status must be one of {LIMIT_STATUSES}, not {status}")
    spec = LimitSpec(
        env=env,
        key=key or env.lower(),
        default=kind(default),
        unit=unit,
        kind=kind,
        minimum=minimum,
        maximum=maximum,
        zero_means_unlimited=zero_means_unlimited,
        http_status=status,
        legacy_env=legacy_env,
        legacy_divisor=legacy_divisor,
    )
    spec._bounded(spec.default, f"The default of {env}")
    spec.get()  # fail at startup, naming the variable, rather than on the first request
    return spec


def resolve_limit(env: str, default: Number, /, **kwargs: Any) -> Number:
    """One-line form: declare the limit and return its effective value now."""
    return limit(env, default, **kwargs).get()


def upload_limit(default_mb: Number, /) -> LimitSpec:
    """The §4.5 upload limit every service has, in megabytes.

    Keeps the deprecated bytes variable working as a fallback, as
    ``atrium_service.resolve_max_upload_mb`` always has. The ``/info`` key stays
    ``max_upload_mb``.
    """
    return limit(
        "MAX_UPLOAD_MB",
        default_mb,
        unit="MB",
        kind=float,
        key="max_upload_mb",
        legacy_env="MAX_UPLOAD_BYTES",
        legacy_divisor=_MIB,
    )


@dataclass
class _Derived:
    fn: Callable[[], Any]
    unit: str
    derived_from: Tuple[str, ...]


class LimitSet:
    """The limits one tool declares. Build one per repo, in ``tool_limits.py``.

    There is no module-level registry: each repo, and each test, holds its own set, so
    two apps built in one process never see each other's limits.

    ``config`` is an optional zero-argument callable returning ``{env name: raw value}``
    for limits a config file can also set (alto-postprocess's ``[TEXT_INGEST]``); the
    environment still wins over it.
    """

    def __init__(self, *specs: LimitSpec, config: Optional[Callable[[], Mapping[str, Any]]] = None) -> None:
        self._specs: Dict[str, LimitSpec] = {}
        self._derived: Dict[str, _Derived] = {}
        self._config = config
        for spec in specs:
            self.add(spec)

    def add(self, spec: LimitSpec) -> LimitSpec:
        if spec.key in self._specs or spec.key in self._derived:
            raise ValueError(f"LimitSet: the key {spec.key!r} is declared twice")
        self._specs[spec.key] = spec
        return spec

    def derived(self, key: str, fn: Callable[[], Any], /, *, unit: str, derived_from: Iterable[str]) -> None:
        """Report a value computed from settings or from the model (e.g. a token window).

        ``fn`` takes no arguments; if it raises or returns ``None`` the value is reported
        as ``null`` rather than failing ``/info``.
        """
        if key in self._specs or key in self._derived:
            raise ValueError(f"LimitSet: the key {key!r} is declared twice")
        self._derived[key] = _Derived(fn=fn, unit=unit, derived_from=tuple(derived_from))

    def _config_values(self) -> Mapping[str, Any]:
        if self._config is None:
            return {}
        return self._config() or {}

    def __getitem__(self, key: str) -> LimitSpec:
        return self._specs[key]

    def __contains__(self, key: object) -> bool:
        return key in self._specs or key in self._derived

    def __iter__(self) -> Iterator[LimitSpec]:
        return iter(self._specs.values())

    def specs(self) -> List[LimitSpec]:
        return list(self._specs.values())

    def get(self, key: str) -> Number:
        """The effective value of one declared limit, config file included."""
        spec = self._specs[key]
        return spec.get(config=self._config_values().get(spec.env))

    def values(self) -> Dict[str, Any]:
        """The flat ``/info`` ``limits`` map: ``{key: effective value}``."""
        cfg = self._config_values()
        out: Dict[str, Any] = {spec.key: spec.get(config=cfg.get(spec.env)) for spec in self._specs.values()}
        for key, d in self._derived.items():
            out[key] = _safe_call(d.fn)
        return out

    def meta(self) -> Dict[str, Dict[str, Any]]:
        """``/info`` ``limits_meta``: which variable sets each limit, its unit, default, source."""
        cfg = self._config_values()
        out = {spec.key: spec.meta(config=cfg.get(spec.env)) for spec in self._specs.values()}
        for key, d in self._derived.items():
            out[key] = {"env": None, "unit": d.unit, "derived_from": list(d.derived_from), "source": "derived"}
        return out


def _safe_call(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # /info must never fail because one derived value is unavailable
        return None


@dataclass
class LimitNote:
    """One limit that shaped a result. ``limit`` is the ``/info`` key."""

    limit: str
    value: Any
    effect: str
    count: int = 1
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "limit": self.limit,
            "value": self.value,
            "effect": self.effect,
            "count": self.count,
            "detail": self.detail,
        }


class LimitNotes:
    """The limits that shaped one request's result (``limits_applied``).

    Notes for the same limit and effect are merged: their counts add up and the first
    non-empty detail is kept, so a limit applied to 400 segments is one note with
    ``count`` 400, not 400 notes. Safe to add to from a worker thread.
    """

    def __init__(self, notes: Optional[Iterable[Any]] = None) -> None:
        self._notes: List[LimitNote] = []
        self._lock = threading.Lock()
        if notes:
            self.extend(notes)

    def note(
        self,
        spec: Union[LimitSpec, str],
        effect: str,
        count: int = 1,
        detail: str = "",
        *,
        value: Any = None,
    ) -> None:
        """Record that ``spec`` shaped the result ``count`` times, with ``effect``."""
        if effect not in EFFECTS:
            raise ValueError(f"LimitNotes.note(): effect must be one of {EFFECTS}, not {effect!r}")
        if count <= 0:
            return
        if isinstance(spec, LimitSpec):
            key = spec.key
            if value is None:
                value = spec.get()
        else:
            key = str(spec)
        self._add(LimitNote(limit=key, value=value, effect=effect, count=int(count), detail=str(detail or "")))

    def _add(self, new: LimitNote) -> None:
        with self._lock:
            for existing in self._notes:
                if existing.limit == new.limit and existing.effect == new.effect:
                    existing.count += new.count
                    if not existing.detail and new.detail:
                        existing.detail = new.detail
                    return
            self._notes.append(new)

    def extend(self, notes: Iterable[Any]) -> None:
        """Add notes from another :class:`LimitNotes`, :class:`LimitNote` objects or dicts
        (e.g. notes carried back from a subprocess in its JSON result)."""
        if isinstance(notes, LimitNotes):
            notes = notes.as_list()
        for item in notes:
            if isinstance(item, LimitNote):
                self._add(LimitNote(item.limit, item.value, item.effect, item.count, item.detail))
            elif isinstance(item, Mapping):
                effect = str(item.get("effect", ""))
                if effect not in EFFECTS:
                    continue
                self._add(
                    LimitNote(
                        limit=str(item.get("limit", "")),
                        value=item.get("value"),
                        effect=effect,
                        count=int(item.get("count", 1) or 1),
                        detail=str(item.get("detail", "") or ""),
                    )
                )

    def copy(self) -> "LimitNotes":
        return LimitNotes(self.as_list())

    def as_list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [n.to_dict() for n in self._notes]

    def header_summary(self) -> str:
        """A compact ASCII form for an HTTP header (header values are latin-1).

        ``key=effect:count`` pairs joined by ``"; "``; empty when there are no notes.
        """
        with self._lock:
            parts = [f"{n.limit}={n.effect}:{n.count}" for n in self._notes]
        return "; ".join(p.encode("ascii", "replace").decode("ascii") for p in parts)

    def __len__(self) -> int:
        return len(self._notes)

    def __bool__(self) -> bool:
        return bool(self._notes)

    def __iter__(self) -> Iterator[LimitNote]:
        with self._lock:
            return iter(list(self._notes))

    def __repr__(self) -> str:
        return f"LimitNotes({self.as_list()!r})"


# ──────────────────────────────────────────────────────────────────────────────
# Self-test (python atrium_limits.py --selftest)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _EnvPatch:
    """Set/unset environment variables for the self-test, restoring them after."""

    values: Dict[str, Optional[str]]
    _saved: Dict[str, Optional[str]] = field(default_factory=dict)

    def __enter__(self) -> "_EnvPatch":
        for name, value in self.values.items():
            self._saved[name] = os.environ.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        return self

    def __exit__(self, *exc: Any) -> None:
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _selftest() -> None:
    # The variable names are built at run time so the source-text scanner in
    # tests/test_env_contract.py does not read them as variables this module uses.
    name = "_".join(("ATRIUM", "LIMITS", "SELFTEST"))
    legacy = name + "_BYTES"

    with _EnvPatch({name: None}):
        spec = limit(name, 50, unit="pages")
        assert spec.key == name.lower() and spec.get() == 50 and spec.source() == "default"
        spec.check(50)
        try:
            spec.check(51, detail="too many pages")
        except LimitExceeded as exc:
            assert exc.http_status == 413 and exc.observed == 51 and str(exc) == "too many pages"
            assert exc.to_dict()["env"] == name
        else:
            raise AssertionError("check() did not refuse an input over the limit")
        assert not isinstance(spec.exceeded(1), ValueError)

    with _EnvPatch({name: " 7 "}):
        assert spec.get() == 7 and spec.source() == "env"
        assert spec.get(config="9") == 7  # the environment wins over the config file
    with _EnvPatch({name: ""}):
        assert spec.get() == 50  # blank counts as unset
        assert spec.get(config="9") == 9 and spec.source(config="9") == "config"

    for bad in ("ten", "2.5", "-1", "inf"):
        with _EnvPatch({name: bad}):
            try:
                limit(name, 50, unit="pages")
            except LimitConfigError as exc:
                assert name in str(exc)
            else:
                raise AssertionError(f"a malformed value {bad!r} was accepted")

    with _EnvPatch({name: "0"}):
        unlimited = limit(name, 4, unit="jobs", zero_means_unlimited=True)
        unlimited.check(10**9)
        assert unlimited.is_unlimited()
        try:
            limit(name, 4, unit="jobs", minimum=1)
        except LimitConfigError:
            pass
        else:
            raise AssertionError("minimum was not enforced")

    with _EnvPatch({name: None, legacy: str(2 * _MIB)}):
        mb = limit(name, 10, unit="MB", kind=float, legacy_env=legacy, legacy_divisor=_MIB)
        assert mb.get() == 2.0 and mb.source() == "env"

    with _EnvPatch({name: None}):
        limits = LimitSet(limit(name, 3, unit="s", key="timeout_s"), config=lambda: {name: "5"})
        limits.derived("window_tokens", lambda: 128, unit="tokens", derived_from=["model"])
        limits.derived("broken", lambda: 1 / 0, unit="tokens", derived_from=["model"])
        assert limits.values() == {"timeout_s": 5, "window_tokens": 128, "broken": None}
        assert limits.meta()["timeout_s"]["source"] == "config"
        assert limits.meta()["window_tokens"]["source"] == "derived"
        try:
            limits.add(limit(name, 1, unit="s", key="timeout_s"))
        except ValueError:
            pass
        else:
            raise AssertionError("a duplicate key was accepted")

    notes = LimitNotes()
    notes.note("lang_id_document_chars", "sampled", detail="first 20000 of 58213 characters", value=20000)
    notes.note("lang_id_document_chars", "sampled", 2)
    notes.note("ne_summary_top_n", "trimmed", 0)  # a zero count records nothing
    assert notes.as_list() == [
        {
            "limit": "lang_id_document_chars",
            "value": 20000,
            "effect": "sampled",
            "count": 3,
            "detail": "first 20000 of 58213 characters",
        }
    ]
    notes.note("vocab", "trimmed", detail="Příliš dlouhé")
    assert notes.header_summary() == "lang_id_document_chars=sampled:3; vocab=trimmed:1"
    assert LimitNotes(notes.as_list() + [{"effect": "unknown"}]).as_list() == notes.as_list()
    try:
        notes.note("x", "cut")
    except ValueError:
        pass
    else:
        raise AssertionError("an unpublished effect was accepted")

    print("atrium_limits selftest: OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        _selftest()
    else:
        print(__doc__)
