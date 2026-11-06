"""tests/test_decision_client.py — the decision-model client (#58 Phase 0; ufal/atrium-project#83).

Everything runs against ``decision_client.FakeSystemOne`` on 127.0.0.1, so the tests need no model,
no GPU and no network. The answer shapes the fake server returns are those of Clef's own
``systemone_answer`` (the model card's ``joint_schema_model.py``).
"""

import json
import math
from pathlib import Path

import pytest

import decision_client as dc

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = REPO_ROOT / "small_data_samples"


def _first_png(category: str) -> Path:
    return sorted((SAMPLES / category).glob("*.png"))[0]


# ── builders ──────────────────────────────────────────────────────────────────────────────────


def test_builders_produce_the_wire_shapes():
    assert dc.choice("Which?", {"a": "first", "b": None}) == {
        "type": "choice",
        "instructions": "Which?",
        "criteria": {"a": "first", "b": None},
    }
    assert dc.score(None, ["low", "high"]) == {"type": "score", "criteria": ["low", "high"]}
    assert dc.noul("Yes?") == {"type": "noul", "instructions": "Yes?"}
    assert dc.noul("Yes?", true="it is")["criteria"] == {"true": "it is"}


@pytest.mark.parametrize(
    "build",
    [
        lambda: dc.choice("x", {"only": "one"}),
        lambda: dc.choice("x", {f"o{i}": None for i in range(dc.MAX_OPTIONS + 1)}),
        lambda: dc.score("x", ["one"]),
        lambda: dc.questions({}),
        lambda: dc.questions({f"q{i}": dc.noul("x") for i in range(dc.MAX_QUESTIONS + 1)}),
        lambda: dc.questions({"q": {"type": "maybe"}}),
        lambda: dc.questions({"q": {"type": "noul", "criteria": {"perhaps": "x"}}}),
    ],
)
def test_builders_refuse_what_the_interface_cannot_carry(build):
    with pytest.raises(dc.DecisionInputError):
        build()


def test_the_limits_themselves_are_allowed():
    items = {"big": dc.choice("x", {f"o{i}": None for i in range(dc.MAX_OPTIONS)})}
    items.update({f"q{i}": dc.noul("x") for i in range(dc.MAX_QUESTIONS - 1)})
    assert len(dc.questions(items)) == dc.MAX_QUESTIONS


def test_options_of_each_type():
    assert dc.options_of(dc.noul("x")) == ["true", "false"]
    assert dc.options_of(dc.choice("x", {"b": None, "a": None})) == ["b", "a"]
    assert dc.options_of(dc.score("x", ["l", "m", "h"])) == ["0", "1", "2"]


def test_schema_digest_is_canonical_and_tracks_any_change():
    one = {"a": dc.noul("x"), "b": dc.choice("y", {"p": "1", "q": "2"})}
    reordered = {"b": dc.choice("y", {"q": "2", "p": "1"}), "a": dc.noul("x")}
    assert dc.schema_digest(one) == dc.schema_digest(reordered)
    reworded = {"a": dc.noul("x?"), "b": dc.choice("y", {"p": "1", "q": "2"})}
    assert dc.schema_digest(one) != dc.schema_digest(reworded)
    assert len(dc.schema_digest(one)) == 64


# ── validation ────────────────────────────────────────────────────────────────────────────────

ITEMS = {
    "department": dc.choice("Team?", {"billing": "Payments", "technical": "Bugs"}),
    "urgency": dc.score(None, ["Can wait", "This week", "Today"]),
    "outage": dc.noul("Down?"),
}


def _good_body():
    return {
        "model": "m",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "technical",
                "confidence": 0.9,
                "probabilities": {"billing": 0.1, "technical": 0.9},
            },
            "urgency": {
                "type": "score",
                "score": 1.7,
                "confidence": 0.7,
                "legend": {},
                "probabilities": {"0": 0.0, "1": 0.3, "2": 0.7},
            },
            "outage": {"type": "noul", "noul": 0.8},
        },
        "usage": {"input_tokens": 123, "output_tokens": 0},
    }


def test_a_well_formed_body_validates():
    answers, reason = dc.validate_answers(ITEMS, _good_body())
    assert reason is None
    assert answers["department"].choice == "technical"
    assert answers["urgency"].score == pytest.approx(1.7)
    assert answers["outage"].p_true == pytest.approx(0.8)
    assert answers["outage"].probabilities == {"true": 0.8, "false": pytest.approx(0.2)}
    assert answers["urgency"].best() == ("2", 0.7)


def test_a_bare_number_is_accepted_as_a_noul_answer():
    body = _good_body()
    body["answers"]["outage"] = 0.25
    answers, reason = dc.validate_answers(ITEMS, body)
    assert reason is None and answers["outage"].p_true == 0.25


def test_rounding_to_four_decimals_is_tolerated():
    items = {"c": dc.choice("x", {f"o{i}": None for i in range(7)})}
    probs = {f"o{i}": round(1 / 7, 4) for i in range(7)}  # sums to 1.0001
    body = {"answers": {"c": {"type": "choice", "choice": "o0", "probabilities": probs}}}
    assert dc.validate_answers(items, body)[1] is None


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda b: b["answers"].pop("outage"), "missing"),
        (lambda b: b["answers"].update(extra={"type": "noul", "noul": 0.5}), "unexpected"),
        (lambda b: b["answers"]["department"].update(choice="sales"), "not one of the options"),
        (lambda b: b["answers"]["department"].update(choice="billing"), "not the most probable"),
        (lambda b: b["answers"]["department"]["probabilities"].update(technical=0.5), "sum to"),
        (lambda b: b["answers"]["department"]["probabilities"].pop("billing"), "cover"),
        (
            lambda b: b["answers"]["department"]["probabilities"].update(billing=-0.1, technical=1.1),
            "not a probability",
        ),
        (lambda b: b["answers"]["outage"].update(noul=1.5), "not a probability"),
        (lambda b: b["answers"]["urgency"].update(score=3.5), "outside"),
        (lambda b: b["answers"]["outage"].update(type="choice"), "answer type"),
        (lambda b: b.pop("answers"), "no 'answers'"),
    ],
)
def test_every_inconsistency_is_reported_not_repaired(mutate, fragment):
    body = _good_body()
    mutate(body)
    answers, reason = dc.validate_answers(ITEMS, body)
    assert answers == {}
    assert fragment in reason


# ── the client against the fake server ───────────────────────────────────────────────────────


def test_ok_records_usage_latency_digest_and_server():
    with dc.FakeSystemOne() as fake:
        d = dc.DecisionClient(fake.url, "clef-test", timeout_s=5, retries=0).decide("state", ITEMS)
    assert d.ok, d.error
    assert set(d.answers) == set(ITEMS)
    assert d.input_tokens and d.latency_ms is not None and d.attempts == 1
    assert d.schema_digest == dc.schema_digest(ITEMS)
    assert d.server.startswith("FakeSystemOne")
    assert fake.requests[0]["model"] == "clef-test" and fake.requests[0]["state"] == "state"


def test_the_endpoint_is_not_doubled_when_the_url_names_the_route():
    with dc.FakeSystemOne() as fake:
        d = dc.DecisionClient(fake.url + "/v1/systemone", "m", timeout_s=5, retries=0).decide("s", ITEMS)
    assert d.ok, d.error


def test_the_selftest_passes():
    assert dc.selftest() == 0


@pytest.mark.parametrize(
    "responder, status, attempts",
    [
        (lambda r: (400, {"error": "bad"}), "rejected", 1),
        (lambda r: (422, {"error": "bad"}), "rejected", 1),
        (lambda r: (429, {"error": "slow down"}), "error", 3),
        (lambda r: (500, {"error": "boom"}), "error", 3),
        (lambda r: (200, b"not json"), "invalid", 1),
        (lambda r: (200, [1, 2]), "invalid", 1),
    ],
)
def test_status_and_retries_per_server_answer(responder, status, attempts):
    with dc.FakeSystemOne(responder) as fake:
        d = dc.DecisionClient(fake.url, "m", timeout_s=5, retries=2, backoff_s=0.001).decide("s", ITEMS)
    assert d.status == status
    assert d.attempts == attempts
    assert d.answers == {}


def test_decide_never_raises_on_bad_input():
    client = dc.DecisionClient("http://127.0.0.1:9", "m", timeout_s=1, retries=0)
    assert client.decide("s", {"q": {"type": "choice", "criteria": {"a": None}}}).status == "rejected"
    assert client.decide(object(), ITEMS).status == "rejected"  # a state that is not JSON
    assert client.decide("s", ITEMS, images=["/no/such/image.png"]).status == "rejected"
    assert client.decide("s", ITEMS, images=["data:x"] * (dc.MAX_IMAGES + 1)).status == "rejected"


@pytest.mark.parametrize("url, model", [(None, "m"), ("", "m"), ("http://x", None), ("ftp://x", "m")])
def test_not_configured_sends_nothing(url, model):
    assert dc.DecisionClient(url, model).decide("s", ITEMS).status == "not_configured"


# ── images ────────────────────────────────────────────────────────────────────────────────────


def test_image_data_uri_downscales_to_the_long_side():
    from PIL import Image

    uri = dc.image_data_uri(_first_png("PHOTO"), max_side=256)
    assert uri.startswith("data:image/jpeg;base64,")
    import base64
    import io

    with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as img:
        assert max(img.size) <= 256


@pytest.mark.parametrize("style", dc.IMAGE_STYLES)
def test_each_image_style_reaches_the_server(style):
    with dc.FakeSystemOne() as fake:
        client = dc.DecisionClient(fake.url, "m", timeout_s=10, retries=0, image_style=style)
        d = client.decide({"task": "look"}, {"p": dc.noul("Photo?")}, images=[_first_png("PHOTO")], max_side=128)
    assert d.ok, d.error
    sent = fake.requests[0]["images"][0]
    expected = dc.image_on_wire("data:image/jpeg;base64,AAAA", style)
    assert type(sent) is type(expected)
    if isinstance(sent, str):
        assert sent.startswith("data:image/jpeg;base64,") == (style == "data_uri")


# ── lanes and metrics ─────────────────────────────────────────────────────────────────────────


def test_lane():
    assert dc.lane(0.95, accept_at=0.9, review_below=0.5) == "accept"
    assert dc.lane(0.7, accept_at=0.9, review_below=0.5) == "escalate"
    assert dc.lane(0.4, accept_at=0.9, review_below=0.5) == "review"
    with pytest.raises(ValueError):
        dc.lane(0.5, accept_at=0.4, review_below=0.6)


def test_brier_by_hand():
    probs = [{"a": 0.8, "b": 0.2}, {"a": 0.4, "b": 0.6}]
    # item 1: (0.8-1)^2 + 0.2^2 = 0.08; item 2: 0.4^2 + (0.6-1)^2 = 0.32 ... gold b
    assert dc.brier(probs, ["a", "b"]) == pytest.approx((0.08 + 0.32) / 2)


def test_ece_by_hand():
    # Two bins used: (0.9, 1.0] holds 0.95 (right) and 0.95 (wrong); (0.5, 0.6] holds 0.55 (right).
    conf, hit = [0.95, 0.95, 0.55], [True, False, True]
    expected = abs(0.5 - 0.95) * 2 / 3 + abs(1.0 - 0.55) * 1 / 3
    assert dc.ece(conf, hit) == pytest.approx(expected)


def test_coverage_curve_and_threshold():
    conf = [0.9, 0.8, 0.8, 0.6]
    hit = [True, True, False, False]
    assert dc.coverage_curve(conf, hit) == [(0.9, 0.25, 0.0), (0.8, 0.75, pytest.approx(1 / 3)), (0.6, 1.0, 0.5)]
    assert dc.pick_threshold(conf, hit, target_error=0.4) == (0.8, 0.75, pytest.approx(1 / 3))
    assert dc.pick_threshold(conf, hit, target_error=0.0) == (0.9, 0.25, 0.0)


def test_mcnemar_exact_by_hand():
    assert dc.mcnemar_exact(0, 0) == 1.0
    # b=1, c=9: 2 * P(X <= 1), X ~ Bin(10, 0.5) = 2 * 11/1024
    assert dc.mcnemar_exact(1, 9) == pytest.approx(22 / 1024)
    assert dc.mcnemar_exact(5, 5) == 1.0


# ── probe ─────────────────────────────────────────────────────────────────────────────────────


def _reads_only(style):
    """A server that sees an image only in the wire form *style*: P(photo) 0.9 with it, 0.1 without."""

    def shape(item):
        return ("str", item.startswith("data:")) if isinstance(item, str) else ("dict", tuple(sorted(item)))

    wanted = shape(dc.image_on_wire("data:image/jpeg;base64,AA", style))

    def responder(request):
        status, body = dc.default_responder(request)
        if status == 200 and "photo_main" in body["answers"]:
            images = request.get("images") or []
            seen = bool(images) and shape(images[0]) == wanted
            body["answers"]["photo_main"] = {"type": "noul", "noul": 0.9 if seen else 0.1}
        return status, body

    return responder


@pytest.mark.parametrize("style", dc.IMAGE_STYLES)
def test_probe_finds_the_image_style_a_server_reads(style):
    images = [str(_first_png("PHOTO")), str(_first_png("TEXT_P"))]
    with dc.FakeSystemOne(_reads_only(style)) as fake:
        report = dc.probe(dc.DecisionClient(fake.url, "m", timeout_s=10, retries=0), images, max_side=128)
    assert report["image_styles_read"] == [style]


def test_probe_reports_without_the_address():
    with dc.FakeSystemOne() as fake:
        client = dc.DecisionClient(fake.url, "m", timeout_s=10, retries=0)
        report = dc.probe(client, [str(_first_png("PHOTO"))], max_side=128)
        text = json.dumps(report)
    assert "127.0.0.1" not in text and fake.url not in text
    assert report["endpoint"] == "$DECISION_MODEL_URL/v1/systemone"
    assert {s["status"] for s in report["steps"].values()} == {"ok"}
    assert set(report["images"]["styles"]) == set(dc.IMAGE_STYLES)
    assert not math.isnan(report["steps"]["limits"]["latency_ms"])


def test_probe_without_images_skips_the_image_check():
    with dc.FakeSystemOne() as fake:
        report = dc.probe(dc.DecisionClient(fake.url, "m", timeout_s=10, retries=0), [])
    assert report["images"] == {"skipped": True}
    assert report["image_styles_read"] == []


def test_probe_cli_needs_a_server(capsys):
    with pytest.raises(SystemExit):
        dc.main(["--probe", "--url", "", "--model", "m"])
