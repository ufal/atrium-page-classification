"""
tests/test_inference_alignment.py
=================================
A directory run must pair every prediction with its own image (#61).

`custom_collate` drops an image that fails to load, and `dataframe_results` used to pair the
remaining predictions with the file names by position: every page after an unreadable one got its
neighbour's category, and the last page was lost, without an error. With `-tn 1` the score column
held the class number. These tests run the real dataloader and `infer_dataloader` with a stub
model whose answer is the image's colour, so a shifted pairing shows up as a wrong category.

No GPU, no trained model, and no network access required.
"""

import types
import warnings

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")
transforms = pytest.importorskip("torchvision.transforms")

import classifier as C  # noqa: E402
from utils import confusion_plot, dataframe_results, keep_readable  # noqa: E402

CATEGORIES = ["RED", "GREEN", "BLUE"] + [f"OTHER_{i}" for i in range(8)]
COLOURS = {"RED": (255, 0, 0), "GREEN": (0, 255, 0), "BLUE": (0, 0, 255)}


class _ColourModel(torch.nn.Module):
    """Logits from the mean of each colour channel: a red page is class 0, green 1, blue 2."""

    def forward(self, pixel_values):
        means = pixel_values.mean(dim=(2, 3))
        logits = torch.full((means.shape[0], len(CATEGORIES)), -10.0)
        logits[:, :3] = means * 10
        return types.SimpleNamespace(logits=logits)


def _classifier():
    clf = C.ImageClassifier.__new__(C.ImageClassifier)
    clf.model, clf.device = _ColourModel(), torch.device("cpu")
    clf.eval_transforms = transforms.ToTensor()
    return clf


def _pages(tmp_path, names):
    """Write one page per name: a colour name gives a page of that colour, BROKEN a non-image."""
    paths = []
    for page, name in enumerate(names, start=1):
        path = tmp_path / f"DOC-{page}.png"
        if name == "BROKEN":
            path.write_bytes(b"not an image")
        else:
            Image.new("RGB", (8, 8), COLOURS[name]).save(path)
        paths.append(str(path))
    return paths


def _run(paths, top_n=3, batch_size=2, raw=False):
    clf = _classifier()
    return clf.infer_dataloader(clf.create_dataloader(paths, batch_size), top_n, raw)


def test_an_unreadable_page_leaves_every_other_page_with_its_own_category(tmp_path):
    names = ["RED", "BROKEN", "BLUE", "GREEN", "RED"]
    paths = _pages(tmp_path, names)
    predictions, raw_scores = _run(paths, raw=True)

    assert len(predictions) == len(paths)
    assert predictions[1] is None and raw_scores[1] is None

    kept, predictions, raw_scores, _, skipped = keep_readable(paths, predictions, raw_scores)
    assert skipped == [paths[1]]
    df, raw_df = dataframe_results(kept, predictions, CATEGORIES, 3, raw_scores)

    assert list(df["PAGE"]) == [1, 3, 4, 5]
    assert list(df["CLASS-1"]) == ["RED", "BLUE", "GREEN", "RED"]
    assert len(raw_df) == 4


def test_a_batch_where_no_page_loads_gives_no_prediction(tmp_path):
    paths = _pages(tmp_path, ["BROKEN", "BROKEN", "GREEN"])
    predictions, _ = _run(paths, batch_size=2)
    assert predictions[:2] == [None, None]
    assert predictions[2][0][0] == CATEGORIES.index("GREEN")


def test_top_1_writes_a_probability_not_the_class_number(tmp_path):
    paths = _pages(tmp_path, ["BLUE", "GREEN"])
    predictions, _ = _run(paths, top_n=1)
    df, _ = dataframe_results(paths, predictions, CATEGORIES, 1)

    assert list(df["CLASS-1"]) == ["BLUE", "GREEN"]
    assert all(0.5 <= score <= 1.0 for score in df["SCORE-1"])
    assert list(df["CATEGORY"]) == ["BLUE", "GREEN"]


def test_the_single_image_path_answers_top_1(tmp_path):
    """`-f` and the service's /predict_image call top_n_predictions; top-1 used to raise."""
    clf = _classifier()
    clf.preprocess_image = lambda image, train=False: transforms.ToTensor()(Image.open(image).convert("RGB"))[None]
    (path,) = _pages(tmp_path, ["RED"])

    ((index, score),) = clf.top_n_predictions(path, top_n=1)
    assert CATEGORIES[index] == "RED" and 0.5 <= score <= 1.0
    assert [i for i, _ in clf.top_n_predictions(path, top_n=3)][0] == index


def test_scores_are_renormalised_over_at_least_two_classes():
    probabilities = torch.tensor([0.6, 0.3, 0.1])
    assert C.topn_pairs(probabilities, 1) == [(0, pytest.approx(0.6 / 0.9))]
    assert [i for i, _ in C.topn_pairs(probabilities, 3)] == [0, 1, 2]
    assert sum(s for _, s in C.topn_pairs(probabilities, 3)) == pytest.approx(1.0)


def test_inference_collate_returns_the_indices_of_the_pages_that_loaded():
    batch = [
        {"pixel_values": torch.zeros((3, 4, 4)), "label": None, "index": 4},
        None,
        {"pixel_values": torch.ones((3, 4, 4)), "label": None, "index": 6},
    ]
    collated = C.inference_collate(batch)
    assert collated["indices"] == [4, 6]
    assert collated["pixel_values"].shape == (2, 3, 4, 4)
    assert C.inference_collate([None]) == (None, None)


def test_the_trainer_collate_carries_no_indices():
    """The Trainer passes the collated keys to the model, so `custom_collate` must not add any."""
    batch = [{"pixel_values": torch.zeros((3, 4, 4)), "label": torch.tensor([1.0, 0.0]), "index": 0}]
    assert set(C.custom_collate(batch)) == {"pixel_values", "labels"}


def test_predictions_that_do_not_match_the_images_are_refused():
    with pytest.raises(ValueError, match="aligned"):
        dataframe_results(["a-1.png", "a-2.png"], [[(0, 1.0)]], CATEGORIES, 1)


def test_the_evaluation_report_skips_unreadable_images(tmp_path):
    (tmp_path / "plots").mkdir()
    predictions = [[(0, 0.9), (1, 0.1)], None, [(1, 0.8), (0, 0.2)]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        confusion_plot(predictions, [0, 0, 1], ["A", "B"], "m", top_N=1, output_dir=str(tmp_path))
    assert list((tmp_path / "plots").glob("*.png"))
