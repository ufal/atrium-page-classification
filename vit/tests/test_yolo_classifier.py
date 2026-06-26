import numpy as np

from yolo_classifier import YOLOClassifier, build_yolo_dataset


def test_yolo_resolve_checkpoint():
    """Verify standard short tags resolve to correct Ultralytics checkpoints."""
    assert YOLOClassifier._resolve_checkpoint("yv8s") == "yolov8s-cls.pt"
    assert YOLOClassifier._resolve_checkpoint("y11n") == "yolo11n-cls.pt"
    # Unknown paths should pass through unaltered
    assert YOLOClassifier._resolve_checkpoint("local_custom_model.pt") == "local_custom_model.pt"


def test_build_yolo_dataset(tmp_path):
    """Verify images are copied into the expected `<split>/<CLASS>/<file>` structure."""

    # Mock data
    cats = ["DRAW", "TEXT"]
    labels = np.array(
        [
            [1, 0],  # DRAW
            [0, 1],  # TEXT
            [1, 0],  # DRAW
        ]
    )

    # Create fake files
    files = []
    for i in range(3):
        f = tmp_path / f"img_{i}.png"
        f.touch()
        files.append(str(f))

    dest_dir = tmp_path / "yolo_run"

    # Execute build
    built_path = build_yolo_dataset(files, labels, cats, dest_dir, "train")

    # Verify expected YOLO directory tree
    assert built_path.name == "train"
    assert (built_path / "DRAW").is_dir()
    assert (built_path / "TEXT").is_dir()

    assert (built_path / "DRAW" / "img_0.png").exists()
    assert (built_path / "TEXT" / "img_1.png").exists()
    assert (built_path / "DRAW" / "img_2.png").exists()
