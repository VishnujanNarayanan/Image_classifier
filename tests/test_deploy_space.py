"""What the Space deployment stages.

The staging list is easy to get wrong in ways that only show up as a broken Space
minutes after an upload: a missing model, or bytecode from the wrong Python. Both
are cheap to check here.
"""
import os

import pytest

from scripts.deploy_space import CONTENT, stage


@pytest.fixture
def staged(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for src, _ in CONTENT:
        p = root / src
        p.parent.mkdir(parents=True, exist_ok=True)
        if os.path.splitext(src)[1]:
            p.write_text("x")
        else:
            p.mkdir(exist_ok=True)
            (p / "mod.py").write_text("x")
            (p / "__pycache__").mkdir()
            (p / "__pycache__" / "mod.cpython-311.pyc").write_bytes(b"\x00")
    monkeypatch.setattr("scripts.deploy_space.ROOT", str(root))
    out = tmp_path / "out"
    out.mkdir()
    stage(str(out))
    return {os.path.relpath(os.path.join(r, f), out)
            for r, _, fs in os.walk(out) for f in fs}


def test_the_space_gets_an_entry_point_and_a_model(staged):
    assert "app.py" in staged
    assert "requirements.txt" in staged
    assert os.path.join("artifacts", "deep.keras") in staged


def test_no_bytecode_is_shipped(staged):
    assert not any("__pycache__" in f or f.endswith(".pyc") for f in staged)


def test_training_only_files_are_left_behind(staged):
    assert not any(f.endswith(("train.py", "prep.py", "train_transfer.py")) for f in staged)


def test_a_missing_model_fails_loudly_rather_than_uploading_a_broken_space(
        tmp_path, monkeypatch):
    root = tmp_path / "bare"
    root.mkdir()
    monkeypatch.setattr("scripts.deploy_space.ROOT", str(root))
    with pytest.raises(SystemExit, match="missing"):
        stage(str(tmp_path / "out2"))
