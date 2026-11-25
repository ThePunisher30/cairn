import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("make_space", ROOT / "scripts" / "make_space.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_space_folder_has_what_the_image_needs_and_a_readme_with_settings(tmp_path):
    out = tmp_path / "space"
    assert load().main(["--out", str(out)]) == 0
    for name in ("Dockerfile", ".dockerignore", "pyproject.toml", "README.md"):
        assert (out / name).is_file()
    assert (out / "src" / "cairn" / "serve" / "static" / "index.html").is_file()
    assert (out / "scripts" / "serve.py").is_file() and (out / "eval" / "llm_prices.json").is_file()
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\n") and "sdk: docker" in readme and "app_port: 8000" in readme
    assert "github.com/ThePunisher30/cairn" in readme


def test_the_space_folder_never_carries_data_notes_or_caches(tmp_path):
    out = tmp_path / "space"
    load().main(["--out", str(out)])
    names = {p.name for p in out.rglob("*")}
    assert "data" not in names and "learn" not in names and "__pycache__" not in names
    assert not list(out.rglob("*.npz")) and not list(out.rglob("*.pyc"))


def test_dense_flips_the_default_of_the_build_argument_in_every_stage(tmp_path):
    plain, dense = tmp_path / "plain", tmp_path / "dense"
    maker = load()
    maker.main(["--out", str(plain)])
    maker.main(["--out", str(dense), "--dense"])
    before = (plain / "Dockerfile").read_text(encoding="utf-8")
    after = (dense / "Dockerfile").read_text(encoding="utf-8")
    assert before.count("ARG WITH_DENSE=0") == 2 and "ARG WITH_DENSE=1" not in before
    assert after.count("ARG WITH_DENSE=1") == 2 and "ARG WITH_DENSE=0" not in after


def test_it_refuses_to_write_into_a_folder_that_is_not_empty(tmp_path, capsys):
    (tmp_path / "keep.txt").write_text("mine", encoding="utf-8")
    assert load().main(["--out", str(tmp_path)]) == 1
    assert "not empty" in capsys.readouterr().err
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "mine"
