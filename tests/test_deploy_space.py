"""Tests for scripts/deploy_space.py: the Space folder must be complete, and nothing is uploaded in tests."""
from __future__ import annotations

import ast
import importlib.util

from conftest import ROOT


def _load_script():
    spec = importlib.util.spec_from_file_location("deploy_space", ROOT / "scripts" / "deploy_space.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _src_imports(path) -> set[str]:
    """Names of the src modules a Python file imports, e.g. {"config", "predict"}."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
            found.add(node.module.split(".")[1])
        elif isinstance(node, ast.ImportFrom) and node.module == "src":
            found.update(alias.name for alias in node.names)
    return found


def test_space_folder_has_the_layout_a_space_needs(tmp_path):
    script = _load_script()
    files = script.build_space_folder(tmp_path)
    for needed in ("app.py", "README.md", "requirements.txt", "configs/distilbert.yaml", "src/__init__.py"):
        assert needed in files and (tmp_path / needed).is_file()
    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\n") and "sdk: gradio" in card and "app_file: app.py" in card


def test_every_src_module_the_app_needs_is_copied(tmp_path):
    """If someone adds an import to the app or to a module it uses, the Space would crash on start."""
    script = _load_script()
    script.build_space_folder(tmp_path)
    to_check, seen = [tmp_path / "app.py"], set()
    while to_check:
        for module in _src_imports(to_check.pop()) - seen:
            seen.add(module)
            path = tmp_path / "src" / f"{module}.py"
            assert path.is_file(), f"src/{module}.py is imported by the demo but not copied to the Space"
            to_check.append(path)
    assert {"demo", "predict", "config"} <= seen


def test_space_requirements_pin_transformers_like_the_project(tmp_path):
    script = _load_script()
    script.build_space_folder(tmp_path)
    requirements = (tmp_path / "requirements.txt").read_text(encoding="utf-8")
    assert "transformers>=4.46,<5" in requirements and "torch" in requirements
    assert "gradio" not in requirements, "the Space installs Gradio itself, from sdk_version in README.md"


def test_deploy_creates_a_gradio_space_and_uploads_the_folder(tmp_path):
    script = _load_script()
    calls = []

    class FakeHub:
        def create_repo(self, repo_id, **kwargs):
            calls.append(("create", repo_id, kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload", kwargs["repo_id"], kwargs))

    url = script.deploy("someone/demo", tmp_path, api=FakeHub())
    assert url == "https://huggingface.co/spaces/someone/demo"
    assert calls[0] == ("create", "someone/demo", {"repo_type": "space", "space_sdk": "gradio", "exist_ok": True})
    assert calls[1][0] == "upload" and calls[1][2]["repo_type"] == "space" and calls[1][2]["folder_path"] == str(tmp_path)
