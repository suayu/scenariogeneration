import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from scenario_generator import AdversarialScenarioGenerator


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_with_llm_provider.sh"


def _generator(model_names):
    generator = object.__new__(AdversarialScenarioGenerator)
    generator.cfg = SimpleNamespace(
        sim=SimpleNamespace(llm=SimpleNamespace(model_names=model_names))
    )
    return generator


def _provider_environment(tmp_path, provider, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)
    monkeypatch.delenv("LLM_MODEL_NAMES", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-placeholder")
    program = (
        "import json,os; "
        "print(json.dumps({'name': os.getenv('LLM_MODEL_NAME'), "
        "'names': os.getenv('LLM_MODEL_NAMES')}))"
    )
    completed = subprocess.run(
        ["bash", str(RUNNER), provider, sys.executable, "-c", program],
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def test_dashscope_runner_leaves_candidate_pool_to_planner(tmp_path, monkeypatch):
    environment = _provider_environment(tmp_path, "dashscope", monkeypatch)
    assert environment == {"name": "qwen-plus", "names": None}


def test_other_provider_remains_fixed_to_one_model(tmp_path, monkeypatch):
    environment = _provider_environment(tmp_path, "deepseek", monkeypatch)
    assert environment["name"] == "deepseek-v4-pro"
    assert environment["names"] == "deepseek-v4-pro"


def test_explicit_dashscope_pool_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL_NAMES", "model-a,model-b")
    monkeypatch.setenv("LLM_MODEL_NAME", "model-a")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only-placeholder")
    program = "import os; print(os.environ['LLM_MODEL_NAMES'])"
    completed = subprocess.run(
        ["bash", str(RUNNER), "dashscope", sys.executable, "-c", program],
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout.strip() == "model-a,model-b"


def test_generator_uses_configured_or_explicit_pool(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_NAME", "preferred-model")
    monkeypatch.delenv("LLM_MODEL_NAMES", raising=False)
    assert _generator(None)._read_llm_model_names() is None
    assert _generator(["configured-a", "configured-b"])._read_llm_model_names() == [
        "configured-a",
        "configured-b",
    ]
    monkeypatch.setenv("LLM_MODEL_NAMES", "explicit-a, explicit-b")
    assert _generator(None)._read_llm_model_names() == ["explicit-a", "explicit-b"]
