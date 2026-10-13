from pathlib import Path

from policies.llm_adversarial_planner import LLMAdversarialPlanner, ModelOutputError


class FakeAPIError(Exception):
    def __init__(self, status_code, code, message):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.body = {"code": code, "message": message}


def _planner(models):
    planner = object.__new__(LLMAdversarialPlanner)
    planner.provider = "dashscope"
    planner.model_names = tuple(models)
    planner._preferred_model_index = 0
    planner.model_name = models[0]
    return planner


def test_explicit_free_quota_exhaustion_disables_model_and_falls_back(monkeypatch):
    planner = _planner(["quota-empty", "available"])
    calls = []
    commented = []

    def request_once(model_name, prompt, scene_image):
        calls.append(model_name)
        if model_name == "quota-empty":
            raise FakeAPIError(
                403,
                "AllocationQuota.FreeTierOnly",
                "Free quota expired or exhausted",
            )
        return "", {"attack": False}

    monkeypatch.setattr(planner, "_request_attack_plan_once", request_once)
    monkeypatch.setattr(
        planner,
        "_comment_out_exhausted_model",
        lambda model_name: commented.append(model_name) or True,
    )

    assert planner._request_attack_plan("prompt") == ("", {"attack": False})
    assert calls == ["quota-empty", "available"]
    assert planner.model_names == ("available",)
    assert planner.model_name == "available"
    assert commented == ["quota-empty"]


def test_transient_rate_limit_falls_back_without_permanent_disable(monkeypatch):
    planner = _planner(["busy", "available"])
    commented = []

    def request_once(model_name, prompt, scene_image):
        if model_name == "busy":
            raise FakeAPIError(429, "Throttling.RateQuota", "Requests per minute exceeded")
        return "", {"attack": False}

    monkeypatch.setattr(planner, "_request_attack_plan_once", request_once)
    monkeypatch.setattr(
        planner,
        "_comment_out_exhausted_model",
        lambda model_name: commented.append(model_name) or True,
    )

    planner._request_attack_plan("prompt")
    assert planner.model_names == ("busy", "available")
    assert commented == []


def test_generic_permission_error_is_not_misclassified_as_quota_exhaustion():
    error = FakeAPIError(403, "AccessDenied", "Model access has not been granted")
    assert not LLMAdversarialPlanner._is_free_quota_exhausted(error)


def test_commented_dashscope_default_is_not_reintroduced():
    planner = LLMAdversarialPlanner(
        model_name="qwen3.5-plus",
        model_names=None,
        provider="dashscope",
        client=object(),
    )
    assert "qwen3.5-plus" not in planner.model_names


def test_non_dashscope_provider_does_not_inherit_dashscope_pool():
    planner = LLMAdversarialPlanner(
        model_name="deepseek-test",
        model_names=None,
        provider="deepseek",
        client=object(),
    )
    assert planner.model_names == ("deepseek-test",)


def test_comment_out_exhausted_model_changes_only_active_pool_entry(tmp_path):
    source = tmp_path / "planner.py"
    source.write_text(
        "FREE_MODEL_POOL = (\n"
        "    # \"already-off\",\n"
        "    \"quota-empty\",\n"
        "    \"still-active\",\n"
        ")\n"
        "OTHER_MODELS = (\"quota-empty\",)\n",
        encoding="utf-8",
    )

    assert LLMAdversarialPlanner._comment_out_exhausted_model("quota-empty", source)
    updated = source.read_text(encoding="utf-8")
    assert '    # "quota-empty",  # 百炼免费额度已耗尽，自动停用' in updated
    assert '    "still-active",\n' in updated
    assert 'OTHER_MODELS = ("quota-empty",)' in updated
    assert not LLMAdversarialPlanner._comment_out_exhausted_model("quota-empty", source)


def test_explicit_pool_filters_models_marked_exhausted_in_source():
    planner = LLMAdversarialPlanner(
        model_name="qwen-plus",
        model_names=["qwen-plus", "qwen3-8b"],
        provider="dashscope",
        client=object(),
    )
    assert planner.model_names == ("qwen3-8b",)


def test_exhausted_model_parser_only_reads_permanent_markers(tmp_path):
    source = tmp_path / "planner.py"
    source.write_text(
        '    # "manual-comment",\n'
        '    # "quota-empty",  # 百炼免费额度已耗尽，自动停用\n'
        '    "active",\n',
        encoding="utf-8",
    )
    assert LLMAdversarialPlanner._exhausted_models_from_source(source) == {
        "quota-empty"
    }


def test_successful_fallback_records_public_model_name(monkeypatch):
    planner = _planner(["quota-empty", "available"])
    monkeypatch.setattr(
        planner,
        "_comment_out_exhausted_model",
        lambda model_name: True,
    )

    def request_once(model_name, prompt, scene_image):
        if model_name == "quota-empty":
            raise FakeAPIError(
                403,
                "AllocationQuota.FreeTierOnly",
                "Free quota expired or exhausted",
            )
        return "", {"attack": False}

    monkeypatch.setattr(planner, "_request_attack_plan_once", request_once)
    planner._request_attack_plan("prompt")
    assert planner.last_attempted_model_name == "available"
    assert planner.last_successful_model_name == "available"


def test_invalid_json_output_falls_back_without_service_failure(monkeypatch):
    planner = _planner(["bad-json", "available"])

    def request_once(model_name, prompt, scene_image):
        if model_name == "bad-json":
            raise ModelOutputError("纯文本大模型输出不是有效 JSON")
        return "", {"attack": False}

    monkeypatch.setattr(planner, "_request_attack_plan_once", request_once)
    assert planner._request_attack_plan("prompt") == ("", {"attack": False})
    assert planner.last_attempted_model_names == ["bad-json", "available"]
    assert planner.last_model_failures == [{
        "model": "bad-json",
        "category": "invalid_json_output",
        "status_code": None,
    }]


def test_all_invalid_json_outputs_are_not_service_failure(monkeypatch):
    planner = _planner(["bad-a", "bad-b"])
    monkeypatch.setattr(
        planner,
        "_request_attack_plan_once",
        lambda model_name, prompt, scene_image: (_ for _ in ()).throw(
            ModelOutputError("纯文本大模型输出不是有效 JSON")
        ),
    )
    try:
        planner._request_attack_plan("prompt")
    except ModelOutputError as error:
        assert not planner._is_service_failure(error)
    else:
        raise AssertionError("应保留无效 JSON 输出错误")
