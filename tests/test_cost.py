from types import SimpleNamespace

from llm.cost import estimate_cost, price_for, usage_from_result


def test_price_lookup_known_and_unknown():
    p = price_for("gpt-4o")
    assert p[0] > 0 and p[1] > 0
    fallback = price_for("totally-made-up-model")
    assert fallback == (0.0025, 0.0100)


def test_estimate_cost_arith():
    # gpt-4o-mini: 0.00015 / 0.0006 per 1k
    cost = estimate_cost("gpt-4o-mini", 1000, 1000)
    assert abs(cost - (0.00015 + 0.0006)) < 1e-9


def test_usage_from_result_robust_field_names():
    # PydanticAI's usage shape has varied; we should handle both names.
    fake_usage = SimpleNamespace(request_tokens=42, response_tokens=17)
    fake_result = SimpleNamespace(usage=lambda: fake_usage)
    tin, tout = usage_from_result(fake_result)
    assert tin == 42 and tout == 17

    fake_usage2 = SimpleNamespace(input_tokens=10, output_tokens=20)
    fake_result2 = SimpleNamespace(usage=fake_usage2)
    tin2, tout2 = usage_from_result(fake_result2)
    assert tin2 == 10 and tout2 == 20

    none_result = SimpleNamespace(usage=lambda: None)
    assert usage_from_result(none_result) == (0, 0)
