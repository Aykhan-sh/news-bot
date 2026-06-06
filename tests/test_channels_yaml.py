from pathlib import Path

from orchestrator.models import load_channels_dir

ROOT = Path(__file__).resolve().parent.parent


def test_bundled_channels_load_and_have_required_fields():
    specs = load_channels_dir(ROOT / "channels")
    assert specs, "no channels found"
    ids = {s["id"] for s in specs}
    assert {"ai_career", "arabic", "kazakh", "math"} <= ids
    for s in specs:
        for k in (
            "id",
            "display_name",
            "hashtag",
            "mode",
            "model_writer",
            "topic_prompt",
            "schedule",
        ):
            assert k in s, f"channel {s.get('id')} missing {k}"
        assert s["mode"] in ("sourced", "llm_only")
        assert s["schedule"]["kind"] in ("cron", "interval", "probabilistic")
