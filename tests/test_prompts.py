from datetime import datetime, timezone

from emporio.prompts import build_system_prompt


def test_prompt_contains_store_local_time_and_weekday():
    # 2026-11-04 15:30 UTC is 11:30 in Campo Grande (UTC-4), a Wednesday
    prompt = build_system_prompt(datetime(2026, 11, 4, 15, 30, tzinfo=timezone.utc))
    assert "quarta-feira, 04/11/2026 11:30" in prompt
    assert "{now}" not in prompt
