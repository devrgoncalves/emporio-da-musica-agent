from typing import get_args

from emporio import data_store, policies, tools


def test_every_topic_maps_to_existing_sections():
    sections = policies.load_sections()
    assert sorted(sections) == list(range(1, 11))
    for topic in get_args(policies.Topic):
        assert policies.get_policy(topic).startswith("## ")


def test_returns_policy_contains_the_7_day_regret_rule():
    text = policies.get_policy("returns")
    assert "7 (sete) dias corridos após o recebimento" in text
    assert "## 5." not in text  # does not leak the next section


def test_hours_policy_keeps_the_table():
    assert "| Sábado | 09:00 às 13:00 |" in policies.get_policy("hours")


def test_unknown_topic_lists_valid_ones():
    assert "Valid topics" in policies.get_policy("weather")


def test_category_enum_matches_the_csv():
    assert list(get_args(tools.Category)) == data_store.category_names()


def test_tools_expose_schemas_to_the_llm():
    names = {t.name for t in tools.ALL_TOOLS}
    assert names == {"search_products", "check_order", "get_policy"}
    schema = tools.check_order.args
    assert set(schema) == {"order_id", "email_or_phone"}
