"""End-to-end tests: real Gemini calls through the same `chat()` used by the UI.

They check behaviour (which tool was used, key facts in the reply), not exact
wording, because LLM output varies. Skipped when no API key is configured.
Run only these with:  pytest -m llm
"""

import os
import uuid

import pytest
from dotenv import load_dotenv

load_dotenv()
pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")), reason="needs GOOGLE_API_KEY"
    ),
]


@pytest.fixture(scope="module")
def agent():
    from emporio.agent import build_agent

    return build_agent()


def ask(agent, *messages):
    from emporio.agent import chat

    thread = str(uuid.uuid4())
    turns = [chat(agent, thread, m) for m in messages]
    return turns[-1], [c["name"] for t in turns for c in t.tool_calls]


def test_price_question_uses_catalog_tool(agent):
    turn, tools = ask(agent, "Quanto custa o Takamine GD20?")
    assert "search_products" in tools
    assert "2.199" in turn.reply


def test_browsing_excludes_out_of_stock_items(agent):
    turn, tools = ask(agent, "Quais opções de violões disponíveis custando até R$1000?")
    assert "search_products" in tools
    assert "GF-3D" not in turn.reply  # Giannini GF-3D has zero stock


def test_store_address_comes_from_policy(agent):
    turn, tools = ask(agent, "Qual o endereço da loja?")
    assert "get_policy" in tools
    assert "14 de Maio" in turn.reply


def test_order_status_requires_verification(agent):
    turn, tools = ask(agent, "Qual o status do meu pedido 8?")
    assert "check_order" not in tools  # must ask for e-mail/phone first
    turn, tools = ask(agent, "Qual o status do meu pedido 8? Meu e-mail é anacarol.ferreira@coldmail.com")
    assert "check_order" in tools
    assert "BRJL5544332BR" in turn.reply


def test_regressão_return_is_not_promised_before_checking_the_order(agent):
    # Bug: the agent answered "Claro, você pode devolver, sim" before looking at the order,
    # and later offered a 30-day defect exchange that had already expired.
    from emporio.agent import chat

    thread = str(uuid.uuid4())
    first = chat(agent, thread, "Me arrependi da minha compra, posso devolver meu pedido?")
    assert not any(c["name"] == "check_order" for c in first.tool_calls)
    assert "pode sim" not in first.reply.lower() and "claro" not in first.reply.lower()

    second = chat(agent, thread, "Pedido 4, email lucas.mendes@jmail.com")
    names = [c["name"] for c in second.tool_calls]
    assert "check_order" in names and "get_policy" in names
    assert "22/12/2025" in second.reply  # states the date it used
    assert "30 dias" not in second.reply  # defect window (from 10/12/2025) has expired too


def test_accessories_are_out_of_scope(agent):
    turn, _ = ask(agent, "Vocês vendem cordas de violão?")
    assert "instrumento" in turn.reply.lower()
