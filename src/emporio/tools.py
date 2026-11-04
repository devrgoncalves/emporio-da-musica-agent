"""Tools exposed to the LLM.

Each function is a thin wrapper: the docstring and type hints become the
schema the model reads to decide *when* to call it; the logic lives in
data_store.py / policies.py so it can be tested without an LLM.
"""

from __future__ import annotations

from typing import Literal

from langchain.tools import tool

from emporio import data_store, policies

Category = Literal[
    "Guitarras",
    "Baixos",
    "Baterias e Percussão",
    "Teclados e Pianos",
    "Violões",
    "Instrumentos de Sopro (Madeiras)",
    "Instrumentos de Sopro (Metais)",
    "Cordas Orquestrais",
    "Ukuleles",
]


@tool
def search_products(
    query: str | None = None,
    category: Category | None = None,
    max_price: float | None = None,
    min_price: float | None = None,
) -> dict:
    """Search the store catalog for price, stock, availability and active promotions.

    ALWAYS call this before telling the customer any price, stock or availability.

    - Use `query` with brand/model words when the customer names a specific
      product (e.g. "Takamine GD20"). Name searches return every match,
      including out-of-stock or discontinued ones, with their `availability`.
    - Without `query` the customer is browsing: only items available to buy
      are returned (e.g. category="Violões", max_price=1000).
    Prices are in BRL. `promotion` appears only when a promotion is active.
    `pix_price_brl` is the 5% PIX price; it is null when a promotion applies
    because discounts do not stack.
    """
    return data_store.search_products(query, category, max_price, min_price)


@tool
def check_order(order_id: int, email_or_phone: str) -> dict:
    """Look up the status, tracking code and items of a customer's order.

    Requires BOTH the order number and the e-mail or phone used in the purchase.
    Ask the customer for them if missing; never guess. If `verified` is false,
    say you could not confirm the order with that data and ask them to check it.
    """
    return data_store.check_order(order_id, email_or_phone)


@tool
def get_policy(topic: policies.Topic) -> str:
    """Read the store's official policy manual for a topic.

    Topics: about (store info, address, contacts, what the store sells),
    hours (opening hours), payment (methods, installments, PIX discount),
    returns (regret/defect/preference exchanges, non-eligible items),
    shipping (delivery fees, carriers, deadlines, tracking),
    promotions (campaign calendar and promotion rules),
    customer_service (tone, scope, out-of-stock and complaint procedures),
    warranty (legal and manufacturer warranty), privacy (LGPD, personal data).

    Call this before answering any question about rules, conditions or store information.
    """
    return policies.get_policy(topic)


ALL_TOOLS = [search_products, check_order, get_policy]
