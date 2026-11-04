"""Loads the store's CSV tables and answers structured queries over them.

All business rules that depend on *data* (availability, active promotions,
order ownership) live here as plain Python, so they are deterministic and
testable without an LLM. The agent only decides *which* query to run.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PREFIX = "desafio_tecnico_ai_eng - "

PIX_DISCOUNT_PERCENT = 5  # Policy 3: 5% off list price, not cumulative with promotions (6.2)

ORDER_STATUS_PT = {
    "pending": "pendente (aguardando confirmação do pagamento)",
    "confirmed": "confirmado (em separação, ainda não enviado)",
    "shipped": "enviado",
    "delivered": "entregue",
    "cancelled": "cancelado",
}


def normalize(text: str) -> str:
    """Lowercase and strip accents, so 'Violão' matches 'violao'."""
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in text if not unicodedata.combining(c)).lower().strip()


def _spec_keywords(specs: dict) -> str:
    words = []
    if specs.get("strings") == "steel":
        words.append("aço")
    elif specs.get("strings") == "nylon":
        words.append("nylon")
    if specs.get("electronics") == "yes":
        words.append("elétrico eletroacústico")
    return " ".join(words)


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{PREFIX}{name}.csv", encoding="utf-8")


@lru_cache(maxsize=1)
def load_tables() -> dict[str, pd.DataFrame]:
    categories = _read("categories")
    products = _read("products")
    promotions = _read("promotions")
    orders = _read("orders")
    order_items = _read("order_items")
    customers = _read("customers")

    # Only active promotions are valid (policy 7.3: never promise an expired discount).
    # If a product ever had two active promotions, keep the biggest (they are not cumulative).
    active = (
        promotions[promotions["is_active"] == 1]
        .sort_values("discount_percent", ascending=False)
        .drop_duplicates("product_id")
        [["product_id", "discount_percent", "description"]]
        .rename(columns={"description": "promotion_name"})
    )

    products = products.merge(
        categories[["category_id", "name"]].rename(columns={"name": "category"}),
        on="category_id",
        how="left",
    ).merge(active, on="product_id", how="left")
    products["specs"] = products["specs"].apply(lambda s: json.loads(s) if isinstance(s, str) else {})
    # Customers search by features ("aço", "elétrico"), which are in specs, not always in the name.
    products["search_text"] = (
        products["name"] + " " + products["category"] + " " + products["specs"].map(_spec_keywords)
    ).map(normalize)

    return {
        "categories": categories,
        "products": products,
        "orders": orders,
        "order_items": order_items,
        "customers": customers,
    }


def category_names() -> list[str]:
    return load_tables()["categories"]["name"].tolist()


def _availability(row: pd.Series) -> str:
    if row["status"] == "discontinued":
        return "discontinued"
    if row["status"] == "coming_soon":
        return "coming_soon"
    if row["stock_quantity"] <= 0:
        return "out_of_stock"
    return "available"


def _product_view(row: pd.Series, with_details: bool = False) -> dict:
    price = float(row["price_brl"])
    view = {
        "product_id": int(row["product_id"]),
        "name": row["name"],
        "category": row["category"],
        "list_price_brl": price,
        "availability": _availability(row),
        "stock_quantity": int(row["stock_quantity"]),
    }
    if pd.notna(row.get("discount_percent")):
        pct = int(row["discount_percent"])
        view["promotion"] = {
            "name": row["promotion_name"],
            "discount_percent": pct,
            "promo_price_brl": round(price * (1 - pct / 100), 2),
        }
        view["pix_price_brl"] = None  # PIX discount does not stack with promotions (policy 6.2)
    else:
        view["pix_price_brl"] = round(price * (1 - PIX_DISCOUNT_PERCENT / 100), 2)
    if with_details:
        view["description"] = row["description"]
        view["specs"] = row["specs"]
    return view


def search_products(
    query: str | None = None,
    category: str | None = None,
    max_price: float | None = None,
    min_price: float | None = None,
    limit: int = 10,
) -> dict:
    """Filter the catalog.

    Without a `query`, the customer is browsing, so only items available to buy
    are returned. With a `query` (a specific product name), every match is
    returned with its availability, so the agent can say "out of stock" or
    "discontinued" instead of "not found".
    """
    df = load_tables()["products"]

    if category:
        wanted = normalize(category)
        df = df[df["category"].map(normalize).str.contains(wanted, regex=False)]
    if max_price is not None:
        df = df[df["price_brl"] <= max_price]
    if min_price is not None:
        df = df[df["price_brl"] >= min_price]
    if query:
        tokens = [t for t in re.split(r"[\s\-]+", normalize(query)) if t]
        df = df[df["search_text"].apply(lambda text: all(t in text for t in tokens))]
    else:
        df = df[(df["status"] == "active") & (df["stock_quantity"] > 0)]

    df = df.sort_values("price_brl")
    total = len(df)
    items = [_product_view(row, with_details=bool(query)) for _, row in df.head(limit).iterrows()]
    return {"total_matches": total, "returned": len(items), "products": items}


def _digits(text: str) -> str:
    return re.sub(r"\D", "", str(text))


def check_order(order_id: int, email_or_phone: str) -> dict:
    """Return an order only if the contact matches the customer who placed it.

    The same generic error is returned for "order not found" and "contact does
    not match", so the tool cannot be used to discover which order IDs exist (LGPD).
    """
    t = load_tables()
    not_verified = {
        "verified": False,
        "message": "Could not verify this order with the contact provided.",
    }

    order = t["orders"][t["orders"]["order_id"] == order_id]
    if order.empty:
        return not_verified
    order = order.iloc[0]
    customer = t["customers"][t["customers"]["customer_id"] == order["customer_id"]].iloc[0]

    contact = str(email_or_phone).strip()
    email_ok = normalize(contact) == normalize(customer["email"])
    phone_digits = _digits(contact)
    phone_ok = len(phone_digits) >= 8 and _digits(customer["phone"]).endswith(phone_digits[-9:])
    if not (email_ok or phone_ok):
        return not_verified

    items = t["order_items"][t["order_items"]["order_id"] == order_id].merge(
        t["products"][["product_id", "name"]], on="product_id", how="left"
    )

    def opt(value):
        return None if pd.isna(value) else value

    return {
        "verified": True,
        "order_id": int(order["order_id"]),
        "customer_first_name": customer["name"].split()[0],
        "order_date": order["order_date"],
        "status": order["status"],
        "status_pt": ORDER_STATUS_PT.get(order["status"], order["status"]),
        "total_brl": float(order["total_brl"]),
        "payment_method": order["payment_method"],
        "tracking_code": opt(order["tracking_code"]),
        "estimated_delivery": opt(order["estimated_delivery"]),
        "notes": opt(order["notes"]),
        "items": [
            {"product": row["name"], "quantity": int(row["quantity"])} for _, row in items.iterrows()
        ],
    }
