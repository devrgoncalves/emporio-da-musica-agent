from emporio import data_store as ds


def names(result):
    return [p["name"] for p in result["products"]]


def test_browse_guitars_under_1000_only_returns_items_you_can_buy():
    result = ds.search_products(category="Violões", max_price=1000, limit=50)
    assert result["total_matches"] > 0
    for p in result["products"]:
        assert p["list_price_brl"] <= 1000
        assert p["availability"] == "available"
    # Giannini GF-3D (id 96) costs 799.90 but has zero stock
    assert "Giannini GF-3D Dreadnought Sunburst" not in names(result)


def test_browse_is_sorted_by_price():
    prices = [p["list_price_brl"] for p in ds.search_products(category="Violões", limit=50)["products"]]
    assert prices == sorted(prices)


def test_search_by_name_ignores_case_and_accents():
    result = ds.search_products(query="takamine gd20")
    assert names(result) == ["Takamine GD20 Dreadnought Natural"]
    assert result["products"][0]["list_price_brl"] == 2199


def test_search_by_name_still_shows_out_of_stock_items():
    result = ds.search_products(query="Giannini GF-3D")
    assert result["products"][0]["availability"] == "out_of_stock"


def test_search_by_name_shows_discontinued_items():
    result = ds.search_products(query="Shelby SN-7C")
    assert result["products"][0]["availability"] == "discontinued"


def test_active_promotion_is_applied_and_pix_does_not_stack():
    # Crafter HT-100 (id 90) has the active 18% "Desconto Exclusivo"
    p = ds.search_products(query="Crafter HT-100")["products"][0]
    assert p["promotion"]["discount_percent"] == 18
    assert p["promotion"]["promo_price_brl"] == round(2399 * 0.82, 2)
    assert p["pix_price_brl"] is None


def test_expired_promotion_is_ignored():
    # Martin D-28 (id 93) only has inactive promotions
    p = ds.search_products(query="Martin D-28")["products"][0]
    assert "promotion" not in p
    assert p["pix_price_brl"] == round(11499 * 0.95, 2)


def test_regressão_steel_string_filter_uses_specs_not_only_the_name():
    # Bug: query="aço" missed the Tagima Woodstock (steel strings in specs, no "Aço" in its name).
    result = ds.search_products(query="aço", category="Violões", max_price=1000, limit=50)
    assert "Tagima Woodstock Dreadnought Natural" in names(result)
    assert all(p["specs"]["strings"] == "steel" for p in result["products"])


def test_electric_filter_uses_specs():
    result = ds.search_products(query="elétrico", category="Violões", limit=50)
    assert result["total_matches"] > 0
    assert all(p["specs"]["electronics"] == "yes" for p in result["products"])


def test_unknown_product_returns_no_matches():
    assert ds.search_products(query="Steinway piano de cauda")["total_matches"] == 0


def test_order_lookup_with_matching_email():
    # Order 8 belongs to customer 2 (Ana Carolina Ferreira)
    order = ds.check_order(8, "anacarol.ferreira@coldmail.com")
    assert order["verified"] is True
    assert order["status"] == "shipped"
    assert order["tracking_code"] == "BRJL5544332BR"
    assert order["customer_first_name"] == "Ana"


def test_order_lookup_with_matching_phone_in_any_format():
    assert ds.check_order(8, "67 99654-7821")["verified"] is True
    assert ds.check_order(8, "(67) 99654-7821")["verified"] is True


def test_order_lookup_with_wrong_contact_reveals_nothing():
    result = ds.check_order(8, "someone.else@jmail.com")
    assert result == {"verified": False, "message": "Could not verify this order with the contact provided."}


def test_nonexistent_order_gives_same_answer_as_wrong_contact():
    assert ds.check_order(999, "anacarol.ferreira@coldmail.com") == ds.check_order(8, "wrong@x.com")


def test_cancelled_order_includes_reason():
    order = ds.check_order(17, "matheus.correia@hayoo.com.br")
    assert order["status"] == "cancelled"
    assert "cancelado pelo cliente" in order["notes"]
