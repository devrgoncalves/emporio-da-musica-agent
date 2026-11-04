from emporio.ui_format import to_streamlit_markdown


def test_regressão_brl_prices_are_not_rendered_as_latex():
    # Bug: Streamlit treated the text between two "$" as a LaTeX formula,
    # so "R$ 699,90 (ou R$ 664,90 no PIX)" was shown as garbled math.
    text = "Yamaha F310 – R$ 699,90 (ou R$ 664,90 no PIX)"
    assert to_streamlit_markdown(text) == r"Yamaha F310 – R\$ 699,90 (ou R\$ 664,90 no PIX)"


def test_text_without_dollar_is_unchanged():
    assert to_streamlit_markdown("Olá! **Tudo bem?**") == "Olá! **Tudo bem?**"
