def to_streamlit_markdown(text: str) -> str:
    """Escape "$" so Streamlit does not render text between two of them as LaTeX (e.g. "R$ 10 ... R$ 20")."""
    return text.replace("$", r"\$")
