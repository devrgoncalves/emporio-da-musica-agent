"""Streamlit chat UI for the Empório da Música agent.

Run with:  streamlit run app.py
"""

import json
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from emporio.agent import build_agent, chat  # noqa: E402

st.set_page_config(page_title="Empório da Música — Atendimento", page_icon="🎸")


@st.cache_resource
def get_agent():
    # One agent per server process; each browser session has its own thread_id.
    return build_agent()


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []  # [{"role", "content", "tool_calls"}]

with st.sidebar:
    st.header("🎸 Empório da Música")
    st.caption("Sua música começa aqui.")
    st.markdown(
        "Protótipo de agente de atendimento.\n\n"
        "**Experimente:**\n"
        "- Quais violões até R$ 1000?\n"
        "- Quanto custa o Takamine GD20?\n"
        "- Qual o status do pedido 8? (e-mail: anacarol.ferreira@coldmail.com)\n"
        "- Me arrependi da compra, posso devolver?\n"
        "- Vocês vendem cordas?"
    )
    show_tools = st.toggle("Mostrar ferramentas consultadas", value=True)
    if st.button("Nova conversa"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

st.title("Atendimento Empório da Música")

try:
    agent = get_agent()
except RuntimeError as err:
    st.error(str(err))
    st.stop()


def render_tool_calls(calls):
    if not (show_tools and calls):
        return
    with st.expander(f"🔧 {len(calls)} consulta(s): " + ", ".join(c["name"] for c in calls)):
        for c in calls:
            st.markdown(f"**{c['name']}** `{json.dumps(c['args'], ensure_ascii=False)}`")
            st.code(str(c["result"])[:1500], language="json")


for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_tool_calls(msg["tool_calls"])

if prompt := st.chat_input("Digite sua mensagem..."):
    st.session_state.history.append({"role": "user", "content": prompt, "tool_calls": []})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Consultando..."):
            try:
                turn = chat(agent, st.session_state.thread_id, prompt)
                reply, calls = turn.reply, turn.tool_calls
            except Exception as err:  # network / quota errors should not crash the chat
                reply, calls = f"Desculpe, tive um problema técnico agora ({type(err).__name__}). Pode tentar de novo?", []
        st.markdown(reply)
        render_tool_calls(calls)
    st.session_state.history.append({"role": "assistant", "content": reply, "tool_calls": calls})
