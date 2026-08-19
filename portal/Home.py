"""Portal Inference — Visão geral (health, versão ativa, uptime). Streamlit, app externa: consome a API
só via HTTP, nunca importa `app/` diretamente (ver portal/README.md)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import streamlit as st

from portal.lib.config import HEALTH_STATUS_PT, sidebar_client

st.set_page_config(page_title="House Pricing — Portal Inference", page_icon="🏠", layout="wide")

st.title("Portal Inference — House Pricing API")
st.caption("Consome a API de inferência/monitoramento via HTTP.")

client = sidebar_client()

st.subheader("Visão geral")
try:
    health = client.health_detailed()
except httpx.HTTPError as exc:
    st.error(f"Não foi possível falar com a API em `{client.base_url}`: {exc}")
else:
    cols = st.columns(4)
    cols[0].metric("Status", HEALTH_STATUS_PT.get(health["status"], health["status"]))
    cols[1].metric("Uptime (s)", health["uptime_seconds"])

    active_version_id = health["active_model_version"]
    active_label = active_version_id or "—"
    if active_version_id:
        try:
            versions = client.model_versions()
        except httpx.HTTPError:
            versions = []
        matched = next((v for v in versions if v["version"] == active_version_id), None)
        if matched and matched.get("notes"):
            active_label = matched["notes"]
    cols[2].metric("Versão do modelo ativo", active_label, help=f"id interno: {active_version_id}")

    model = health.get("model") or {}
    cols[3].metric("MAE em val (US$)", f"{model.get('val_mae_dollar', 0):,.0f}" if model else "—")

    if model:
        st.markdown("**Contrato do modelo ativo**")
        st.json(model, expanded=False)

st.divider()
st.subheader("Páginas do portal")
PORTAL_PAGES = [
    ("pages/1_Predições.py", "🔮", "Predições", "Testar o modelo, registrar o valor real de venda, ver a explicação SHAP local."),
    ("pages/8_Inferência_em_Lote.py", "📦", "Inferência em Lote", "Upload de CSV para prever (e opcionalmente registrar valor real) em massa."),
]
for path, icon, title, description in PORTAL_PAGES:
    link_col, desc_col = st.columns([1, 3])
    link_col.page_link(path, label=title, icon=icon)
    desc_col.caption(description)
