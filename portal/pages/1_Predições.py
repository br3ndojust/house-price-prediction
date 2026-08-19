from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
import pandas as pd
import streamlit as st

from portal.lib.config import (
    CONFIDENCE_COLOR,
    COVERAGE_LABELS_PT,
    PRICE_BAND_LABEL,
    PROPERTY_FIELDS,
    sidebar_client,
)

st.set_page_config(page_title="Predições — Portal Inference", page_icon="🏠", layout="wide")
st.title("Predições")
st.caption(
    "Explore predições em tempo real. Ao prever, o portal também busca a explicação SHAP local — "
    "quais features empurraram o preço para cima/baixo neste imóvel."
)

client = sidebar_client()

_FIELD_KIND = {key: kind for key, _, kind, _, _ in PROPERTY_FIELDS}

with st.expander("Preencher todos os valores como uma linha (colar de planilha/CSV)"):
    st.caption(
        "Cole a ordem das colunas numa caixa e os valores correspondentes na outra, separados por "
        "vírgula — útil para colar direto de uma linha de planilha/CSV em vez de preencher campo a campo."
    )
    paste_col_order = st.text_input(
        "Ordem das colunas", value=", ".join(k for k, _, _, _, _ in PROPERTY_FIELDS), key="paste_col_order"
    )
    paste_col_values = st.text_input("Valores da linha (na mesma ordem)", key="paste_col_values")
    if st.button("Aplicar valores"):
        order = [c.strip() for c in paste_col_order.split(",") if c.strip()]
        raw_values = [v.strip() for v in paste_col_values.split(",")]
        valid_keys = set(_FIELD_KIND)
        unknown = [c for c in order if c not in valid_keys]
        if len(order) != len(raw_values):
            st.error(f"Ordem tem {len(order)} coluna(s), valores tem {len(raw_values)} — ajuste para o mesmo tamanho.")
        elif unknown:
            st.error(f"Coluna(s) desconhecida(s): {', '.join(unknown)}")
        else:
            try:
                parsed = {col: (int(float(raw)) if _FIELD_KIND[col] == "int" else float(raw))
                          for col, raw in zip(order, raw_values)}
            except ValueError as exc:
                st.error(f"Valor inválido: {exc}")
            else:
                missing = valid_keys - set(order)
                if missing:
                    st.warning(f"Coluna(s) não informada(s), mantidas com o valor atual: {', '.join(sorted(missing))}")
                for col, val in parsed.items():
                    st.session_state[f"in_{col}"] = val
                st.success('Valores aplicados abaixo — revise e clique em "Prever preço".')
                st.rerun()

with st.form("predict_form"):
    st.subheader("Atributos do imóvel")
    cols = st.columns(3)
    values: dict = {}
    for i, (key, label, kind, default, step) in enumerate(PROPERTY_FIELDS):
        col = cols[i % 3]
        if kind == "int":
            values[key] = col.number_input(label, value=int(default), step=step or 1, key=f"in_{key}")
        else:
            values[key] = col.number_input(
                label, value=float(default), step=step or 0.1, format="%.3f" if step else None, key=f"in_{key}"
            )
    submitted = st.form_submit_button("Prever preço")

if submitted:
    try:
        prediction = client.predict(values)
    except httpx.HTTPStatusError as exc:
        st.error(f"Erro da API ({exc.response.status_code}): {exc.response.text}")
    except httpx.HTTPError as exc:
        st.error(f"Não foi possível falar com a API: {exc}")
    else:
        # guardado em session_state pra sobreviver ao rerun do botão de feedback (form separado, abaixo)
        st.session_state["prediction_result"] = prediction
        st.session_state["prediction_values"] = values
        st.session_state.pop("feedback_done_for", None)

if "prediction_result" in st.session_state:
    prediction = st.session_state["prediction_result"]
    values = st.session_state["prediction_values"]

    st.divider()
    st.subheader("Registrar valor real de venda")
    st.caption(
        "Fecha o ciclo de aprendizado contínuo (docs/08_continuous_learning.md) — o valor fica "
        "disponível pra API usar em avaliação de performance e retraining do modelo."
    )
    prediction_id = prediction["prediction_id"]
    already_done = st.session_state.get("feedback_done_for") == prediction_id
    if already_done:
        st.success(f"Valor real já registrado para a predição `{prediction_id}`.")
        actual = st.session_state.get("feedback_actual_price")
        if actual is not None:
            predicted = prediction["predicted_price"]
            erro_abs = abs(predicted - actual)
            erro_pct = (erro_abs / actual * 100) if actual else 0.0
            st.markdown("**Comparativo — preço previsto vs. preço registrado**")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Preço previsto", f"US$ {predicted:,.0f}")
            cc2.metric("Preço registrado (real)", f"US$ {actual:,.0f}")
            cc3.metric("Erro", f"US$ {erro_abs:,.0f}", delta=f"{erro_pct:.1f}%", delta_color="inverse")
    else:
        with st.form("feedback_form"):
            actual_price = st.number_input(
                "Valor real (US$)", min_value=0.0, value=float(prediction["predicted_price"]), step=1000.0
            )
            feedback_submitted = st.form_submit_button("Registrar valor real")
        if feedback_submitted:
            try:
                client.record_feedback(prediction_id, actual_price)
            except httpx.HTTPStatusError as exc:
                st.error(f"Erro da API ({exc.response.status_code}): {exc.response.text}")
            except httpx.HTTPError as exc:
                st.error(f"Não foi possível falar com a API: {exc}")
            else:
                st.session_state["feedback_done_for"] = prediction_id
                st.session_state["feedback_actual_price"] = actual_price
                st.rerun()

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Preço previsto", f"US$ {prediction['predicted_price']:,.0f}")
    c2.metric(PRICE_BAND_LABEL, prediction["price_band"])
    c3.metric("Versão do modelo", prediction["model_version"])

    st.subheader("Confiança da previsão")
    if prediction.get("confidence_score") is None:
        st.info(
            "Calibração de confiança ainda não gerada — rode "
            "`python src/scripts/build_confidence_matrix.py`."
        )
    else:
        score = prediction["confidence_score"]
        category = prediction["confidence_category"]
        color = CONFIDENCE_COLOR.get(category, "⚪")
        cc1, cc2 = st.columns([1, 3])
        cc1.metric("Confidence Score", f"{score:.0f}/100")
        cc2.markdown(f"### {color} {category}")
        st.progress(min(max(score, 0), 100) / 100)

        try:
            detail = client.prediction_confidence(values)
        except httpx.HTTPError as exc:
            st.caption(f"Detalhe de confiança indisponível: {exc}")
        else:
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("Erro % esperado", f"{detail['expected_ape']:.1%}")
            dc2.metric("Erro US$ esperado", f"US$ {detail['expected_abs_error']:,.0f}")
            dc3.metric(
                "Representatividade no dados de treino",
                COVERAGE_LABELS_PT.get(detail["coverage_bucket"], detail["coverage_bucket"]),
            )
            dc4.metric("Amostra do segmento", detail["segment_n"])
            if detail["low_sample"]:
                st.warning("Amostra pequena mesmo após backoff — confiança neste segmento é indicativa, não conclusiva (P4).")
            if detail["hard_segment"]:
                st.warning("Segmento conhecido de erro maior (waterfront=1 ou grade≥10).")

    st.subheader("Explicação local (SHAP)")
    try:
        explanation = client.explain(values)
    except httpx.HTTPError as exc:
        st.warning(f"Não foi possível obter a explicação: {exc}")
    else:
        df = pd.DataFrame(explanation["contributions"])
        df = df.sort_values("pct_of_total_abs_contribution", ascending=True)
        st.bar_chart(df.set_index("feature")["shap_value_log"], horizontal=True)
        st.dataframe(df, width='stretch', hide_index=True)
