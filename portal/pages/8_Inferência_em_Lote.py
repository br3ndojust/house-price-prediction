"""Inferência em lote via upload de CSV — checa colunas antes de chamar a API (POST /predictions/batch,
respeita o limite de 500 imóveis por chamada, faz chunking automático) e, se o CSV trouxer uma coluna
de valor real (ex: `price`), permite registrar feedback em lote (POST /feedback) — fecha o ciclo de
aprendizado contínuo pra várias predições de uma vez."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import altair as alt
import httpx
import numpy as np
import pandas as pd
import streamlit as st

from portal.lib.api_client import ApiClient
from portal.lib.config import (
    CONFIDENCE_COLOR,
    COVERAGE_LABELS_PT,
    PROPERTY_FIELDS,
    REQUIRED_CSV_COLUMNS,
    VALUE_COLUMN_CANDIDATES,
    sidebar_client,
)

CATEGORY_LABEL = "Categoria do Imóvel"  # pedido específico desta tela (outras telas usam "Segmento do Imóvel")

st.set_page_config(page_title="Inferência em Lote — Portal Inference", page_icon="🏠", layout="wide")
st.title("Inferência em lote (upload de CSV)")
st.caption(
    "Colunas obrigatórias (mesmo schema de `future_unseen_examples.csv`): "
    + ", ".join(REQUIRED_CSV_COLUMNS)
)

client = sidebar_client()

_FIELD_KIND = {key: kind for key, _, kind, _, _ in PROPERTY_FIELDS}


def _render_row_detail(row_number: int, payload: dict, result: dict, client: ApiClient) -> None:
    """Mesmas informações mostradas em Predições (single inference) para a linha selecionada na tabela."""
    st.divider()
    st.subheader(f"Detalhe da linha {row_number}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Preço previsto", f"US$ {result['predicted_price']:,.0f}")
    c2.metric(CATEGORY_LABEL, result["price_band"])
    c3.metric("Versão do modelo", result["model_version"])

    st.markdown("**Confiança da previsão**")
    if result.get("confidence_score") is None:
        st.info("Calibração de confiança ainda não gerada — rode `python src/scripts/build_confidence_matrix.py`.")
    else:
        score, category = result["confidence_score"], result["confidence_category"]
        cc1, cc2 = st.columns([1, 3])
        cc1.metric("Confidence Score", f"{score:.0f}/100")
        cc2.markdown(f"### {CONFIDENCE_COLOR.get(category, '⚪')} {category}")
        try:
            detail = client.prediction_confidence(payload)
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

    st.markdown("**Explicação local (SHAP)**")
    try:
        explanation = client.explain(payload)
    except httpx.HTTPError as exc:
        st.warning(f"Não foi possível obter a explicação: {exc}")
    else:
        shap_df = pd.DataFrame(explanation["contributions"]).sort_values(
            "pct_of_total_abs_contribution", ascending=True
        )
        st.bar_chart(shap_df.set_index("feature")["shap_value_log"], horizontal=True)
        st.dataframe(shap_df, width='stretch', hide_index=True)

    with st.expander("Atributos enviados"):
        st.json(payload)


uploaded = st.file_uploader("Arquivo CSV", type=["csv"])

column_order_raw = st.text_input(
    "Ordem das colunas do CSV (opcional, separado por vírgula)",
    help=(
        "Preencha só se o arquivo não tiver cabeçalho, ou o cabeçalho não bater com os nomes "
        "esperados — informe os nomes na mesma ordem das colunas do arquivo. Deixe em branco para "
        "usar o cabeçalho do próprio arquivo."
    ),
)
skip_header_row = True
if column_order_raw.strip():
    skip_header_row = st.checkbox("A primeira linha do arquivo é um cabeçalho (ignorar)", value=True)

if uploaded is None:
    st.info("Envie um arquivo CSV para começar.")
    st.stop()

file_signature = f"{uploaded.name}:{uploaded.size}"
if st.session_state.get("batch_file_signature") != file_signature:
    st.session_state["batch_file_signature"] = file_signature
    st.session_state.pop("batch_out", None)  # novo arquivo -> descarta resultado anterior

# ---- leitura ------------------------------------------------------------------------------
try:
    if column_order_raw.strip():
        override_columns = [c.strip() for c in column_order_raw.split(",") if c.strip()]
        df = pd.read_csv(uploaded, header=0 if skip_header_row else None)
        if len(override_columns) != df.shape[1]:
            st.error(
                f"A ordem de colunas informada tem {len(override_columns)} nome(s), mas o arquivo tem "
                f"{df.shape[1]} coluna(s). Ajuste a lista separada por vírgula."
            )
            st.stop()
        df.columns = override_columns
    else:
        df = pd.read_csv(uploaded, header=0)
except Exception as exc:  # arquivo malformado — feedback direto, sem traceback cru
    st.error(f"Não foi possível ler o CSV: {exc}")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]

# ---- checagem de colunas obrigatórias ------------------------------------------------------
missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
if missing:
    st.error(
        "Colunas obrigatórias faltando no CSV: **" + ", ".join(missing) + "**.\n\n"
        "Colunas encontradas no arquivo: " + ", ".join(df.columns) + ".\n\n"
        "Se o arquivo não tem cabeçalho (ou tem nomes diferentes), preencha o campo "
        "\"Ordem das colunas do CSV\" acima com os nomes na ordem correta."
    )
    st.stop()

extra_columns = [c for c in df.columns if c not in REQUIRED_CSV_COLUMNS]
value_column = None
if extra_columns:
    auto_candidates = [c for c in extra_columns if c.lower() in VALUE_COLUMN_CANDIDATES]
    default_idx = extra_columns.index(auto_candidates[0]) + 1 if auto_candidates else 0
    value_column = st.selectbox(
        "Coluna com o valor real de venda (opcional)",
        options=["(nenhuma)"] + extra_columns,
        index=default_idx,
    )
    value_column = None if value_column == "(nenhuma)" else value_column

auto_register_feedback = False
if value_column:
    auto_register_feedback = st.checkbox(
        f"Registrar automaticamente como feedback ao concluir (usa '{value_column}' como valor real)",
        value=True,
        help=(
            "Fecha o ciclo de aprendizado contínuo: os pares (imóvel, preço real) ficam disponíveis "
            "pra API usar em retraining, sem precisar clicar de novo depois."
        ),
    )

st.success(f"{len(df)} linha(s) lida(s), colunas obrigatórias presentes.")
st.dataframe(df.head(10), width='stretch', hide_index=True)

# ---- coerção de tipo + validação leve no portal (a validação de regra de negócio é da API, P5) ----
coerced = df[REQUIRED_CSV_COLUMNS].copy()
type_errors: list[dict] = []
for col in REQUIRED_CSV_COLUMNS:
    numeric = pd.to_numeric(coerced[col], errors="coerce")
    bad_mask = numeric.isna() & coerced[col].notna()
    for idx in coerced.index[bad_mask]:
        type_errors.append({"linha_csv": int(idx) + 1, "coluna": col, "valor": coerced.at[idx, col]})
    if _FIELD_KIND[col] == "int":
        coerced[col] = numeric.round()
    else:
        coerced[col] = numeric
missing_mask = coerced.isna().any(axis=1)
for idx in coerced.index[missing_mask & ~coerced.index.isin([e["linha_csv"] - 1 for e in type_errors])]:
    type_errors.append({"linha_csv": int(idx) + 1, "coluna": "(vazio)", "valor": None})

if type_errors:
    st.error(f"{len(type_errors)} valor(es) inválido(s)/vazio(s) encontrados:")
    st.dataframe(pd.DataFrame(type_errors), width='stretch', hide_index=True)
    drop_invalid = st.checkbox("Ignorar linhas inválidas e continuar só com as válidas", value=False)
    if not drop_invalid:
        st.stop()
    bad_rows = {e["linha_csv"] - 1 for e in type_errors}
    valid_idx = [i for i in df.index if i not in bad_rows]
    df, coerced = df.loc[valid_idx], coerced.loc[valid_idx]
    if df.empty:
        st.warning("Nenhuma linha válida restante.")
        st.stop()

for col in REQUIRED_CSV_COLUMNS:
    if _FIELD_KIND[col] == "int":
        coerced[col] = coerced[col].astype(int)

df = df.reset_index(drop=True)
coerced = coerced.reset_index(drop=True)

st.caption(f"{len(coerced)} linha(s) válida(s) prontas para inferência.")

run_clicked = st.button("Rodar inferência em lote", type="primary")

# ---- chamada à API em chunks de até 500 (limite de POST /predictions/batch) -------------------
# Resultado guardado em session_state: um `st.dataframe` interativo abaixo (seleção de linha) dispara
# reruns por conta própria — se a tabela só existisse "atrás" do `if run_clicked`, cada rerun de
# seleção perderia tudo (botão não fica True de novo sozinho). Ver docs/09.../app/README para o padrão.
if run_clicked:
    BATCH_LIMIT = 500
    records = coerced.to_dict(orient="records")
    results: list[dict] = []
    progress = st.progress(0.0, text="Enviando para a API...")
    chunks = [records[i : i + BATCH_LIMIT] for i in range(0, len(records), BATCH_LIMIT)]

    run_failed = False
    for i, chunk in enumerate(chunks):
        try:
            response = client.predict_batch(chunk)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 422:
                detail = exc.response.json().get("detail", [])
                rows = []
                for err in detail:
                    loc = err.get("loc", [])
                    row_in_chunk = next((p for p in loc if isinstance(p, int)), None)
                    field = loc[-1] if loc else "?"
                    csv_row = (
                        df.index[i * BATCH_LIMIT + row_in_chunk] + 1 if row_in_chunk is not None else "?"
                    )
                    rows.append({"linha_csv": csv_row, "campo": field, "erro": err.get("msg", "")})
                st.error(f"A API rejeitou o lote {i + 1}/{len(chunks)} — {len(rows)} erro(s) de validação:")
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            else:
                st.error(f"Erro da API ({exc.response.status_code}) no lote {i + 1}/{len(chunks)}: {exc.response.text}")
            run_failed = True
            break
        except httpx.HTTPError as exc:
            st.error(f"Não foi possível falar com a API: {exc}")
            run_failed = True
            break
        results.extend(response["predictions"])
        progress.progress((i + 1) / len(chunks), text=f"Lote {i + 1}/{len(chunks)} concluído")

    progress.empty()
    if not run_failed:
        result_df = pd.DataFrame(results)
        out = pd.concat([result_df, df], axis=1)
        if value_column:
            actual = pd.to_numeric(out[value_column], errors="coerce")
            out["erro_abs_usd"] = (out["predicted_price"] - actual).abs()
            out["erro_pct"] = np.where(actual > 0, out["erro_abs_usd"] / actual * 100, np.nan)

        st.session_state["batch_records"] = records
        st.session_state["batch_out"] = out.to_dict(orient="records")
        st.session_state["batch_value_column"] = value_column
        st.session_state["batch_results_raw"] = results

        if value_column and auto_register_feedback:
            if not client.api_key:
                st.warning(
                    "Registro automático de feedback ligado, mas não foi possível registrar agora — "
                    "use o botão \"Registrar todos os valores reais como feedback\" mais abaixo pra "
                    "tentar de novo."
                )
            else:
                ok, failed = 0, []
                fb_progress = st.progress(0.0, text="Registrando valores reais como feedback...")
                rows_out = out.to_dict(orient="records")
                for i, row in enumerate(rows_out):
                    actual = row.get(value_column)
                    pid = row.get("prediction_id")
                    if pid and actual is not None and not (isinstance(actual, float) and np.isnan(actual)):
                        try:
                            client.record_feedback(pid, float(actual))
                            ok += 1
                        except httpx.HTTPError as exc:
                            failed.append({"prediction_id": pid, "erro": str(exc)})
                    fb_progress.progress((i + 1) / len(rows_out))
                fb_progress.empty()
                st.success(
                    f"{ok} valor(es) real(is) registrado(s) automaticamente como feedback."
                )
                if failed:
                    st.warning(f"{len(failed)} falha(s) no registro automático:")
                    st.dataframe(pd.DataFrame(failed), width='stretch', hide_index=True)

if "batch_out" not in st.session_state:
    st.stop()

out = pd.DataFrame(st.session_state["batch_out"])
records = st.session_state["batch_records"]
results = st.session_state["batch_results_raw"]
value_column = st.session_state["batch_value_column"]

st.success(f"{len(results)} predição(ões) concluída(s).")

# ---- resumo do lote -------------------------------------------------------------------------
s1, s2, s3 = st.columns(3)
s1.metric("Preço previsto médio", f"US$ {out['predicted_price'].mean():,.0f}")
s2.metric(
    "Faixa de preço previsto",
    f"US\\$ {out['predicted_price'].min():,.0f} – US\\$ {out['predicted_price'].max():,.0f}",
)
if value_column:
    s3.metric(
        "MAPE do lote", f"{out['erro_pct'].mean():.1f}%",
        help="Mean Absolute Percentage Error — média do erro percentual absoluto entre preço previsto "
             "e preço real, considerando todas as linhas do lote com valor real informado.",
    )
else:
    s3.metric("Coluna de valor real", "não informada")

# ---- tabela — preço previsto e demais resultados em destaque, clique na linha p/ detalhe -----
st.subheader("Resultados")
result_cols = ["predicted_price", "price_band", "confidence_score", "confidence_category"]
if value_column:
    result_cols += [value_column, "erro_abs_usd", "erro_pct"]
other_cols = [
    c for c in out.columns
    if c not in result_cols and c not in ("prediction_id", "model_version", "property_cluster")
]
display_cols = result_cols + other_cols

column_config = {
    "predicted_price": st.column_config.NumberColumn("Preço previsto", format="$ %.0f"),
    "price_band": CATEGORY_LABEL,
    "erro_abs_usd": st.column_config.NumberColumn("Erro (US$)", format="$ %.0f"),
    "erro_pct": st.column_config.NumberColumn("Erro (%)", format="%.1f%%"),
}
if value_column:
    column_config[value_column] = st.column_config.NumberColumn("Valor real", format="$ %.0f")

event = st.dataframe(
    out[display_cols],
    width='stretch',
    hide_index=True,
    column_config=column_config,
    on_select="rerun",
    selection_mode="single-row",
)
st.caption("Clique numa linha para ver o detalhe completo (mesmas informações do endpoint de inferência individual).")

selected_rows = list(event.selection.rows) if event and event.selection else []
if selected_rows:
    idx = selected_rows[0]
    _render_row_detail(idx + 1, records[idx], results[idx], client)

# ---- gráficos por categoria do imóvel ----------------------------------------------------------
st.divider()
st.subheader("Distribuição de preço por categoria do imóvel")
dist_chart = (
    alt.Chart(out)
    .mark_boxplot(size=40)
    .encode(
        x=alt.X("price_band:N", title=CATEGORY_LABEL),
        y=alt.Y("predicted_price:Q", title="Preço previsto (US$)"),
        color=alt.Color("price_band:N", legend=None),
    )
    .properties(title=alt.TitleParams("Preço previsto por categoria", fontSize=13), height=320)
)
st.altair_chart(dist_chart, width='stretch')

if value_column:
    st.subheader("Erro por categoria do imóvel")
    error_by_cat = out.dropna(subset=["erro_pct"]).groupby("price_band", as_index=False)["erro_pct"].mean()
    error_chart = (
        alt.Chart(error_by_cat)
        .mark_bar()
        .encode(
            x=alt.X("price_band:N", title=CATEGORY_LABEL),
            y=alt.Y("erro_pct:Q", title="Erro percentual médio (%)"),
            color=alt.Color("price_band:N", legend=None),
            tooltip=["price_band", "erro_pct"],
        )
        .properties(title=alt.TitleParams("MAPE por categoria", fontSize=13), height=320)
    )
    st.altair_chart(error_chart, width='stretch')

# ---- gráfico de acerto — previsto x real + distribuição do erro ------------------------------
if value_column:
    st.divider()
    st.subheader("O quanto o modelo acertou")

    acerto_threshold = st.slider(
        "Limiar de acerto (erro percentual máximo considerado \"Acerto\")",
        min_value=1, max_value=50, value=10, step=1, format="%d%%",
        help="Só classifica os pontos dos gráficos abaixo como Acerto/Erro — não afeta o MAPE do lote nem nenhuma métrica registrada.",
    )
    ERRO_COLORS = {"Acerto": "#2563eb", "Erro": "#e05252"}

    scored = out.dropna(subset=["erro_pct"]).copy()
    scored["classificação"] = np.where(scored["erro_pct"] <= acerto_threshold, "Acerto", "Erro")

    chart_data = scored[[value_column, "predicted_price", "erro_pct", "classificação", "price_band"]].copy()
    chart_data.columns = ["valor_real", "valor_previsto", "erro_pct", "classificação", CATEGORY_LABEL]
    if not chart_data.empty:
        lo = float(min(chart_data[["valor_real", "valor_previsto"]].min()))
        hi = float(max(chart_data[["valor_real", "valor_previsto"]].max()))
        scatter = (
            alt.Chart(chart_data)
            .mark_circle(size=70, opacity=0.65)
            .encode(
                x=alt.X("valor_real", title="Valor real (US$)", scale=alt.Scale(domain=[lo, hi])),
                y=alt.Y("valor_previsto", title="Valor previsto (US$)", scale=alt.Scale(domain=[lo, hi])),
                color=alt.Color(
                    "classificação:N", title=f"Erro ≤ {acerto_threshold}%",
                    scale=alt.Scale(domain=["Acerto", "Erro"], range=[ERRO_COLORS["Acerto"], ERRO_COLORS["Erro"]]),
                ),
                tooltip=["valor_real", "valor_previsto", "erro_pct", "classificação", CATEGORY_LABEL],
            )
        )
        reference = (
            alt.Chart(pd.DataFrame({"x": [lo, hi], "y": [lo, hi]}))
            .mark_line(color="#9ca3af", strokeDash=[6, 4])
            .encode(x="x", y="y")
        )
        st.altair_chart(
            (scatter + reference).properties(title=alt.TitleParams("Previsto vs. real", fontSize=13), height=400),
            width='stretch',
        )
        st.caption(
            "Linha tracejada = acerto perfeito. Acima da linha = modelo superestimou; abaixo = subestimou. "
            f"Cor: **azul** = erro ≤ {acerto_threshold}% (Acerto), **vermelho** = erro > {acerto_threshold}% (Erro), "
            "conforme o limiar escolhido acima."
        )

        n_acerto = int((chart_data["classificação"] == "Acerto").sum())
        n_erro = len(chart_data) - n_acerto
        m1, m2 = st.columns(2)
        m1.metric("Acerto (dentro do limiar)", f"{n_acerto} ({n_acerto / len(chart_data):.0%})")
        m2.metric("Erro (fora do limiar)", f"{n_erro} ({n_erro / len(chart_data):.0%})")

    view_mode = st.radio(
        "Distribuição do erro",
        ["Geral", "Segmentada por Erro/Acerto", f"Segmentada por Erro/Acerto e {CATEGORY_LABEL}"],
        horizontal=True,
    )

    if not chart_data.empty and view_mode == "Geral":
        hist = (
            alt.Chart(chart_data)
            .mark_bar(color=ERRO_COLORS["Acerto"])
            .encode(
                x=alt.X("erro_pct", bin=alt.Bin(maxbins=30), title="Erro percentual absoluto (%)"),
                y=alt.Y("count()", title="Nº de imóveis"),
            )
            .properties(title=alt.TitleParams("Distribuição do erro percentual absoluto", fontSize=13))
        )
        st.altair_chart(hist, width='stretch')

    elif not chart_data.empty and view_mode == "Segmentada por Erro/Acerto":
        hist_split = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X("erro_pct", bin=alt.Bin(maxbins=30), title="Erro percentual absoluto (%)"),
                y=alt.Y("count()", title="Nº de imóveis"),
                color=alt.Color(
                    "classificação:N", title=f"Erro ≤ {acerto_threshold}%",
                    scale=alt.Scale(domain=["Acerto", "Erro"], range=[ERRO_COLORS["Acerto"], ERRO_COLORS["Erro"]]),
                ),
                tooltip=["classificação", "count()"],
            )
            .properties(title=alt.TitleParams("Distribuição do erro — por Acerto/Erro", fontSize=13))
        )
        st.altair_chart(hist_split, width='stretch')

    elif not chart_data.empty:
        by_cat = chart_data.groupby([CATEGORY_LABEL, "classificação"], as_index=False).size()
        cat_chart = (
            alt.Chart(by_cat)
            .mark_bar()
            .encode(
                x=alt.X(f"{CATEGORY_LABEL}:N", title=CATEGORY_LABEL),
                y=alt.Y("size:Q", title="Nº de imóveis"),
                color=alt.Color(
                    "classificação:N", title=f"Erro ≤ {acerto_threshold}%",
                    scale=alt.Scale(domain=["Acerto", "Erro"], range=[ERRO_COLORS["Acerto"], ERRO_COLORS["Erro"]]),
                ),
                xOffset="classificação:N",
                tooltip=[CATEGORY_LABEL, "classificação", "size"],
            )
            .properties(title=alt.TitleParams(f"Acerto/Erro por {CATEGORY_LABEL}", fontSize=13), height=360)
        )
        st.altair_chart(cat_chart, width='stretch')

# ---- registro de feedback em lote ------------------------------------------------------------
if value_column:
    st.divider()
    st.subheader("Registrar valores reais em lote")
    st.caption(
        f"Usa a coluna **{value_column}** como valor real de cada linha. Útil se o registro "
        "automático (checkbox acima do botão de inferência) estava desligado, ou para tentar de novo "
        "as linhas que falharam."
    )
    if st.button("Registrar todos os valores reais como feedback"):
        rows = st.session_state.get("batch_out", [])
        vcol = st.session_state.get("batch_value_column")
        ok, failed = 0, []
        fb_progress = st.progress(0.0)
        for i, row in enumerate(rows):
            actual = row.get(vcol)
            pid = row.get("prediction_id")
            if pid and actual is not None and not (isinstance(actual, float) and np.isnan(actual)):
                try:
                    client.record_feedback(pid, float(actual))
                    ok += 1
                except httpx.HTTPError as exc:
                    failed.append({"prediction_id": pid, "erro": str(exc)})
            fb_progress.progress((i + 1) / len(rows))
        fb_progress.empty()
        st.success(f"{ok} valor(es) real(is) registrado(s) com sucesso.")
        if failed:
            st.warning(f"{len(failed)} falha(s):")
            st.dataframe(pd.DataFrame(failed), width='stretch', hide_index=True)
