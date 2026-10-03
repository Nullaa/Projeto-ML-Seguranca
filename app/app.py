"""MVP – Análise Espacial da Criminalidade no Brasil (Sprint 3).

App Streamlit que consome o pipeline treinado na Sprint 2 (modelos/*.joblib)
e estima o nível de risco (Baixo / Médio / Alto) de uma UF em um ano.
Os campos do formulário são gerados a partir de `metadados["atributos"]`,
então o app continua funcionando se o modelo for retreinado.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Risco criminal por UF", page_icon="📍", layout="wide")


# O pipeline foi salvo no Colab referenciando esta função em __main__.
# Ela precisa existir aqui, com o mesmo nome, para o joblib.load funcionar.
def limpar_infinitos(X):
    X = np.asarray(X, dtype=float)
    return np.where(np.isinf(X), np.nan, X)


sys.modules["__main__"].limpar_infinitos = limpar_infinitos

UFS = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
       "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"]
CLASSES = ["Baixo", "Médio", "Alto"]
CORES = {"Baixo": "#2e9e5b", "Médio": "#e0a21b", "Alto": "#d64545"}
PRINCIPAIS = {
    "populacao": "População da UF (habitantes)",
    "taxa_alvo_100k_lag1": "Taxa do indicador-alvo no ano anterior (por 100 mil hab.)",
    "variacao_pct_alvo": "Variação % da taxa do indicador-alvo",
    "total_geral": "Total geral de ocorrências no ano",
    "taxa_total_100k": "Taxa total de ocorrências (por 100 mil hab.)",
}


@st.cache_resource
def carregar_modelos():
    """Carrega todos os .joblib de models/ (ou modelos/), na raiz do repo ou ao lado do app."""
    aqui = Path(__file__).resolve().parent
    modelos = {}
    for base in [aqui, aqui.parent]:
        for nome_pasta in ["models", "modelos"]:
            for arq in sorted((base / nome_pasta).glob("*.joblib")):
                m = joblib.load(arq)
                modelos.setdefault(m["indicador_alvo"], m)
    return modelos


todos = carregar_modelos()
if not todos:
    st.error("Nenhum modelo encontrado em `models/*.joblib`. Adicione os arquivos exportados na Sprint 2.")
    st.stop()

meta = todos[st.sidebar.selectbox("Indicador-alvo", sorted(todos))]
pipe = meta["pipeline"]
atributos = meta["atributos"]
alvo = meta["indicador_alvo"]
numericos = [a for a in atributos if a != "uf"]


def prever(df):
    X = df.reindex(columns=atributos).copy()
    for c in numericos:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    pred = pipe.predict(X)
    proba = pd.DataFrame(pipe.predict_proba(X), columns=pipe.classes_).reindex(columns=CLASSES)
    return pred, proba


# ---------------- Barra lateral ----------------
with st.sidebar:
    st.header("Sobre o modelo")
    st.write(f"**Indicador-alvo:** {alvo}")
    st.write(f"**Algoritmo:** {meta['modelo']}")
    st.write(f"**Treino:** {min(meta['anos_treino'])}–{max(meta['anos_treino'])}")
    st.write(f"**Teste:** {', '.join(map(str, meta['anos_teste']))}")
    st.markdown(
        f"**Classes de risco** (taxa por 100 mil hab.):\n"
        f"- Baixo: ≤ {meta['tercil_1']:.2f}\n"
        f"- Médio: ≤ {meta['tercil_2']:.2f}\n"
        f"- Alto: > {meta['tercil_2']:.2f}"
    )
    st.caption("Protótipo acadêmico (UFFS). O risco é relativo (tercis da série histórica), "
               "não é um critério oficial e não estabelece causalidade. Dados: Sinesp VDE e IBGE/SIDRA.")

st.title("📍 Risco criminal por UF")
st.write(f"Estimativa do nível de risco para **{alvo}** a partir dos atributos da UF no ano.")
if "uf" not in atributos:
    st.info("Este modelo não usa a UF como atributo: o risco depende apenas dos valores informados.")

aba1, aba2 = st.tabs(["Predição individual", "Predição em lote (CSV)"])

# ---------------- Predição individual ----------------
with aba1:
    st.caption("Campos deixados em branco são preenchidos com a mediana do treino (mesmo tratamento do modelo).")
    with st.form("form_individual"):
        valores = {}
        if "uf" in atributos:
            valores["uf"] = st.selectbox("UF", UFS, index=UFS.index("SC"))
        cols = st.columns(2)
        principais = [a for a in PRINCIPAIS if a in atributos]
        for i, a in enumerate(principais):
            valores[a] = cols[i % 2].number_input(PRINCIPAIS[a], value=None, step=1.0,
                                                  placeholder="vazio = mediana", key=f"{alvo}|{a}")
        outros = [a for a in numericos if a not in PRINCIPAIS]
        with st.expander(f"Contagens de outros indicadores no ano ({len(outros)}) – opcional"):
            cols2 = st.columns(3)
            for i, a in enumerate(outros):
                valores[a] = cols2[i % 3].number_input(a, value=None, step=1.0,
                                                       placeholder="vazio", key=f"{alvo}|o|{a}")
        enviar = st.form_submit_button("Estimar risco", type="primary")

    if enviar:
        df = pd.DataFrame([valores])
        pred, proba = prever(df)
        classe = pred[0]
        n_vazios = sum(1 for a in numericos if valores.get(a) is None)
        st.markdown(
            f"<div style='padding:16px;border-radius:10px;background:{CORES[classe]}22;"
            f"border:2px solid {CORES[classe]}'><span style='font-size:14px'>Risco estimado{' para ' + valores['uf'] if 'uf' in valores else ''}</span><br>"
            f"<span style='font-size:34px;font-weight:700;color:{CORES[classe]}'>{classe}</span></div>",
            unsafe_allow_html=True,
        )
        st.subheader("Probabilidade por classe")
        st.bar_chart(proba.iloc[0].rename("probabilidade"), horizontal=True)
        st.caption(f"{n_vazios} de {len(numericos)} atributos numéricos ficaram em branco e foram imputados pela mediana.")

# ---------------- Predição em lote ----------------
with aba2:
    st.write("Envie um CSV com uma linha por UF/ano. Colunas ausentes serão tratadas como vazias.")
    modelo_csv = pd.DataFrame(columns=atributos).to_csv(index=False).encode("utf-8-sig")
    st.download_button("Baixar modelo de CSV", modelo_csv, "modelo_entrada.csv", "text/csv")
    arq = st.file_uploader("CSV de entrada", type="csv")
    if arq is not None:
        entrada = pd.read_csv(arq)
        if "uf" in atributos and "uf" not in entrada.columns:
            st.error("O CSV precisa ter a coluna `uf`.")
        else:
            pred, proba = prever(entrada)
            saida = entrada.copy()
            saida["risco_previsto"] = pred
            saida = pd.concat([saida, proba.add_prefix("prob_")], axis=1)
            st.dataframe(saida, use_container_width=True)
            st.bar_chart(saida["risco_previsto"].value_counts().reindex(CLASSES).fillna(0))
            st.download_button("Baixar resultados", saida.to_csv(index=False).encode("utf-8-sig"),
                               "predicoes_risco.csv", "text/csv")
