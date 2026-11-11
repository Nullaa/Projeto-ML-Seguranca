import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

st.set_page_config(page_title="Risco criminal por UF", page_icon="📍", layout="wide")

UFS = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
       "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"]
CLASSES = ["Baixo", "Médio", "Alto"]
CORES = {"Baixo": "#2e9e5b", "Médio": "#e0a21b", "Alto": "#d64545"}
AQUI = Path(__file__).resolve().parent


def limpar_infinitos(X):
    """Função usada dentro do pipeline salvo (FunctionTransformer) no notebook.

    O joblib guarda apenas a referência `__main__.limpar_infinitos`; por isso ela precisa
    existir aqui, no escopo do app, para o modelo carregar.
    """
    if hasattr(X, "replace"):
        return X.replace([np.inf, -np.inf], np.nan)
    return np.where(np.isinf(X), np.nan, X)


def slug(texto):
    t = unicodedata.normalize("NFKD", str(texto).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_")


# ------------------------------------------------------------------ carregamento
@st.cache_resource(show_spinner=False)
def carregar_modelos():
    """Carrega os .joblib de modelos/ ou models/ (ao lado do app ou na pasta acima)."""
    modelos, erros = {}, []
    for base in dict.fromkeys([AQUI, AQUI.parent]):
        for pasta in ["modelos", "models"]:
            for arq in sorted((base / pasta).glob("*.joblib")):
                try:
                    m = joblib.load(arq)
                    if not isinstance(m, dict) or "pipeline" not in m or "indicador_alvo" not in m:
                        raise ValueError("formato de metadados inesperado")
                    modelos.setdefault(m["indicador_alvo"], m)
                except Exception as erro:  # versão do scikit-learn diferente, arquivo corrompido...
                    erros.append(f"{arq.name}: {erro}")
    return modelos, erros


@st.cache_data(show_spinner=False)
def ler_painel(caminho):
    return pd.read_csv(caminho)


def achar_painel(alvo):
    nome = f"painel_uf_ano_{slug(alvo)}.csv"
    for pasta in [AQUI / "resultados", AQUI / "dados" / "processados", AQUI / "dados",
                  AQUI.parent / "resultados", AQUI]:
        if (pasta / nome).exists():
            return pasta / nome
    return None


def preparar_painel(df):
    """Marca as linhas utilizáveis: com histórico (lag1 e lag2) para o modelo ter sentido."""
    df = df.copy()
    df["uf"] = df["uf"].astype(str).str.strip().str.upper()
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ano"])
    df["ano"] = df["ano"].astype(int)
    tem = lambda c: df[c].notna() if c in df.columns else pd.Series(True, index=df.index)
    df["historico_ok"] = tem("taxa_alvo_100k_lag1") & tem("taxa_alvo_100k_lag2")
    df["alvo_reportado"] = tem("taxa_alvo_100k")
    return df


# ------------------------------------------------------------------ metadados (novo e antigo)
def anos_treino_modelo(m):
    return m.get("anos_treino_final") or m.get("anos_treino") or []


def anos_teste_modelo(m):
    return m.get("anos_teste_avaliacao") or m.get("anos_teste") or []


def limites_modelo(m):
    """(tercil_1, tercil_2) ou None (modo relativo: tercis calculados dentro de cada ano)."""
    lim = m.get("limites_risco_modelo_final")
    if isinstance(lim, dict) and "tercil_1" in lim:
        return float(lim["tercil_1"]), float(lim["tercil_2"])
    if "tercil_1" in m and "tercil_2" in m:
        return float(m["tercil_1"]), float(m["tercil_2"])
    return None


def classe_por_taxa(taxa, lim):
    if lim is None or pd.isna(taxa):
        return None
    return "Baixo" if taxa <= lim[0] * (1 + 1e-9) + 1e-9 else ("Médio" if taxa <= lim[1] * (1 + 1e-9) + 1e-9 else "Alto")


def classe_ano_anterior(df, lim):
    """Baseline de persistência: classe que a UF teria com a taxa do ano anterior."""
    if "taxa_alvo_100k_lag1" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="object")
    if lim is not None:
        return df["taxa_alvo_100k_lag1"].map(lambda t: classe_por_taxa(t, lim))
    # modo relativo: posição entre as UFs presentes
    r = df["taxa_alvo_100k_lag1"].rank(method="first")
    if r.notna().sum() < 3:
        return pd.Series(np.nan, index=df.index, dtype="object")
    return pd.qcut(r, 3, labels=CLASSES).astype("object")


def rotulos_atributos(m):
    alvo = m["indicador_alvo"]
    por_slug = {f"taxa_{slug(i)}_100k": i for i in m.get("indicadores_utilizados", [])}
    r = {
        "populacao": "População da UF (habitantes)",
        "taxa_alvo_100k_lag1": f"Taxa de {alvo} no ano anterior (por 100 mil hab.)",
        "variacao_pct_alvo": "Variação % da taxa do alvo (ano anterior vs. dois anos antes)",
        "taxa_total_outros_100k": "Soma das taxas dos outros indicadores, no ano (por 100 mil)",
        "taxa_total_outros_100k_lag1": "Soma das taxas dos outros indicadores, ano anterior (por 100 mil)",
        # modelos antigos
        "total_geral": "Total de ocorrências no ano (exceto o alvo)",
        "taxa_total_100k": "Taxa total de ocorrências, exceto o alvo (por 100 mil)",
    }
    for a in m["atributos"]:
        if a in r or a == "uf":
            continue
        base, lag = (a[:-5], True) if a.endswith("_lag1") else (a, False)
        if base in por_slug:
            r[a] = f"{por_slug[base]} — {'ano anterior' if lag else 'no ano'} (por 100 mil)"
        else:
            r[a] = a
    return r


ATRIBUTOS_PRINCIPAIS = ["populacao", "taxa_alvo_100k_lag1", "variacao_pct_alvo",
                        "taxa_total_outros_100k", "taxa_total_outros_100k_lag1",
                        "total_geral", "taxa_total_100k"]


def prever(df, m):
    atributos = m["atributos"]
    numericos = [a for a in atributos if a != "uf"]
    X = df.reindex(columns=atributos).copy()
    if "uf" in X.columns:
        X["uf"] = X["uf"].astype(str).str.strip().str.upper()
    for c in numericos:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X[numericos] = X[numericos].replace([np.inf, -np.inf], np.nan)
    pipe = m["pipeline"]
    pred = pipe.predict(X)
    proba = pd.DataFrame(pipe.predict_proba(X), columns=pipe.classes_, index=df.index)
    return pred, proba.reindex(columns=CLASSES)


# ------------------------------------------------------------------ componentes visuais
def cartao_risco(classe, titulo):
    cor = CORES[classe]
    st.markdown(
        f"<div style='padding:16px;border-radius:10px;background:{cor}22;border:2px solid {cor}'>"
        f"<span style='font-size:14px'>{titulo}</span><br>"
        f"<span style='font-size:34px;font-weight:700;color:{cor}'>{classe}</span></div>",
        unsafe_allow_html=True,
    )


def barras_probabilidade(proba_linha):
    cols = st.columns(3)
    for col, c in zip(cols, CLASSES):
        p = float(proba_linha[c])
        col.metric(c, f"{p:.0%}")
        col.progress(min(max(p, 0.0), 1.0))


# ------------------------------------------------------------------ início
modelos, erros_carga = carregar_modelos()

if not modelos:
    st.error("Nenhum modelo encontrado em `modelos/*.joblib` (ou `models/`). "
             "Adicione os arquivos exportados na Sprint 2.")
    for e in erros_carga:
        st.warning(f"Falha ao carregar {e}")
    st.stop()

with st.sidebar:
    alvo = st.selectbox("Indicador-alvo", sorted(modelos))
meta = modelos[alvo]
atributos = meta["atributos"]
numericos = [a for a in atributos if a != "uf"]
lim = limites_modelo(meta)
rotulos = rotulos_atributos(meta)
cenario = meta.get("cenario", {})
modo_risco = cenario.get("modo_risco", "absoluto" if lim else "relativo")
tipo_atributos = cenario.get("atributos", "mesmo_ano")

# Painel (opcional): do repositório ou enviado pelo usuário
caminho_painel = achar_painel(alvo)
with st.sidebar:
    st.header("Sobre o modelo")
    st.write(f"**Indicador-alvo:** {alvo}")
    st.write(f"**Algoritmo:** {meta.get('modelo', '—')}")
    anos_tr = anos_treino_modelo(meta)
    if anos_tr:
        st.write(f"**Anos do treino final:** {min(anos_tr)}–{max(anos_tr)}")
    if lim:
        st.markdown(
            f"**Classes de risco** (taxa por 100 mil hab.):\n"
            f"- Baixo: ≤ {lim[0]:.2f}\n- Médio: ≤ {lim[1]:.2f}\n- Alto: > {lim[1]:.2f}"
        )
    else:
        st.markdown("**Classes de risco:** relativas — tercis entre as UFs, dentro de cada ano.")
    versao_modelo = str(meta.get("sklearn_version", ""))
    if versao_modelo and versao_modelo.split(".")[:2] != sklearn.__version__.split(".")[:2]:
        st.warning(f"Modelo treinado com scikit-learn {versao_modelo}; este ambiente usa "
                   f"{sklearn.__version__}. Se der erro, fixe a mesma versão no requirements.txt.")
    arq_painel = st.file_uploader("Painel UF×ano (opcional)", type="csv",
                                  help=f"resultados/painel_uf_ano_{slug(alvo)}.csv, exportado na Sprint 2.")
    st.caption("Protótipo acadêmico (UFFS). A classe de risco vem de tercis da série histórica; "
               "não é critério oficial e não estabelece causalidade. Dados: Sinesp VDE e IBGE/SIDRA.")

painel = None
try:
    if arq_painel is not None:
        painel = pd.read_csv(arq_painel)
    elif caminho_painel is not None:
        painel = ler_painel(str(caminho_painel))
    if painel is not None:
        if not {"ano", "uf"} <= set(painel.columns):
            raise ValueError("o painel precisa ter as colunas 'ano' e 'uf'")
        painel = preparar_painel(painel)
except Exception as erro:
    st.sidebar.error(f"Painel não pôde ser usado: {erro}")
    painel = None

st.title("📍 Risco criminal por UF")
if tipo_atributos == "so_historico":
    st.write(f"Previsão do nível de risco de **{alvo}** de uma UF em um ano, usando apenas "
             "informações do **ano anterior**.")
else:
    st.write(f"Estimativa do nível de risco de **{alvo}** de uma UF em um ano, a partir do "
             "histórico da UF e dos **outros indicadores do próprio ano**. É uma estimativa "
             "contemporânea, não uma previsão do futuro.")
for e in erros_carga:
    st.sidebar.warning(f"Modelo ignorado — {e}")

aba_pan, aba_sim, aba_lote, aba_sobre = st.tabs(
    ["Panorama do ano", "Simulador", "Lote (CSV)", "Sobre o modelo"]
)

# ------------------------------------------------------------------ 1. Panorama
with aba_pan:
    if painel is None:
        st.info("Para ver o panorama de todas as UFs, coloque o painel exportado na Sprint 2 em "
                f"`resultados/painel_uf_ano_{slug(alvo)}.csv` ou envie-o pela barra lateral.")
    else:
        anos_ok = sorted(painel.loc[painel["historico_ok"], "ano"].unique().tolist())
        if not anos_ok:
            st.warning("O painel não tem anos com histórico suficiente (lag1 e lag2).")
        else:
            ano = st.selectbox("Ano", anos_ok, index=len(anos_ok) - 1, key="pan_ano")
            d = painel[(painel["ano"] == ano) & painel["historico_ok"]].copy()
            pred, proba = prever(d, meta)
            d["Risco estimado"] = pred
            d["Classe do ano anterior"] = classe_ano_anterior(d, lim)
            if lim:
                d["Classe observada"] = d["taxa_alvo_100k"].map(lambda t: classe_por_taxa(t, lim))
            tab = pd.concat([d[["uf"]], proba.round(2).add_prefix("P(") .add_suffix(")")], axis=1)
            tab["Taxa observada"] = d["taxa_alvo_100k"].round(2)
            if lim:
                tab["Classe observada"] = d["Classe observada"]
            tab["Risco estimado"] = d["Risco estimado"]
            tab["Classe do ano anterior"] = d["Classe do ano anterior"]
            if lim:
                tab.loc[~d["alvo_reportado"], "Classe observada"] = "não reportado"
            tab = tab.sort_values("Taxa observada", ascending=False, na_position="last").rename(columns={"uf": "UF"})

            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader(f"UFs por risco estimado — {ano}")
                st.bar_chart(pd.Series(pred).value_counts().reindex(CLASSES).fillna(0))
            with c2:
                st.subheader("Detalhe por UF")
                st.dataframe(tab, hide_index=True)

            n_nr = int((~d["alvo_reportado"]).sum())
            if n_nr:
                st.warning(f"{n_nr} UF(s) não reportaram {alvo} em {ano}: " +
                           ", ".join(sorted(d.loc[~d["alvo_reportado"], "uf"])) +
                           ". Elas ficam sem taxa observada.")
            st.caption("Atenção: o modelo final foi treinado com todos os anos do painel, então a "
                       "comparação com a classe observada **não** mede o desempenho (isso está na aba "
                       "'Sobre o modelo'). 'Classe do ano anterior' é a referência simples de repetir "
                       "a classe do ano anterior.")

# ------------------------------------------------------------------ 2. Simulador
with aba_sim:
    fontes = ["Partir de uma UF/ano do painel", "Preencher manualmente"] if painel is not None \
        else ["Preencher manualmente"]
    fonte = st.radio("Ponto de partida", fontes, horizontal=True)

    base_linha, chave = None, "manual"
    valores = {}
    if fonte.startswith("Partir"):
        c1, c2 = st.columns(2)
        ufs_painel = sorted(painel["uf"].unique().tolist())
        uf_sel = c1.selectbox("UF", ufs_painel, index=ufs_painel.index("SC") if "SC" in ufs_painel else 0,
                              key="sim_uf")
        anos_uf = sorted(painel.loc[(painel["uf"] == uf_sel) & painel["historico_ok"], "ano"].unique().tolist())
        if not anos_uf:
            st.warning("Esta UF não tem anos com histórico suficiente no painel.")
        else:
            ano_sel = c2.selectbox("Ano", anos_uf, index=len(anos_uf) - 1, key="sim_ano")
            base_linha = painel[(painel["uf"] == uf_sel) & (painel["ano"] == ano_sel)].iloc[0]
            chave = f"{uf_sel}|{ano_sel}"
            valores["uf"] = uf_sel
        st.caption("Os campos abaixo vêm do painel. Altere qualquer valor para simular outro cenário.")
    else:
        st.caption("Campos em branco são preenchidos com a mediana do treino (mesmo tratamento do modelo). "
                   "Os valores são **taxas por 100 mil habitantes**.")
        if "uf" in atributos:
            valores["uf"] = st.selectbox("UF", UFS, index=UFS.index("SC"), key="sim_uf_manual")

    if fonte.startswith("Partir") and base_linha is None:
        pass
    else:
        def valor_base(a):
            if base_linha is None or a not in base_linha.index or pd.isna(base_linha[a]):
                return None
            return float(base_linha[a])

        principais = [a for a in ATRIBUTOS_PRINCIPAIS if a in numericos]
        outros = [a for a in numericos if a not in principais]
        cols = st.columns(2)
        for i, a in enumerate(principais):
            valores[a] = cols[i % 2].number_input(
                rotulos[a], value=valor_base(a), step=1.0, format="%.2f" if a != "populacao" else "%.0f",
                placeholder="vazio = mediana", key=f"{alvo}|{chave}|{a}")
        if outros:
            with st.expander(f"Taxas dos outros indicadores ({len(outros)}) — opcional"):
                cols2 = st.columns(2)
                for i, a in enumerate(outros):
                    valores[a] = cols2[i % 2].number_input(
                        rotulos[a], value=valor_base(a), step=0.1, format="%.2f",
                        placeholder="vazio", key=f"{alvo}|{chave}|o|{a}")

        with st.expander("Converter contagem em taxa por 100 mil"):
            k1, k2 = st.columns(2)
            casos = k1.number_input("Nº de casos", value=None, min_value=0.0, step=1.0, key=f"conv_c|{chave}")
            pop = k2.number_input("População", value=None, min_value=1.0, step=1000.0, key=f"conv_p|{chave}")
            if casos is not None and pop:
                st.write(f"Taxa = **{casos / pop * 100_000:.2f}** por 100 mil habitantes")

        df_in = pd.DataFrame([valores])
        pred, proba = prever(df_in, meta)
        classe = pred[0]
        n_vazios = sum(1 for a in numericos if valores.get(a) is None)

        st.divider()
        titulo = f"Risco estimado" + (f" para {valores['uf']}" if "uf" in valores else "")
        r1, r2 = st.columns([1, 2])
        with r1:
            cartao_risco(classe, titulo)
            ref = classe_ano_anterior(
                pd.DataFrame([{"taxa_alvo_100k_lag1": valores.get("taxa_alvo_100k_lag1")}]), lim).iloc[0] \
                if lim else None
            if isinstance(ref, str):
                st.caption(f"Só pelo ano anterior, a classe seria **{ref}**.")
        with r2:
            st.subheader("Probabilidade por classe")
            barras_probabilidade(proba.iloc[0])
        st.caption(f"{n_vazios} de {len(numericos)} atributos numéricos em branco (imputados pela mediana do treino).")
        if n_vazios > len(numericos) / 2:
            st.warning("Mais da metade dos atributos está em branco; o resultado depende sobretudo das medianas.")

        if base_linha is not None and "taxa_alvo_100k" in painel.columns:
            hist = painel[painel["uf"] == valores["uf"]].sort_values("ano").set_index("ano")[["taxa_alvo_100k"]]
            hist = hist.rename(columns={"taxa_alvo_100k": f"Taxa de {alvo}"})
            if lim:
                hist["Limite Baixo/Médio"] = lim[0]
                hist["Limite Médio/Alto"] = lim[1]
            st.subheader(f"Histórico de {valores['uf']}")
            st.line_chart(hist)

# ------------------------------------------------------------------ 3. Lote
with aba_lote:
    st.write("Envie um CSV com uma linha por UF/ano. Colunas ausentes são tratadas como vazias "
             "(imputadas pela mediana do treino). Valores em **taxa por 100 mil habitantes**.")
    if painel is not None:
        ultimo = painel[painel["historico_ok"]]["ano"].max()
        exemplo = painel[(painel["ano"] == ultimo) & painel["historico_ok"]].reindex(columns=["uf", "ano"] + [a for a in atributos if a != "uf"]).head(3)
    else:
        exemplo = pd.DataFrame(columns=["uf", "ano"] + [a for a in atributos if a != "uf"])
    st.download_button("Baixar modelo de CSV", exemplo.to_csv(index=False).encode("utf-8-sig"),
                       "modelo_entrada.csv", "text/csv")
    arq = st.file_uploader("CSV de entrada", type="csv", key="lote")
    if arq is not None:
        try:
            entrada = pd.read_csv(arq)
        except Exception as erro:
            st.error(f"Não foi possível ler o CSV: {erro}")
            entrada = None
        if entrada is not None:
            if "uf" in atributos and "uf" not in entrada.columns:
                st.error("O CSV precisa ter a coluna `uf`.")
            else:
                faltam = [a for a in atributos if a not in entrada.columns]
                if faltam:
                    st.warning("Colunas ausentes (serão imputadas): " + ", ".join(faltam))
                if len(faltam) == len(atributos):
                    st.error("Nenhuma coluna de atributos reconhecida. Use o modelo de CSV.")
                else:
                    pred, proba = prever(entrada, meta)
                    saida = entrada.copy()
                    saida["risco_previsto"] = pred
                    saida = pd.concat([saida, proba.round(3).add_prefix("prob_")], axis=1)
                    st.dataframe(saida)
                    st.bar_chart(saida["risco_previsto"].value_counts().reindex(CLASSES).fillna(0))
                    st.download_button("Baixar resultados", saida.to_csv(index=False).encode("utf-8-sig"),
                                       "predicoes_risco.csv", "text/csv")

# ------------------------------------------------------------------ 4. Sobre o modelo
with aba_sobre:
    st.subheader("O que este modelo responde")
    if tipo_atributos == "so_historico":
        st.write("Usa apenas informações **até o ano anterior** (taxa do alvo, variação e taxas dos outros "
                 "indicadores no ano anterior). É uma previsão de fato.")
    else:
        st.write("Usa o histórico da UF e os **outros indicadores do próprio ano**. Os indicadores medidos "
                 "juntos podem ter causas em comum (por exemplo, a qualidade do registro de cada UF), então "
                 "isto é uma estimativa contemporânea e **não** uma previsão do futuro.")
    if modo_risco == "relativo":
        st.write("As classes são **relativas**: tercis entre as UFs dentro de cada ano. 'Alto' quer dizer "
                 "entre as UFs mais violentas daquele ano, não que o crime aumentou.")
    else:
        st.write("As classes usam **limites fixos** de taxa por 100 mil (tercis da série histórica). Como as "
                 "taxas mudam ao longo do tempo, os limites podem ficar defasados em anos recentes.")

    st.subheader("Desempenho (validação walk-forward)")
    f1_m = meta.get("desempenho_walk_forward_f1_macro")
    f1_p = meta.get("desempenho_persistencia_f1_macro")
    if f1_m is not None and f1_p is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric(f"F1-macro — {meta.get('modelo', 'modelo')}", f"{f1_m:.3f}")
        c2.metric("F1-macro — repetir classe do ano anterior", f"{f1_p:.3f}")
        c3.metric("Diferença", f"{f1_m - f1_p:+.3f}")
        if f1_m - f1_p < 0.05:
            st.warning("O ganho sobre simplesmente repetir a classe do ano anterior é pequeno e está "
                       "dentro da variação entre rodadas. Interprete o resultado como uma estimativa "
                       "aproximada, não como um instrumento de decisão.")
        st.caption("Média do F1-macro em rodadas anuais: treina com os anos anteriores e testa no ano seguinte. "
                   "O modelo exportado foi reajustado com todos os anos, então não deve ser avaliado nos mesmos dados.")
    else:
        st.info("Este arquivo de modelo não traz as métricas de desempenho (formato anterior à v5).")

    st.subheader("Detalhes")
    anos_te = anos_teste_modelo(meta)
    st.write(f"- Anos usados no treino final: {', '.join(map(str, anos_tr)) or '—'}")
    st.write(f"- Anos de teste da avaliação: {', '.join(map(str, anos_te)) or '—'}")
    st.write(f"- Atributos ({len(atributos)}): " + ", ".join(rotulos.get(a, a) for a in atributos))
    st.write(f"- Versão do pipeline: {meta.get('versao_pipeline', '—')}")

    st.subheader("Limitações")
    st.markdown(
        "- Poucos anos e poucas UFs: as métricas têm incerteza alta.\n"
        "- UFs que não reportaram o indicador em determinado ano foram retiradas da modelagem; "
        "o modelo não foi treinado com esses casos.\n"
        "- As classes vêm de tercis, sem critério oficial de risco em segurança pública.\n"
        "- A UF entra como atributo, então o modelo aprende o nível típico de cada estado.\n"
        "- É um modelo de base, sem atributos espaciais (vizinhança) nem ajuste de hiperparâmetros.\n"
        "- Não estabelece relação causal."
    )
