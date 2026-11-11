# 🗺️ Análise Espacial da Criminalidade no Brasil

Projeto acadêmico desenvolvido na disciplina de **Planejamento e Gestão de Projetos** (UFFS), unindo Ciência de Dados e Machine Learning para pensar a segurança pública no Brasil.

> ✅ **Status: projeto finalizado.** Dados organizados, análise exploratória concluída, modelos treinados e protótipo interativo publicado.

🔗 **App online:** (https://projeto-ml-seguranca-fqsspcunigosdonyykxemj.streamlit.app/)

---

## 📌 Sobre o projeto

Quem já tentou comparar dados de criminalidade entre estados brasileiros sabe o quanto isso pode ser trabalhoso: as informações estão espalhadas em bases diferentes, com formatos que nem sempre conversam entre si.

Este projeto nasceu dessa dificuldade. Ele organiza dados públicos de segurança, analisa padrões entre as Unidades da Federação (UFs) e oferece um **protótipo interativo** que estima o nível de risco de uma UF em um ano, com base em Machine Learning.

---

## ❓ Pergunta-problema

> Como dados públicos e técnicas de análise espacial podem ser utilizados para identificar e visualizar padrões de criminalidade entre diferentes regiões do Brasil?

---

## 🎯 Objetivo

Organizar, analisar e apresentar dados públicos de criminalidade no Brasil, construindo uma base analítica e um modelo de Machine Learning que classifique o risco por UF em **Baixo, Médio ou Alto**.

---


## 🤖 O modelo

- **Algoritmo:** Random Forest, com pipeline de pré-processamento (imputação pela mediana e padronização).
- **Indicadores-alvo disponíveis:** *Arma de Fogo Apreendida* e *Tentativa de feminicídio*.
- **Treino:** anos de 2016 a 2021.
- **Classes de risco:** Baixo, Médio e Alto, definidas por **tercis da série histórica** da taxa por 100 mil habitantes.

| Indicador | Baixo | Médio | Alto |
|---|---|---|---|
| Arma de Fogo Apreendida | ≤ 37,14 | ≤ 62,33 | > 62,33 |
| Tentativa de feminicídio | ≤ 0 | ≤ 1,48 | > 1,48 |

**O que o modelo responde:** uma estimativa do nível de risco de uma UF em um ano, a partir do histórico da própria UF e dos outros indicadores do mesmo ano. É uma estimativa contemporânea, **não uma previsão do futuro**.

---

## 🖥️ O app

O protótipo em Streamlit possui três abas:

- **Panorama do ano:** risco estimado para todas as UFs em um ano (usa o painel UF × ano exportado na Sprint 2).
- **Simulador:** calcula o risco de uma UF/ano, partindo do painel ou de campos preenchidos manualmente. Campos em branco usam a mediana do treino.
- **Lote (CSV):** previsão para várias linhas de uma vez, com download dos resultados.

### Como rodar localmente

```bash
git clone https://github.com/<seu-usuario>/Projeto-ML-Seguranca.git
cd Projeto-ML-Seguranca
pip install -r requirements.txt
streamlit run app/app.py
```

> ⚠️ Os modelos foram treinados com **scikit-learn 1.6.1**. Use essa versão (já fixada no `requirements.txt`) para que os arquivos `.joblib` carreguem corretamente.

---



## 🗃️ Fonte dos dados

- **Sinesp VDE** — indicadores de segurança pública;
- **IBGE / SIDRA** — população, usada para calcular taxas por 100 mil habitantes.

---

## ⚠️ Limitações

- Poucos anos e poucas UFs: as métricas de desempenho têm incerteza alta.
- UFs que não reportaram um indicador em determinado ano ficaram fora da modelagem.
- As classes vêm de tercis da própria série histórica; **não são um critério oficial** de risco em segurança pública.
- A UF entra como atributo, então o modelo aprende o nível típico de cada estado.
- É um modelo de base, sem atributos espaciais (vizinhança) nem ajuste de hiperparâmetros.
- Os resultados **não estabelecem relação causal** e não devem ser usados, sozinhos, como instrumento de decisão.

---

## 👥 Público-alvo

- 🏛️ **Gestores públicos**, que precisam de dados confiáveis para tomar decisões;
- 🔬 **Pesquisadores**, interessados em explorar os dados sob diferentes perspectivas;
- 🎓 **Estudantes**, que queiram aprender ou se aprofundar no tema;
- 👮 **Profissionais da área de segurança pública**;
- 🙋 **Cidadãos**, curiosos em entender melhor os indicadores da própria região.

---

## 📄 Licença

Distribuído sob a licença presente no arquivo [`LICENSE`](LICENSE).
