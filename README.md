# Deep Learning vs. Machine Learning Clássico na Predição de Evasão Estudantil

Trabalho individual da disciplina **Deep Learning** — Mestrado em Ciência de Dados e Inteligência Artificial (MCDIA), IDP.

**Modalidade 1:** reproduzir um estudo publicado que usou ML clássico e tentar superá-lo com um modelo de Deep Learning, no mesmo dataset.

## Resumo

Comparamos um modelo de Deep Learning (MLP com *entity embeddings*) contra baselines clássicos de *gradient boosting* (XGBoost, LightGBM) e Random Forest na predição de evasão estudantil, usando o dataset *Predict Students' Dropout and Academic Success* (UCI #697, 4.424 alunos, 34 variáveis).

O artigo-base — Villar & de Andrade (2024) — reporta F1 de 0,86–0,88. Mostramos que esses números **não se replicam sob um protocolo de avaliação sem vazamento de dados** (split estratificado *antes* de qualquer balanceamento). Sob avaliação honesta:

| Modelo | F1 Dropout | F1 Enrolled | F1 Graduate | Macro-F1 | AUC |
|---|---|---|---|---|---|
| Random Forest | 0,82 | 0,47 | 0,88 | 0,722 | 0,910 |
| **XGBoost** (melhor clássico) | 0,81 | 0,56 | 0,87 | **0,747** | 0,910 |
| LightGBM | 0,80 | 0,53 | 0,87 | 0,735 | 0,904 |
| **Deep Learning (Emb-MLP ×5)** | 0,79 | 0,54 | 0,84 | **0,721** | 0,893 |

**Três achados:**
1. O DL fica *competitivo* mas **não supera** o boosting nas métricas agregadas — coerente com a literatura de dados tabulares (Grinsztajn et al., 2022).
2. **Contribuição real:** o teto de desempenho honesto é ~0,75 (macro-F1) para *todos* os métodos; os 0,86–0,88 publicados eram inflados por vazamento (SMOTE antes do split).
3. O DL **agrega valor operacional** no alerta precoce binário (Dropout vs. resto): AUC 0,923, com limiar ajustável para priorizar *recall*.

## Estrutura do repositório

```
.
├── Evasao_DeepLearning_vs_ML.ipynb   # notebook executado (ponta a ponta)
├── dados_evasao.csv                  # cópia local do dataset
├── requirements.txt                  # dependências
├── README.md
├── figs/                             # figuras geradas pelo notebook
├── Relatorio_Evasao_DeepLearning.pdf # relatório (13 seções)
└── apresentacao_evasao.pdf           # slides
```

## Dados

Dataset **UCI #697 — Predict Students' Dropout and Academic Success** (Realinho et al., 2021, licença CC BY 4.0). Instituto Politécnico de Portalegre (Portugal), matrículas de 2008 a 2019. 4.424 alunos, 34 variáveis preditoras, alvo com 3 classes (Graduate / Dropout / Enrolled), sem valores ausentes.

O notebook carrega os dados diretamente de uma URL pública, em uma linha:

```python
import pandas as pd
url = "https://raw.githubusercontent.com/hamzaezzine/Predict-students-dropout-and-academic-success-using-machine-learning-algorithms/main/dataset.csv"
df = pd.read_csv(url, sep=";")
```

## Como executar

```bash
pip install -r requirements.txt
jupyter notebook Evasao_DeepLearning_vs_ML.ipynb
# ou, para reexecutar tudo de ponta a ponta:
jupyter nbconvert --to notebook --execute Evasao_DeepLearning_vs_ML.ipynb
```

O notebook é reprodutível (sementes fixadas: NumPy e PyTorch = 42; *ensemble* de 5 sementes [42, 7, 123, 2024, 99]).

## Referências principais

- **Artigo-base:** Villar, A. & de Andrade, C. R. V. (2024). Supervised machine learning algorithms for predicting student dropout and academic success: a comparative study. *Discover Artificial Intelligence*, 4(1), 2. https://doi.org/10.1007/s44163-023-00079-z
- **Dataset:** Realinho, V., Machado, J., Baptista, L. & Martins, M. V. (2022). Predicting Student Dropout and Academic Success. *Data*, 7(11), 146.
- **Técnica:** Guo, C. & Berkhahn, F. (2016). Entity Embeddings of Categorical Variables. arXiv:1604.06737.

A lista completa (10 referências) está no relatório.

---

*Autor(a): [seu nome] · MCDIA/IDP · 2026*
