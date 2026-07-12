# 🍷 Wine Quality Classification

> **POSTECH — Data Analytics · Tech Challenge · Fase 2**

Modelo de Machine Learning capaz de prever se um vinho tinto é de **alta qualidade** ou **baixa/média qualidade** com base em suas características físico-químicas — sem depender de avaliação sensorial humana.

---

## 📋 Índice

- [Sobre o projeto](#-sobre-o-projeto)
- [Problema e solução](#-problema-e-solução)
- [Dataset](#-dataset)
- [Resultados](#-resultados)
- [Estrutura do repositório](#-estrutura-do-repositório)
- [Como executar](#-como-executar)
- [Pipeline do projeto](#-pipeline-do-projeto)
- [Principais descobertas](#-principais-descobertas)
- [Tecnologias](#-tecnologias)
- [Equipe](#-equipe)

---

## 📌 Sobre o projeto

A avaliação da qualidade de vinhos é tradicionalmente feita por especialistas (enólogos) por meio de análise sensorial — um processo subjetivo, demorado e custoso. Este projeto demonstra que é possível **automatizar essa classificação** com alta confiabilidade usando apenas dados físico-químicos coletados durante a produção.

O modelo desenvolvido atinge **AUC-ROC de 0.918** e **F1-Score de 0.638**, sendo capaz de identificar corretamente 69% dos vinhos de alta qualidade antes mesmo de qualquer avaliação sensorial.

---

## 🎯 Problema e solução

**Problema:** dado um conjunto de medições físico-químicas de um vinho tinto, ele é de alta ou baixa/média qualidade?

**Variável alvo (transformação binária):**

| Nota original | Classificação | Valor |
|:---:|:---:|:---:|
| ≥ 7 | Alta qualidade | `1` |
| < 7 | Baixa/Média qualidade | `0` |

**Aplicação prática:** o modelo pode ser usado como **ferramenta de triagem** — classificando lotes automaticamente e direcionando apenas os candidatos a alta qualidade para avaliação sensorial completa pelo enólogo, otimizando tempo e recursos.

---

## 📊 Dataset

**Fonte:** [Wine Quality Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)

| Característica | Detalhe |
|---|---|
| Amostras | 1.143 vinhos tintos |
| Features | 11 variáveis físico-químicas |
| Variável alvo | `quality` (nota 3–8) → `binary_quality` (0/1) |
| Valores nulos | Nenhum |
| Desbalanceamento | 86,1% baixa/média · 13,9% alta qualidade |

### Variáveis disponíveis

| Variável | Descrição | Correlação com qualidade |
|---|---|:---:|
| `alcohol` | Teor alcoólico (%) | **+0.484** |
| `volatile acidity` | Acidez volátil — ácido acético (g/L) | **-0.407** |
| `sulphates` | Sulfatos — conservante (g/L) | +0.257 |
| `citric acid` | Ácido cítrico — frescor (g/L) | +0.241 |
| `total sulfur dioxide` | SO₂ total (mg/L) | -0.183 |
| `density` | Densidade (g/mL) | -0.175 |
| `chlorides` | Cloretos — sal (g/L) | -0.124 |
| `fixed acidity` | Acidez fixa (g/L) | +0.122 |
| `free sulfur dioxide` | SO₂ livre (mg/L) | -0.063 |
| `pH` | pH | -0.052 |
| `residual sugar` | Açúcar residual (g/L) | +0.022 |

---

## 🏆 Resultados

### Comparação entre modelos

| Modelo | F1-Score | Precision | Recall | AUC-ROC |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.473 | 0.361 | 0.688 | 0.856 |
| XGBoost | 0.618 | 0.583 | 0.656 | 0.912 |
| **Random Forest** ✅ | **0.638** | **0.595** | **0.688** | **0.918** |

**Modelo escolhido: Random Forest** — melhor F1-Score (0.638) e maior AUC-ROC (0.918).

### Importância das variáveis (Random Forest)

```
alcohol              ██████████████████████  23.6%
sulphates            ███████████████         15.1%
citric acid          █████████████           13.4%
volatile acidity     ███████████             11.7%
total sulfur dioxide ██████                   5.9%
density              ██████                   5.9%
fixed acidity        █████                    5.7%
residual sugar       █████                    5.1%
pH                   █████                    4.9%
free sulfur dioxide  ████                     4.4%
chlorides            ████                     4.3%
```

### Desempenho no conjunto de teste (229 vinhos)

```
                  Previsto: Baixa    Previsto: Alta
Real: Baixa            181               16
Real: Alta              10               22
```

- ✅ 181 de 197 vinhos comuns classificados corretamente (92%)
- ✅ 22 de 32 vinhos de alta qualidade identificados (69%)
- ⚠️ 10 vinhos bons classificados como comuns (31% de perda)

---

## 📁 Estrutura do repositório

```
wine-quality-classification/
│
├── data/
│   └── WineQT.csv                        # Dataset original
│
├── notebooks/
│   └── Tech_Challenge_02_final.ipynb     # Notebook completo (EDA + modelos)
│
├── src/
│   ├── preprocessing.py                  # Funções de pré-processamento
│   └── evaluate.py                       # Funções de avaliação dos modelos
│
├── results/
│   ├── 01_balanceamento_classes.png
│   ├── 02_distribuicao_variaveis.png
│   ├── 03_faixas_categoricas.png
│   ├── 04_correlacoes.png
│   ├── 05_boxplots_por_classe.png
│   ├── 06_outliers.png
│   ├── 07_matrizes_confusao.png
│   ├── 08_curvas_roc.png
│   ├── 09_comparacao_metricas.png
│   ├── 10_feature_importance.png
│   └── 11_medias_por_classe.png
│
├── apresentacao/
│   └── storytelling_eda.pdf              # Apresentação executiva
│
├── requirements.txt                      # Dependências do projeto
└── README.md                             # Este arquivo
```

---

## 🚀 Como executar

### Pré-requisitos

- Python 3.9 ou superior
- pip

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/wine-quality-classification.git
cd wine-quality-classification
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Execute o notebook

```bash
jupyter notebook notebooks/Tech_Challenge_02_final.ipynb
```

> **Atenção:** ajuste o caminho do CSV na célula de carga de dados se necessário:
> ```python
> df_raw = pd.read_csv('../data/WineQT.csv')  # caminho relativo ao notebook
> ```

### Execução no Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Faça upload do arquivo `WineQT.csv` para o Colab
2. Ajuste o caminho na célula de carga:
   ```python
   df_raw = pd.read_csv('/content/WineQT.csv')
   ```
3. Execute todas as células em ordem (`Runtime > Run all`)

---

## 🔄 Pipeline do projeto

```
WineQT.csv
    │
    ▼
┌─────────────────────────────┐
│  1. Carga e inspeção        │  Verificação de nulos · tipos · estatísticas
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  2. EDA                     │  Distribuições · Correlações · Outliers
│                             │  Balanceamento · Faixas categóricas
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  3. Pré-processamento       │  Train/test split estratificado (80/20)
│                             │  Winsorização (percentis 1–99, só treino)
│                             │  RobustScaler (fit só no treino)
│                             │  SMOTE (só no treino)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  4. Modelagem               │  Logistic Regression (baseline)
│                             │  Random Forest ✅ (vencedor)
│                             │  XGBoost
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  5. Avaliação               │  F1-Score · Precision · Recall · AUC-ROC
│                             │  Matrizes de confusão · Curvas ROC
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  6. Interpretação           │  Feature importance · Médias por classe
│                             │  Implicações para produção
└─────────────────────────────┘
```

---

## 🔍 Principais descobertas

### O que determina um vinho de alta qualidade?

**Fatores que aumentam a qualidade:**

- 🟢 **Teor alcoólico** (+12,3% em vinhos bons): fermentação mais completa indica processo bem conduzido e uvas mais maduras. Vinhos bons têm média de 11,5% vs 10,3% dos comuns.
- 🟢 **Sulfatos** (+15,9%): atua como conservante natural, protegendo contra oxidação e bactérias durante o envelhecimento.
- 🟢 **Ácido cítrico** (+57,4%): contribui para o frescor e equilíbrio sensorial do vinho.

**Fatores que reduzem a qualidade:**

- 🔴 **Acidez volátil** (-28,6% em vinhos bons): presença de ácido acético (vinagre) indica contaminação bacteriana ou oxidação durante a produção.
- 🔴 **SO₂ total** (-22,6%): excesso indica maior intervenção química, geralmente associada a uvas de qualidade inferior.

### Recomendações para a produção

1. **Monitorar o teor alcoólico** ao final da fermentação — lotes abaixo de 10,5% merecem revisão
2. **Medir acidez volátil** de forma rotineira — acima de 0,6 g/L indica problema bacteriano
3. **Manter sulfatos entre 0,65 e 0,85 g/L** como faixa-alvo de qualidade
4. **Usar o modelo como triagem** — direcionar candidatos a alta qualidade para avaliação sensorial completa

### Limitações

- Modelo treinado exclusivamente com **vinho tinto** — não aplicável a brancos ou rosés sem retreinamento
- **Recall de 69%** na classe alta: 31% dos vinhos bons ainda são classificados incorretamente
- Variáveis sensoriais (aroma, cor, textura) não estão presentes no dataset

---

## 🛠 Tecnologias

| Biblioteca | Versão | Uso |
|---|:---:|---|
| Python | 3.9+ | Linguagem base |
| pandas | 3.0.2 | Manipulação de dados |
| numpy | 2.4.4 | Operações matemáticas |
| matplotlib | 3.10.8 | Visualizações |
| seaborn | 0.13.2 | Gráficos estatísticos |
| scikit-learn | 1.8.0 | Modelos e pré-processamento |
| xgboost | 3.3.0 | Modelo XGBoost |
| imbalanced-learn | 0.14.2 | SMOTE para balanceamento |
| jupyter | - | Execução do notebook |

---

## 👥 Equipe

Desenvolvido como parte do **Tech Challenge — Fase 2** do curso de Pós-Graduação em Data Analytics da POSTECH.

| Nome | GitHub |
|---|---|
| Seu Nome | [@seu-usuario](https://github.com/seu-usuario) |

---

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos no âmbito do POSTECH Tech Challenge.

---

<div align="center">

**POSTECH — Data Analytics · Fase 2 · 2025**

</div>
