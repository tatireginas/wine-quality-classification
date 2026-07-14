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

O notebook possui **71 células** distribuídas em **8 seções**, cobrindo desde a carga via API do Kaggle até 4 experimentos de otimização de modelos. O melhor resultado alcançado foi **F1-Score de 0.679** e **Precision de 0.857** com XGBoost + threshold otimizado.

---

## 🎯 Problema e solução

**Problema:** dado um conjunto de medições físico-químicas de um vinho tinto, ele é de alta ou baixa/média qualidade?

**Variável alvo (transformação binária):**

| Nota original | Classificação | Valor |
|:---:|:---:|:---:|
| ≥ 7 | Alta qualidade | `1` |
| < 7 | Baixa/Média qualidade | `0` |

**Aplicação prática:** o modelo pode ser usado como **ferramenta de triagem** — classificando lotes automaticamente e direcionando apenas os candidatos a alta qualidade para avaliação sensorial completa pelo enólogo.

---

## 📊 Dataset

**Fonte:** [Wine Quality Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) — carregado diretamente via `kagglehub` (com fallback local automático).

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

### Modelos base — Seção 6

| Modelo | F1-Score | Precision | Recall | AUC-ROC |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.473 | 0.361 | 0.688 | 0.856 |
| XGBoost | 0.618 | 0.583 | 0.656 | 0.912 |
| **Random Forest** | **0.638** | **0.595** | **0.688** | **0.918** |

### Após otimização — Seção 8

| Estratégia | F1 | Precision | Recall | AUC | Δ F1 |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline RF (threshold 0.50) | 0.638 | 0.595 | 0.688 | 0.918 | — |
| RF + Threshold ótimo (0.560) | 0.646 | 0.636 | 0.656 | 0.918 | +0.008 |
| RF Tuned + Threshold | 0.646 | 0.636 | 0.656 | 0.918 | +0.008 |
| **XGB Tuned + Threshold (0.942)** ✅ | **0.679** | **0.857** | **0.563** | 0.900 | **+0.041** |
| Feature Engineering + RF | 0.615 | 0.606 | 0.625 | 0.903 | -0.023 |
| SMOTETomek + RF | 0.618 | — | — | 0.907 | -0.020 |

**Para alta Precision** (não indicar vinho ruim como bom): XGB Tuned + Threshold 0.942 → Precision = **0.857**

**Para máximo Recall** (não perder vinhos bons): RF + Threshold 0.560 → Recall = **0.656**

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

### Desempenho no conjunto de teste — Random Forest base (229 vinhos)

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
│   └── WineQT.csv                             # Dataset (fallback local)
│
├── notebooks/
│   └── Tech_Challenge_02_VERSAO_FINAL.ipynb   # Notebook completo — 71 células, 8 seções
│
├── src/
│   ├── preprocessing.py                       # Funções de pré-processamento
│   └── evaluate.py                            # Funções de avaliação dos modelos
│
├── results/
│   ├── 01_balanceamento_classes.png           # Contagem por categoria + notas originais
│   ├── 02_distribuicao_variaveis.png          # Histogramas com KDE por classe
│   ├── 03_faixas_categoricas.png              # Faixas de álcool e pH
│   ├── 03b_faixas_complementares.png          # Faixas de acidez fixa, açúcar e ác. cítrico
│   ├── 04_correlacoes.png                     # Heatmap + correlação com qualidade
│   ├── 05_boxplots_por_classe.png             # Boxplots das 11 variáveis por classe
│   ├── 06_outliers.png                        # Outliers por variável (método IQR)
│   ├── 07_matrizes_confusao.png               # Matrizes de confusão dos 3 modelos
│   ├── 08_curvas_roc.png                      # Curvas ROC sobrepostas
│   ├── 09_comparacao_metricas.png             # Comparação de métricas por modelo
│   ├── 10_feature_importance.png              # Importância das variáveis (RF)
│   ├── 11_medias_por_classe.png               # Diferença % + médias normalizadas
│   ├── 12_threshold_tuning.png                # Curvas Precision-Recall-F1 por threshold
│   └── 13_comparacao_otimizacao.png           # Todos os experimentos de otimização
│
├── apresentacao/
│   └── wine_storytelling_eda.pptx             # Apresentação executiva (12 slides)
│
├── requirements.txt                           # Dependências do projeto
└── README.md                                  # Este arquivo
```

---

## 🚀 Como executar

### Pré-requisitos

- Python 3.9 ou superior
- pip
- Conta no Kaggle (opcional — há fallback automático para arquivo local)

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/wine-quality-classification.git
cd wine-quality-classification
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as credenciais do Kaggle (opcional)

```bash
# Opção A: arquivo de credenciais
cp kaggle.json ~/.kaggle/kaggle.json

# Opção B: variáveis de ambiente
export KAGGLE_USERNAME=seu_usuario
export KAGGLE_KEY=sua_chave_api
```

> Se as credenciais não estiverem disponíveis, o notebook carrega o arquivo local `data/WineQT.csv` automaticamente — nenhuma alteração de código necessária.

### 5. Crie a pasta de resultados e execute o notebook

```bash
mkdir -p results
jupyter notebook notebooks/Tech_Challenge_02_VERSAO_FINAL.ipynb
```

Execute todas as células em sequência: **Kernel > Restart & Run All**

### Execução no Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

```python
# Instalar dependências extras
!pip install kagglehub imbalanced-learn xgboost --quiet

# Criar pasta de resultados
!mkdir -p results
```

---

## 🔄 Pipeline do projeto

```
Kaggle API (kagglehub) ──→ fallback: WineQT.csv local
         │
         ▼
 1. Carga e inspeção      nulos · tipos · describe()
         │
         ▼
 2. EDA (14 gráficos)     distribuições · correlações · outliers
                          balanceamento · faixas categóricas (5 plots)
         │
         ▼
 3. Feature Engineering   5 features testadas por CV 5-fold
    (Seção 3.8 — EDA)     decisão: manter 11 originais
         │
         ▼
 4. Pré-processamento     split 80/20 estratificado
                          Winsorização (percentis 1–99, só treino)
                          RobustScaler (fit só no treino)
                          SMOTE (só no treino)
         │
         ▼
 5. Modelagem             Logistic Regression (baseline)
                          Random Forest ✅ (melhor AUC)
                          XGBoost
         │
         ▼
 6. Avaliação             F1 · Precision · Recall · AUC-ROC
                          matrizes de confusão · curvas ROC
         │
         ▼
 7. Interpretação         feature importance · médias por classe
                          implicações para produção
         │
         ▼
 8. Otimização            threshold tuning (RF: 0.560 | XGB: 0.942)
                          RandomizedSearchCV (25 RF + 20 XGB)
                          SMOTETomek · ADASYN
                          comparação consolidada
         │
         ▼
Melhor resultado: XGB Tuned + Threshold 0.942
F1 = 0.679  ·  Precision = 0.857  ·  AUC = 0.900
```

---

## 🔍 Principais descobertas

### O que determina um vinho de alta qualidade?

**Fatores que aumentam a qualidade:**

- 🟢 **Teor alcoólico** (+12,3%): fermentação mais completa indica processo bem conduzido. Vinhos bons: média 11,5% vs 10,3% dos comuns. **88,7% dos vinhos de alta qualidade têm álcool > 10%.**
- 🟢 **Ácido cítrico** (+57,4%): maior diferença relativa entre classes — frescor e equilíbrio sensorial.
- 🟢 **Sulfatos** (+15,9%): conservante natural contra oxidação e bactérias.

**Fatores que reduzem a qualidade:**

- 🔴 **Acidez volátil** (–28,6%): presença de ácido acético (vinagre) indica contaminação bacteriana. Vinhos bons: 0,40 g/L vs 0,55 g/L dos comuns.
- 🔴 **SO₂ total** (–22,6%): excesso indica maior intervenção química.

### O que a otimização revelou

- **Threshold tuning** é a estratégia de maior custo-benefício: uma linha de código, ganho de +0.008 (RF) a +0.062 (XGB) em F1
- **Tuning de hiperparâmetros** confirmou que o RF padrão já estava otimizado — parâmetros convergidos foram praticamente idênticos ao baseline
- **Feature Engineering** não trouxe ganho para modelos de árvore — confirmado empiricamente na Seção 3.8 e revalidado na Seção 8.3
- **SMOTETomek e ADASYN** ficaram abaixo do SMOTE padrão para este dataset

### Recomendações para a produção

1. **Monitorar o teor alcoólico** ao final da fermentação — lotes abaixo de 10,5% merecem revisão
2. **Medir acidez volátil** de forma rotineira — acima de 0,6 g/L indica problema bacteriano
3. **Manter sulfatos entre 0,65 e 0,85 g/L** como faixa-alvo de qualidade
4. **Usar o modelo como triagem** — XGB Tuned para máxima Precision ou RF+Threshold para máximo Recall

### Limitações

- Treinado exclusivamente com **vinho tinto** — não aplicável a brancos ou rosés sem retreinamento
- **Recall de 56–69%** dependendo do modelo: parte dos vinhos bons ainda são classificados incorretamente
- Variáveis sensoriais (aroma, cor, textura) não estão no dataset

---

## 🛠 Tecnologias

| Biblioteca | Versão | Uso |
|---|:---:|---|
| Python | 3.9+ | Linguagem base |
| pandas | 3.0.2 | Manipulação de dados |
| numpy | 2.4.4 | Operações matemáticas |
| matplotlib | 3.10.8 | Visualizações (paleta wine #6D1A36) |
| seaborn | 0.13.2 | Gráficos estatísticos |
| scikit-learn | 1.8.0 | Modelos, pré-processamento, RandomizedSearchCV |
| xgboost | 3.3.0 | Modelo XGBoost Tuned (melhor F1) |
| imbalanced-learn | 0.14.2 | SMOTE, SMOTETomek e ADASYN |
| kagglehub | latest | Carga direta do dataset via API Kaggle |
| jupyter | latest | Execução do notebook |
| ipykernel | latest | Kernel Python para Jupyter |

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
