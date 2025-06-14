# --------------------------------------------------------------------------- #
# 1. INSTALAÇÃO E IMPORTAÇÃO DE BIBLIOTECAS
# --------------------------------------------------------------------------- #
# Apenas para garantir que todas as bibliotecas necessárias estão instaladas.
# Descomente a linha abaixo se for executar pela primeira vez.
# !pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer

print("Bibliotecas importadas com sucesso.")

# --------------------------------------------------------------------------- #
# 2. CARREGAMENTO DOS DADOS
# --------------------------------------------------------------------------- #
# Carregando o dataset "Credit Risk Dataset" do Kaggle.
# Para executar este código, baixe o arquivo 'credit_risk_dataset.csv'
# do Kaggle e coloque-o no mesmo diretório que este script.
try:
    df = pd.read_csv('credit_risk_dataset.csv')
    print("Dataset carregado com sucesso.")
    print("Formato do dataset:", df.shape)
except FileNotFoundError:
    print("Erro: Arquivo 'credit_risk_dataset.csv' não encontrado.")
    print("Por favor, baixe o arquivo do Kaggle (laotse/credit-risk-dataset) e coloque-o na mesma pasta.")
    exit()

# --------------------------------------------------------------------------- #
# 3. ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# --------------------------------------------------------------------------- #
print("\n--- INICIANDO ANÁLISE EXPLORATÓRIA (EDA) ---")

# Visualizando as primeiras linhas
print("\nPrimeiras 5 linhas do dataset:")
print(df.head())

# Informações gerais sobre as colunas e tipos de dados
print("\nInformações do Dataset:")
df.info()

# Verificando a distribuição da variável alvo 'loan_status'
print("\nDistribuição da variável alvo (loan_status):")
print(df['loan_status'].value_counts(normalize=True))
# Observação: Existe um desbalanceamento. A classe '0' (não inadimplente) é
# muito mais frequente (aprox. 78%) que a '1' (inadimplente, aprox. 22%).
# Isso reforça a necessidade de usar métricas além da acurácia.

# Lidando com valores ausentes (missing values)
print("\nValores ausentes por coluna:")
print(df.isnull().sum())
# Observação: 'person_emp_length' e 'loan_int_rate' possuem valores ausentes
# que precisarão ser tratados no pré-processamento.

# Visualização da distribuição de algumas variáveis numéricas
plt.figure(figsize=(14, 6))
sns.histplot(data=df, x='person_income', hue='loan_status', kde=True, log_scale=True)
plt.title('Distribuição de Renda por Status do Empréstimo')
plt.show()

plt.figure(figsize=(14, 6))
sns.histplot(data=df, x='loan_amnt', hue='loan_status', kde=True)
plt.title('Distribuição do Valor do Empréstimo por Status')
plt.show()

print("\n--- FIM DA ANÁLISE EXPLORATÓRIA ---")

# --------------------------------------------------------------------------- #
# 4. PRÉ-PROCESSAMENTO E PREPARAÇÃO DOS DADOS
# --------------------------------------------------------------------------- #
print("\n--- INICIANDO PRÉ-PROCESSAMENTO ---")

# Separando features (X) e variável alvo (y)
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Identificando colunas numéricas e categóricas
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

print(f"\nFeatures Numéricas: {numerical_features}")
print(f"Features Categóricas: {categorical_features}")

# Criando pipelines de pré-processamento para cada tipo de feature
# Para features numéricas: preencher valores ausentes com a mediana e padronizar
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Para features categóricas: preencher valores ausentes com a moda e aplicar One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinando os pipelines em um único pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Separando dados em treino e teste (essencial para evitar data leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nDados divididos em treino ({X_train.shape[0]} amostras) e teste ({X_test.shape[0]} amostras).")
print("\n--- FIM DO PRÉ-PROCESSAMENTO ---")

# --------------------------------------------------------------------------- #
# 5. TREINAMENTO E AVALIAÇÃO DOS MODELOS
# --------------------------------------------------------------------------- #
print("\n--- INICIANDO TREINAMENTO E AVALIAÇÃO DOS MODELOS ---")

# --- Modelo 1: Regressão Logística (Baseline) ---
print("\n--- Treinando Modelo de Regressão Logística ---")
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(random_state=42, class_weight='balanced'))])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

print("\nResultados da Regressão Logística:")
print(classification_report(y_test, y_pred_lr))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob_lr):.4f}")

# --- Modelo 2: Random Forest ---
print("\n--- Treinando Modelo Random Forest ---")
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100))])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

print("\nResultados do Random Forest:")
print(classification_report(y_test, y_pred_rf))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob_rf):.4f}")

# --------------------------------------------------------------------------- #
# 6. VISUALIZAÇÃO E COMPARAÇÃO FINAL
# --------------------------------------------------------------------------- #
print("\n--- COMPARAÇÃO FINAL DOS MODELOS ---")

# Matrizes de Confusão
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Matriz de Confusão - Regressão Logística')
axes[0].set_xlabel('Previsto')
axes[0].set_ylabel('Real')

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Matriz de Confusão - Random Forest')
axes[1].set_xlabel('Previsto')
axes[1].set_ylabel('Real')
plt.show()

# Curvas ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(10, 8))
plt.plot(fpr_lr, tpr_lr, label=f'Regressão Logística (AUC = {roc_auc_score(y_test, y_prob_lr):.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Classificador Aleatório')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC Comparativa')
plt.legend()
plt.grid()
plt.show()

print("\nAnálise concluída.")