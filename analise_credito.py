# --------------------------------------------------------------------------- #
# 1. INSTALAÇÃO E IMPORTAÇÃO DE BIBLIOTECAS
# --------------------------------------------------------------------------- #
# !pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
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
try:
    df = pd.read_csv('credit_risk_dataset.csv')
    print("Dataset carregado com sucesso.")
    print("Formato do dataset:", df.shape)
except FileNotFoundError:
    print("Erro: Arquivo 'credit_risk_dataset.csv' não encontrado.")
    exit()

# --------------------------------------------------------------------------- #
# 3. ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# --------------------------------------------------------------------------- #
print("\n--- INICIANDO ANÁLISE EXPLORATÓRIA ---")
print("\nPrimeiras 5 linhas do dataset:")
print(df.head())

print("\nInformações do Dataset:")
df.info()

print("\nDistribuição da variável alvo (loan_status):")
print(df['loan_status'].value_counts(normalize=True))

print("\nValores ausentes por coluna:")
print(df.isnull().sum())

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

X = df.drop('loan_status', axis=1)
y = df['loan_status']

numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

print(f"\nFeatures Numéricas: {numerical_features}")
print(f"Features Categóricas: {categorical_features}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nDados divididos em treino ({X_train.shape[0]} amostras) e teste ({X_test.shape[0]} amostras).")
print("\n--- FIM DO PRÉ-PROCESSAMENTO ---")

# --------------------------------------------------------------------------- #
# 5. TREINAMENTO E AVALIAÇÃO DOS MODELOS
# --------------------------------------------------------------------------- #
print("\n--- TREINAMENTO E AVALIAÇÃO ---")

# Regressão Logística (Baseline)
print("\n--- Regressão Logística ---")
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, class_weight='balanced'))
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

print("\nResultados da Regressão Logística:")
print(classification_report(y_test, y_pred_lr))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob_lr):.4f}")

# Random Forest
print("\n--- Random Forest ---")
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

print("\nResultados do Random Forest:")
print(classification_report(y_test, y_pred_rf))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob_rf):.4f}")

# --------------------------------------------------------------------------- #
# 6. TABELA DE MÉTRICAS CONSOLIDADA
# --------------------------------------------------------------------------- #
from sklearn.metrics import precision_recall_fscore_support

metrics_lr = precision_recall_fscore_support(y_test, y_pred_lr, average=None, labels=[1])[0:3]
metrics_rf = precision_recall_fscore_support(y_test, y_pred_rf, average=None, labels=[1])[0:3]

results = pd.DataFrame({
    "Métrica": ["Precisão (Classe 1)", "Recall (Classe 1)", "F1-Score (Classe 1)", "AUC-ROC"],
    "Regressão Logística": [metrics_lr[0][0], metrics_lr[1][0], metrics_lr[2][0], roc_auc_score(y_test, y_prob_lr)],
    "Random Forest": [metrics_rf[0][0], metrics_rf[1][0], metrics_rf[2][0], roc_auc_score(y_test, y_prob_rf)]
})

print("\nTabela Consolidada de Métricas:\n")
print(results.round(4))

# --------------------------------------------------------------------------- #
# 7. MATRIZES DE CONFUSÃO E CURVA ROC
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Matriz de Confusão - Regressão Logística')

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Matriz de Confusão - Random Forest')
plt.show()

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

# --------------------------------------------------------------------------- #
# 8. VALIDAÇÃO CRUZADA PARA RANDOM FOREST
# --------------------------------------------------------------------------- #
print("\n--- Validação Cruzada (Random Forest) ---")
cv_scores_rf = cross_val_score(rf_pipeline, X_train, y_train, cv=5, scoring='roc_auc')
print(f"AUC-ROC Médio na Validação Cruzada: {cv_scores_rf.mean():.4f}")

print("\nPipeline completo concluído com sucesso.")
