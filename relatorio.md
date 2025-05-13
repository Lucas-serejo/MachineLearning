# Simulação de Aprendizado de Máquina Federado com Python

## 🎯 Objetivo

Simular o processo de aprendizado federado utilizando múltiplos conjuntos de dados locais e um servidor central para agregação dos modelos. A proposta visa compreender os conceitos de privacidade, descentralização e atualização global do modelo.

---

## 👥 Participantes

| Nome Completo         | Matrícula     |
|-----------------------|---------------|
|Lucas José Silva Serejo|202202714356   |
|Beatriz Turi Pinto de Araujo|202203795211|
|Pedro Henrique Rossetto Costa|202108581259|
|Lucas Fernandes Mosqueira|202203369016|

---

## 📁 Dataset Utilizado

**Fashion-MNIST**: Dataset de imagens de roupas (28x28 pixels, tons de cinza).  
- Total de amostras: 70.000  
- Conjunto de treino: 60.000  
- Conjunto de teste: 10.000  
- Classes: 10 tipos de roupas  
- Tarefa: Classificação multiclasse  

---

## 🔀 Divisão dos Dados

- O conjunto de treino foi dividido igualmente entre **5 clientes simulados**, com 12.000 amostras cada.
- A divisão foi feita de forma **balanceada e aleatória** (IID).
- Cada cliente treinou seu modelo **localmente**, sem acesso aos dados dos demais.

---

## 🧠 Modelo Utilizado

- Tipo: **MLP (Multi-Layer Perceptron)**
- Estrutura:
  - Camada densa (128 neurônios, ReLU)
  - Camada de saída (10 neurônios, Softmax)
- Otimizador: **Adam**
- Perda: `sparse_categorical_crossentropy`

---

## 🛠️ Treinamento Local

Cada cliente realizou seu próprio treinamento local com:
- **1 época por rodada federada**
- **Batch size:** 32
- **Inicialização com pesos do modelo global**
- **Retorno dos pesos locais ao servidor**

---

## 🔗 Agregação no Servidor (FedAvg)

A agregação foi realizada via **média simples (FedAvg)** dos pesos recebidos de todos os clientes.  
Após a agregação, o modelo global foi atualizado e redistribuído para os clientes na próxima rodada.

---

## 🔁 Rodadas Federadas

- Número de rodadas: **5**
- A cada rodada:
  1. Clientes recebem pesos do modelo global
  2. Treinam localmente
  3. Enviam pesos atualizados
  4. Servidor agrega e atualiza o modelo

---

## 🧪 Avaliação e Resultados

### 📈 Modelo Federado
- **Acurácia final após 5 rodadas:** `85.90%`  
- **Perda final:** `0.3962`

### 🏁 Modelo Centralizado
- **Acurácia:** `86.99%`  
- **Perda:** `0.3709`

### 📊 Comparativo

| Modelo               | Acurácia | Perda |
|----------------------|----------|--------|
| Federado (5 rounds)  | 85.90%  | 0.3962 |
| Centralizado         | 86.99%   | 0.3709 |

> _Observação: o modelo centralizado apresentou melhor desempenho, como esperado, pois treina com o conjunto completo e homogêneo. Ainda assim, o modelo federado teve desempenho competitivo sem precisar centralizar os dados._

---

## 📌 Conclusões

- O aprendizado federado se mostrou eficaz em alcançar boa acurácia sem centralizar os dados.
- O processo pode ser expandido com dados **não-IID**, mais clientes e arquiteturas mais complexas.
- Em cenários reais, a privacidade e a comunicação são tão importantes quanto a performance do modelo.

"""

# Salvar o relatório em arquivo .md
caminho_arquivo = "/mnt/data/relatorio_federated_learning.md"
with open(caminho_arquivo, "w", encoding="utf-8") as f:
    f.write(relatorio_md)

caminho_arquivo
