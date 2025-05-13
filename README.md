# 🤝 Simulação de Aprendizado de Máquina Federado com Python

Este projeto tem como objetivo simular um sistema de **Aprendizado Federado (Federated Learning)** utilizando Python, onde múltiplos clientes treinam modelos localmente e um servidor central realiza a agregação dos pesos para formar um modelo global.

---

## 👥 Participantes

| Nome Completo         | Matrícula     |
|-----------------------|---------------|
|Lucas José Silva Serejo|202202714356   |
|Beatriz Turi Pinto de Araujo|202203795211|
|Pedro Henrique Rossetto Costa|202108581259|
|Lucas Fernandes Mosqueira|202203369016|


## 🎯 Objetivo

- Simular o processo de aprendizado federado com múltiplos datasets locais.
- Entender os conceitos de privacidade, descentralização e agregação global de modelos.
- Comparar o desempenho de um modelo federado com um modelo centralizado.

---

## 🧩 Descrição do Projeto

- O dataset é dividido entre 5 clientes simulados.
- Cada cliente treina localmente um modelo com seus próprios dados.
- Após o treinamento, os pesos dos modelos locais são enviados para um "servidor".
- O servidor executa a **média ponderada (FedAvg)** para criar o modelo global.
- O processo é repetido por várias rodadas.
- Ao final, o modelo global é avaliado e comparado com um modelo treinado de forma centralizada.

---

## ⚙️ Etapas de Implementação

1. Escolha e preparação de um dataset simples.
2. Divisão dos dados entre 5 clientes simulados.
3. Criação de um modelo preditivo (ex: regressão logística, árvore de decisão).
4. Treinamento local em cada cliente.
5. Envio dos pesos locais para o servidor.
6. Agregação com FedAvg e atualização do modelo global.
7. Repetição do ciclo por várias rodadas.
8. Avaliação final e comparação com modelo centralizado.

---

## 📊 Resultados Esperados

- Métricas de desempenho (acurácia, perda etc.) do modelo federado.
- Comparação com um modelo centralizado.
- Discussão sobre os benefícios e desafios do aprendizado federado.

---

## 📝 Relatório

Acompanhe o relatório para entender:

- Como os dados foram divididos.
- Como ocorreu o treinamento local.
- Como foi realizada a agregação dos modelos.
- Resultados comparativos e análise crítica.

---

## 💻 Tecnologias Utilizadas

- Python 3.10+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn

---

## 📌 Observações

Este projeto é uma **simulação local** (não distribuída em rede), com propósito **educacional** para compreender os fundamentos do aprendizado federado.
