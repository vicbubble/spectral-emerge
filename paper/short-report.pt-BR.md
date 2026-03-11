# Spectral Emerge: Detecção de Anomalias Baseada em Energia com Dinâmicas Latentes Implícitas

*[Read in English](short-report.md)*

**Relatório experimental independente — v0.1**

---

## Resumo

Descrevemos uma exploração experimental independente sobre se dinâmicas
neurais implícitas baseadas em equilíbrio podem produzir representações internas
úteis para detecção de anomalias não supervisionada em sinais contínuos.

A arquitetura combina uma camada implícita de ponto fixo (estilo DEQ) com uma
função de energia calculada no equilíbrio latente. A energia é usada como
um escore de anomalia sem rótulos no momento da inferência.

Em dados reais de ECG do MIT-BIH PhysioNet, o escore de energia atingiu um AUROC
médio de aproximadamente 0.80 em 10 registros sob um protocolo leave-one-record-out.
Este é um resultado positivo genuíno.

Um experimento subsequente cruzado entre domínios (zero-shot cross-domain)
— testando o modelo treinado em ECG diretamente em dados sintéticos de latência
de rede sem nenhuma adaptação — produziu um AUROC de aproximadamente 0.51
em todos os três mapeamentos de entrada testados, o que é indistinguível do
nível de chance aleatória. Este resultado negativo limita severamente qualquer
alegação de transferência universal de estrutura latente.

Ambos os resultados estão documentados de forma transparente.

---

## 1. Motivação

Este projeto foi originalmente inspirado, em nível conceitual, pelo paper *"Emergent quantization from a dynamic vacuum"* (White et al., Physical Review Research, 2026), que explora a possibilidade de estrutura discreta emergir de uma dinâmica contínua. Os experimentos deste repositório foram motivados por essa intuição geral, mas **não** constituem uma validação da teoria física proposta no artigo.

O ponto de partida foi uma questão conceitual:

> Um modelo cujo estado latente é definido por um equilíbrio implícito
> (em vez de um estado oculto explícito de RNN ou bottleneck discreto)
> pode aprender estruturas úteis a partir de sinais contínuos sem
> supervisão explícita?

A hipótese era de que quantidades similares à energia, computadas no ponto fixo,
poderiam servir como sinais de anomalia — sem rótulos, e potencialmente
através de domínios.

---

## 2. Visão Geral do Método

### 2.1 Arquitetura do Modelo

- **Codificador:** pequeno MLP, mapeia a entrada x ∈ ℝ^187 para contexto c ∈ ℝ^d
- **Camada DEQ:** solucionador implícito de ponto fixo, encontra z* tal que f(z*, c) = z*
- **Energia:** escalar E(z*, x) computada no equilíbrio
- **Regularização espectral:** penaliza o raio espectral Jacobiano em z*
  para incentivar paisagens de ponto fixo mais estáveis

O modelo processa cada entrada independentemente.
Sem conexões recorrentes. Sem modelo temporal.

### 2.2 Treinamento

Treinado em dados de ECG (PhysioNet MIT-BIH, registros 100–109, excluindo
um devido aos testes).
Perda: reconstrução + penalidade espectral.
Nenhum rótulo de anomalia usado durante o treinamento.

### 2.3 Escore de anomalia

**Escore principal:** `energy_only` — E(z*, x) normalizada mínimo-máximo
em cada registro.
Nenhum rótulo foi usado para calcular essa pontuação.
Reportado como a métrica principal não supervisionada zero-shot.

**Escore secundário:** distância centroidal em espaço latente (ajuste de
GMM não supervisionado).
**Escore calibrado (exploratório):** distância ao centroide "normal"
informado pelo rótulo.
O último é explicitamente isolado das reivindicações principais não supervisionadas.

---

## 3. Resultados com ECG (Fase 3–4)

### 3.1 Conjunto de Dados

Banco de Dados de Arritmia PhysioNet MIT-BIH.
Registros 100–109, entre 2000–3500 batimentos cada.
Protocolo: leave-one-record-out (10 folds).

### 3.2 Detecção de anomalia

| Escore | AUROC Médio | AUPR Médio |
|-------|-----------|-----------|
| `unsupervised_energy` | **0.801** | 0.185 |
| `unsupervised_centroid_distance` | 0.627 | 0.121 |
| `label_informed_normal_distance` | 0.788 | 0.289 |

AUROC de pontuação primária > 0.75 — excede o critério de sucesso “forte”.

### 3.3 Avaliação temporal (Fase 4)

Avaliação temporal na sequência de batimentos em intervalos de nível de ritmo ordenados:

- Sinal de detecção de anomalias detectado robusto: (AUROC ≈ 0.80 consistente com anterior)
- Detecção de transição de regime com F1 < 0.30 (resultado considerado bem fraco quando validado com alvo planejado)
- Iterações do solucionador relativas não parecem servir repetidas vezes como fonte provida para pistas fortes relativas e seguras

**Interpretação:** hoje o modelo atua bem como detector de eventos via energia, e falta demonstração sobre seu comportamento de regimes.

---

## 4. Transferência Zero-Shot Domínio-Cruzado (Fase 5)

### 4.1 Configuração

- O modelo foi carregado de checkpoint em ECG, e operado completamente "congelado" (frozen)
- Sem novo treinamento, adaptadores ou uso de métrica otimizada em limiar
- Entrada nova alvo (streams com 4 regimes): Redes simuladas determinísticas sem pontes novas aplicadas
- Escores mapeadores: Escore de Energia único (e nenhum rótulo extra aplicado no processamento final de verificação)

### 4.2 Resultados Finais na Latência

| Teste | AUROC Sintético |
|---------|---------------|
| `constant_pad` | 0.519 |
| `reflect_pad` | 0.510 |
| `linear_resize` | 0.513 |

Isso equivale a uma média aleatória entre variáveis. As diferenças menores mapeadas nos números variaram muito minimamente sob 0.008 em distorções.

**Veredito: NÃO** — transferência zero-shot geral e ampla falhou e o resultado não ocorreu.

### 4.3 Experimento com entradas em distorções sem parâmetros padronizados

Ao serem avaliadas fatias aleatórias e partes embaralhadas que destroem lógicas construtivas originais não foi perceptível nenhuma grande alteração nem salto claro em pontuação real em comparativos aleatórios gerais. Sem sinal.

---

## 5. Geometria das Espacialidades Observadas em PCA (Domínios sem contato de Latentes)

O distanciamento cruzado visual nos exibe resultados separados (0.23 distância de semelhanças e pontos dispersos geometricamente nos quadrantes que impedem relações métricas de cruzamentos entre si em domínios alocados divergentes sem correlação imediata de distanciamentos úteis ou paralelos nos seus planos em geral).

Isso valida mais fortemente que os dados processados ocorrem sob regras físicas distintas processuais que impedem o uso final pretendido zero-shot. Tendo limites estritos nas amostras gerais em cruzamentos.

---

## 6. Conclusões

Como resultado final se replicou positivamente as detecções isoladas com valores sólidos do modelo sem validações guiadas. A exploração real na parte física do modelo sem adaptação cruzada obteve forte limite sendo negada e rejeitada nesta fase original proposta e sem indicação temporal final garantida. 

Passos futuros podem focar na camada cruzando dados com 10% ou pouca supervisão e validando EEG similares seguindo rigor nas lógicas base da análise contínua matemática de mapeamentos implícitos gerais.

---

*Isto consiste num acompanhamento independente, código incluído em todos seus passos, aberto para verificação e não avaliado formalmente em academias pares normais. Todos dados contidos podem ser lidos publicamente abertos.*
