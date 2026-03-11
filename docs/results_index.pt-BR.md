# Índice de Resultados

*[Read in English](results_index.md)*

Índice rápido dos artefatos salvos mais importantes.
Destinado a ajudar um novo visitante a encontrar os principais resultados rapidamente.

---

## Detecção de Anomalias em ECG

### Resumos principais

| Arquivo | Descrição |
|------|-------------|
| `experiments/results/temporal_summary.md` | Resumo legível por humanos sobre a avaliação temporal através dos 10 registros de ECG. O documento de resultado mais importante. |
| `experiments/results/temporal_metrics.json` | JSON completo com as métricas por registro e tipo de escore. Fonte da verdade para todos os números reportados. |
| `experiments/results/ecg_summary_phase3.md` | Resumo inicial da avaliação do ECG na Fase 3 (pré-temporal). |

### Por que eles importam

A avaliação temporal estabeleceu o principal resultado positivo:
detecção de anomalia baseada em energia em sinais reais de ECG com desempenho
acima do nível aleatório (random-level), sem utilizar rótulos na hora da inferência.

---

### Figuras de ECG selecionadas

| Arquivo | Descrição |
|------|-------------|
| `experiments/results/figures/timeline_100.png` | Linha do tempo do escore de anomalia por batimento para o registro 100. Padrão típico "na maioria normal com anomalias ocasionais". |
| `experiments/results/figures/timeline_102.png` | Registro 102 com alta prevalência de anomalias. Mostra a resposta do escore de energia longo do registro. |
| `experiments/results/figures/roc_100_unsupervised_energy.png` | Curva ROC para o escore de energia no registro 100. Representativo do escore não-supervisionado com melhor desempenho. |
| `experiments/results/figures/ecg_latent_tsne_phase3.png` | t-SNE / projeção latente da Fase 3. Mostra como o espaço latente do ECG é clusterizado — não perfeitamente separado, mas bem estruturado. |

**Por que foram selecionados:** estas figuras contam a história da anomalia no ECG
de forma concisa — um registro típico, um com alta prevalência, uma curva ROC,
e uma visão na geometria do espaço latente.

---

## Zero-Shot Cross-Domain (Resultado Negativo)

### Resumos principais

| Arquivo | Descrição |
|------|-------------|
| `experiments/results/zeroshot_network_summary.md` | Resumo legível do teste zero-shot cross-domain. Inclui uma tabela do veredicto negativo final e o limite lógico atestado contra universalidade da arquitetura de modelo de rede natural gerada pela análise base e a aplicação. |
| `experiments/results/zeroshot_network_metrics.json` | JSON com pontuações plenas sobre os 3 mapeamentos que foram atestadamente fracassados nos dados finais das redes artificiais criadas sem ajustes diretos (adapter, padding e ref). |
| `experiments/results/zeroshot_ablation_combined.json` | Resultados sobre anulações: compara janelas limpas da estrutura vs ruidos embaralhados em randomicidade completa na pontuação e mapeamentos cruzados provando ruído nativo real sem ganho estrutural para zero-shot em limites falsos possíveis em mappings. |

### Figuras selecionadas no Zero-Shot

| Arquivo | Descrição |
|------|-------------|
| `experiments/results/figures/zeroshot_latent_comparison_reflect_pad.png` | PCA da espacialidade do ECG projetada sobre os domínios mapeados em PCA com "anomalous-network" z* vetores vs stable-network. Demonstra o distanciamento nítido e visual de isolamento entre classes mapeadas latentes. |
| `experiments/results/figures/timeline_network_synthetic_reflect_pad.png` | Plot histórico timeline com anomalias de sintéticos que foram completamente planificados. Reflete claramente na prática o modelo que fracassa a detecção e varia randomicamente plano falhando mapeamento útil generalista na anomalia total em si. |
| `experiments/results/figures/energy_hist_synthetic_reflect_pad.png` | Curva comparativa nas energias gerando distribuições do normal x as pontas de janelas da rede com erros — completamente alocadas sobrepostas negando assim utilidade e divisão útil e prática (fracasso total em discriminar e identificar as distribuições locais) |

**Causas do uso documentado:** As informações mostram de porquês as falhas do modelo, da estrutura geométrica mapeável falsa em distanciamento prático do modelo falso entre latentes de energia. 

---

## Modelagens Analíticas (Fases Iniciais - 1-2)

| Arquivo | Descrição |
|------|-------------|
| `experiments/results/figures/fixedpoint_landscape.png` | Panorama do atrator. A única dimensão atrai valores aos pares do centro e levava analiticamente à averiguação do colapso possível ou distorções originais na teoria aplicada do espaço dimensionado da base teórica matemática testada em DEQ (Deep Equilibrium). |
| `experiments/results/figures/jacobian_spectrum_comparison.png` | Demonstra comparativos e raio da métrica nos Jacobian regularizados x as Feedforward tradicionais limitadoras nas partes dos modelos em espaços menores mapeados em 2D. |
| `experiments/results/collapse_sweep.json` | JSON das métricas avaliadas nas conformações das funções limitadoras do atrator, nas análises que o colapso dos modelos base seria uma das maiores fontes dos fracassos da física da teoria em testes mais complexos provadas também empíricos nestas lógicas atestadas cruzadas ao ser modelada fisicamente nos testes computacionais em si do repo documentado |
