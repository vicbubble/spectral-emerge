# Spectral Emerge

*[Read in English](README.md)*

**Exploração experimental independente de representações implícitas baseadas em energia para detecção de anomalias em sinais contínuos.**

---

## O que este projeto é

Este projeto explora se uma estrutura discreta útil pode emergir de
dinâmicas latentes contínuas — especificamente, se um modelo neural implícito
baseado em equilíbrio pode aprender estados internos a partir de sinais
contínuos, e se sua função de energia atua como um sinal útil de anomalia
sem supervisão explícita.

Eu não sou físico nem teórico de machine learning. Movo-me aqui por
uma exploração prática de uma hipótese conceitual.

> **Inspiração Científica:** Este projeto foi originalmente inspirado, em nível conceitual, pelo paper *"Emergent quantization from a dynamic vacuum"* (White et al., Physical Review Research, 2026), que explora a possibilidade de estrutura discreta emergir de uma dinâmica contínua. Os experimentos deste repositório foram motivados por essa intuição geral, mas **não** constituem uma validação da teoria física proposta no artigo.

A resposta até agora:

- **Sim, parcialmente** — detecção de anomalias baseada em energia funciona em dados reais de ECG
- **Não** — transferência forte zero-shot entre domínios de ECG para latência de rede não foi demonstrada
- **Incerto / não demonstrado** — descoberta robusta de regimes; estrutura de estados emergentes universal

Este repositório é compartilhado de forma aberta para que outros possam inspecionar, reutilizar, criticar ou estender os experimentos.

---

## Resultados principais

### 1. Detecção de anomalias em ECG real (positivo)

Usando dados de ECG do PhysioNet MIT-BIH, o escore de energia do modelo produziu um sinal
de anomalia significativo sem usar rótulos no momento da inferência.

| Escore | AUROC Médio |
|-------|-----------|
| `energy_only` (primário, não supervisionado) | **0.801** |
| `centroid_distance` (secundário, não supervisionado) | 0.627 |

Protocolo: leave-one-record-out em 10 registros. Nenhum rótulo usado na inferência.

### 2. Avaliação temporal (misto)

Análise sequencial de batimentos sobre registros de ECG:
- Sinal de detecção de anomalias confirmado (AUROC ≈ 0.80)
- Detecção de transição de regime: F1 < 0.30 — abaixo da meta, resultado fraco
- Hipótese de iterações do DEQ: não consistentemente suportada

### 3. Transferência cross-domain zero-shot (negativo)

O modelo de ECG foi testado diretamente em dados sintéticos de latência de rede sem nenhuma adaptação:

| Mapeamento | AUROC Sintético |
|---------|----------------|
| constant_pad | 0.519 |
| reflect_pad | 0.510 |
| linear_resize | 0.513 |

Veredito: **NÃO** — transferência zero-shot não suportada. AUROC ≈ acaso em todos os mapeamentos.
Ablação confirmou a ausência de artefatos estruturais devido ao preenchimento (padding).

---

## Conclusão honesta

Este projeto **não** suporta:
- Representações emergentes universais entre domínios de sinais
- Teoria física validada de emergência
- Detecção de anomalias zero-shot entre domínios

Este projeto **suporta**:
- Detecção supervisionada reduzida de anomalias baseada em energia funcionando em ECG real
- Histórico de experimentos aberto e reproduzível, incluindo resultados negativos

---

## Estrutura do repositório

```text
src/                        código central do modelo e avaliação
configs/                    arquivos de configuração de experimentos
experiments/                scripts de experimentos
experiments/results/        métricas JSON, resumos e figuras salvos
tests/                      testes de comportamento de software (13 testes)
docs/                       notas do projeto, índice de resultados, guia de reprodutibilidade
paper/                      reportagem curta
```

## Ordem de leitura recomendada

1. [`docs/project_note.pt-BR.md`](docs/project_note.pt-BR.md) — histórico narrativo
2. [`paper/short-report.pt-BR.md`](paper/short-report.pt-BR.md) — resultados estruturados
3. [`experiments/results/temporal_summary.md`](experiments/results/temporal_summary.md) — resultado do ECG
4. [`experiments/results/zeroshot_network_summary.md`](experiments/results/zeroshot_network_summary.md) — resultado zero-shot negativo
5. [`docs/results_index.pt-BR.md`](docs/results_index.pt-BR.md) — índice de figuras e artefatos

---

## Reprodução

```bash
pip install -r requirements.txt
python -m pytest tests/ -v

# Fase 4 — Avaliação temporal do ECG
python experiments/temporal_eval.py --config configs/ecg.yaml

# Fase 5 — Cross-domain zero-shot
python experiments/zeroshot_eval.py
python experiments/zeroshot_ablation.py
```

Veja [`docs/reproducibility.pt-BR.md`](docs/reproducibility.pt-BR.md) para notas completas, incluindo ressalvas.

---

## Para que este projeto é bom

Escopo atual e realista:

- **Detecção de anomalias não supervisionada em sinais contínuos** (ECG, monitoramento fisiológico, fluxos de sensores)
- Cenários em que anomalias rotuladas são escassas
- Prototipagem de pesquisa com arquiteturas implícitas/DEQ

---

## Licença

MIT — veja [`LICENSE`](LICENSE)

## Citação

Se reutilizar ideias, código ou resultados, por favor cite este repositório.
Veja [`CITATION.cff`](CITATION.cff).

---

*Este repositório documenta um experimento independente. Não é um pacote de pesquisa polido. Resultados negativos são mantidos e documentados.*
