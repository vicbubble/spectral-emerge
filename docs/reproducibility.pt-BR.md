# Notas de Reprodutibilidade

*[Read in English](reproducibility.md)*

Este documento descreve como reproduzir os principais experimentos deste repositório.
É intencionalmente honesto sobre o que é simples, o que requer configuração
e o que possui ressalvas conhecidas.

---

## Ambiente

**Versão testada do Python:** 3.13.x (Windows)

As dependências estão listadas em `requirements.txt`.

Instale com:

```bash
pip install -r requirements.txt
```

> **Nota:** o `torchdeq` pode exigir um caminho de instalação específico
> dependendo da sua versão PyTorch. Se a instalação via PyPI falhar, tente:
> ```bash
> pip install git+https://github.com/locuslab/torchdeq
> ```

---

## Rodando a suíte de testes

```bash
cd spectral_emerge
python -m pytest tests/ -v
```

A suíte de testes foca nos invariantes de comportamento de software.
Ela **não** afirma ou cruza limiares com garantias ativas base de performance do modelo exata limitável ou matemática.

**Categorias dos Testes Conhecidos:**

| Arquivo     | Foco de verificação      |
|-----------|---------------|
| `tests/test_temporal.py` | Garante ordenamento correto do OrderedBeatDataset, limiares isolados, ranges de escores em cache. |
| `tests/test_zeroshot.py` | Modelo travado 'congelado' fisicamente sem adapter; verificações de integridade sintética dos mapeamentos, padding linear |

Os 13 passos do repositório correm positivamente desde momento de download se testados os componentes no inicio sem anomalia no ambiente local ou instalação da engine em si, em local do ambiente Python seguro básico original testado (base PyTorch 2.1).

---

## Download de Dados

Arquivos reais físicos vindos dos ECG locais aplicáveis baixados pela ferramenta padrão pública base MIT a `wfdb` automáticos (no ambiente online) no rodar primeiro comando base original da avaliação a partir de seu acesso do PhysioNet:

```bash
python experiments/temporal_eval.py --config configs/ecg.yaml
```

Isto vai pedir acesso contínuo real de internet, em caso offline não será permitido seguir a avaliação inteira real e os arquivos ficam presos locais gravados temporariamente locais (já salvos e mapeados do ambiente na exclusão por .gitignore padrão por limites tamanhos reais em git repo base). 

Dados base originais Atlas de BGP RIPE são completamente opcionais da base física de roteamento físico. É baixado mas permite falha na avaliação cruzada original do modelo na fase opcional local da análise sem frear ou inviabilizar o veredicto negativo testado de zero-shot que avalia falsos nos sintéticos gerais da base já validada de teste base local do pacote Python e testes de ruídos também locais criados na base geral determinável local provados falsos (model falha de todo modo) no zero-shot da amostra que seja sintética (não trava execução).

---

## Executando (Reproduzindo) as Fases do Projeto e Resultados Gravados 

### Fase 4 (Reprodução - Sucesso) - Escore do ECG Contínuo (Métrica e Verificações Avaliações na Métrica temporal isolada)

```bash
python experiments/temporal_eval.py --config configs/ecg.yaml
```

**Esta parte inclui:**
1. Checkpoint `best_model.pt` gerado nativamente será levantado da memoria original interna e cacheada.
2. 10 arquivos ECGs do MIT baixados (100–109)
3. Executável o protocolo Leave-One-Record-Out
4. Métrica salva: `experiments/results/temporal_metrics.json` + `temporal_summary.md` original real

O *best_model.pt* foi extraído fisicamente (peso total) mas não enviado na matriz (arquivo pesava mais de ~megabytes, e os testes falharão em modelos instáveis recriados - **baixe do lugar próprio ou deixe link aqui e extraia em experiments/checkpoints e refaça comando test - Opcional e depende do ambiente online**) 

Sem isto tudo volta na pontuação aleatória randômica do erro sem peso inicial. O arquivo inicial caso mantido fisicamente deverá ser providenciado em repo git de arquivos extensos fisicamente anexado ou LFS na base remota online real antes!

### Fase 5 (Reprodução Restritiva e Negativa do Modelo Base) Zero-Shot Mapeada
O teste do escopo atestado físico do limite nativo: 

```bash
python experiments/zeroshot_eval.py
python experiments/zeroshot_ablation.py
```

Estas duas lógicas testadas:
- Sem retreinos
- Precisam do check-point (peso local inicial do base ECG original que aprende padrões vitais batimentos). 
- Cria os padronizadores sintéticos mapeadores da própria matriz do modelo local na Python base (constant-pad, noise, etc).

**Previsão exata verificável** 
- Sem adaptar nem usar adapter ou limite - AUROC será estático aleatoriamente mediano variando sem significado 0.51 ou similar no geral do cruzado de resultados nativos isolados avaliados em todas 3 tentativas criadas de mapas e inputs (AUROC ≈ 0.51–0.52 com ou sem falso padding randômico, e falha total confirmada - O Modelo de fato NÃO processava sinais distantes dos originários. 
(Se atestar algo acima será uma provável randomicidade errada / overfittings graves na sua versão baixada/versão. Falsos alarmes em ambientes e falsas impressões serão anuladas).

---

## Resumo Breve Limitações e Escopo

| Problema Original Avaliável Local | Ponto / Avaliação Métrica  |
|--------|--------|
| Checkpoint base matriz do ECG real | O peso da matriz é imutável: Sem o peso, modelo será lixo. Testes dão erro na rede do checkpoint e nada funcionará (AUROC zero aleatório sem a memória original extraída). |
| O ECG via API | Pode vir e ser lerdo ou travar base local sem internet rápida acessível na criação original da `wfdb` online física inicial. |
| Fase Final do Regime Tempo-Longo | Fraco. O modelo se perde sem focar janelas originais, F1-Temporal fraco atestado na documentação em geral baseada e isolada das avaliações |
| Transferências Universais Puras/Zero | Zero. Falhas mapeáveis (padding zero/cross = erro sintético). É a evidência mais robusta no teste atestado geral testada sobre anulações da física/natureza cruzada! |
