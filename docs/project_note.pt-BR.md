# Nota do Projeto

*[Read in English](project_note.md)*

## O que é isto

Este repositório não é um paper acadêmico formal.
Ele é um registro organizado de uma exploração experimental independente.

O projeto começou a partir de uma curiosidade simples:

> um modelo com dinâmica implícita de equilíbrio pode desenvolver estrutura interna útil a partir de sinais contínuos? E essa estrutura pode ser usada de forma prática?

A implementação evoluiu em cinco fases.
Algumas ideias funcionaram melhor do que eu esperava.
Outras falharam de forma clara.
Ambos os tipos de resultado estão mantidos aqui.

---

## Motivação original

A motivação original deste projeto veio da leitura do paper *“Emergent quantization from a dynamic vacuum”*, de Harold White e colaboradores, publicado em *Physical Review Research* em 2026. O artigo trabalha, no contexto da física teórica, a ideia de que estruturas discretas podem emergir de uma dinâmica contínua. 

Esta exploração computacional foi inspirada por essa intuição conceitual, mas os resultados aqui apresentados devem ser lidos como uma investigação independente em modelagem e detecção de anomalias em sinais, e **não** como teste ou validação direta do framework físico do artigo.

Eu estava interessado na possibilidade de que:
- dinâmicas latentes contínuas
- pontos de equilíbrio implícitos
- e regularização espectral do Jacobiano no equilíbrio

pudessem produzir estados internos úteis sem impor explicitamente uma estrutura discreta no modelo. A intuição era de que uma utilidade discreta pudesse *emergir* em vez de ser projetada desde o início.

Essa motivação mais ampla continua intelectualmente interessante.
Mas os resultados obtidos até aqui não justificam claims grandes sobre representações emergentes universais.

O resultado prático mais forte hoje é mais estreito:
o modelo aprendeu um sinal útil de anomalia em ECG real usando apenas a energia em seu ponto fixo implícito.

---

## O que foi realmente testado

### Fase 1–2: Experimentos sintéticos
Nas fases iniciais, exploramos:
- se modos latentes emergiam em dados controlados
- se a regularização espectral do Jacobiano mudava a geometria latente
- se havia colapso em dimensões latentes mais altas

Esses experimentos ajudaram bastante a diagnosticar o comportamento do sistema e moldar intuições, mas não provaram por si só um princípio novo forte.

### Fase 3: Dados de ECG — primeiro resultado positivo
O primeiro resultado aplicado convincente veio dos dados de ECG do MIT-BIH PhysioNet:
- dados reais e anotados de ECG
- avaliação não supervisionada (nenhum rótulo no momento da inferência)
- sinal de anomalia acima do acaso em vários registros

Um protocolo de leave-one-record-out foi usado para evitar vazamento de dados.
A pontuação primária — a energia do modelo no ponto fixo implícito — foi computada sem qualquer acesso aos rótulos de anomalia.

Foi neste ponto que o projeto começou a parecer praticamente significativo.

### Fase 4: Avaliação temporal
Foi adicionada uma avaliação temporal sobre sequências ordenadas de batimentos cardíacos.
O modelo continuou a processar cada batimento independentemente; a camada temporal tratava-se de *avaliação*, e não da adição de um modelo temporal.

Principais descobertas:
- sinal razoável de detecção de anomalia (AUROC médio ≈ 0.80 com escore de energia)
- sinal fraco de transição de regime (F1 abaixo da meta)
- a hipótese de iterações do DEQ não foi consistentemente suportada como um sinal de anomalia

Esta fase esclareceu o que o modelo realmente estava fazendo:
detecção útil de anomalias no nível do evento, análise de nível de regime mais fraca.

### Fase 5: Transferência zero-shot entre domínios (cross-domain)
Um experimento crítico testou se o modelo treinado em ECG generalizaria para um domínio de sinais muito diferente (dados sintéticos de latência/RTT de rede):
- sem retreinamento
- sem adapter
- sem ponte aprendida
- apenas mapeamento determinístico de entradas (constant-padding, reflection-padding, linear resize)

Três modos de mapeamento foram testados. Todos falharam:
- AUROC ≈ 0.51–0.52 em todos os mapeamentos (nível aleatório)
- Ablação confirmou: entradas embaralhadas e com ruído aleatório geraram escores parecidos com as reais, descartando transferências de estruturas reais

Esse resultado negativo é extremamente importante porque limita com clareza o escopo da claim mais forte que se poderia querer fazer.

---

## Interpretação honesta

O projeto atualmente suporta que:

1. A arquitetura pode produzir sinais de anomalia baseados em energia úteis em ECG real.
2. A história sobre regimes e estados temporais continua mais fraca do que a história sobre anomalia.
3. Universalidade forte zero-shot entre domínios não é suportada pelos experimentos atuais.

---

## Por que isso ainda vale a pena ser compartilhado

Um resultado negativo limpo é útil.
Um resultado positivo modesto, mas real, é útil.
Registros de experimentos abertos e legíveis são úteis.

O projeto não é apresentado como verdade final.
Ele é apresentado como uma exploração séria, com artefatos inspecionáveis.

Isso, por si só, pode ajudar outras pessoas a:
- evitar exageros parecidos
- reutilizar as partes de detecção de anomalia
- testar versões mais fortes da hipótese original
- ou desenvolver o código em uma direção mais rigorosa.
