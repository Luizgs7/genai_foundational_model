"""Gera o relatório visual de análise exploratória (EDA) da base H&M a partir de
runs/hm_clientes/eda/{eda.json, eda_relatorio.json}.

Uso: python3 pipeline/relatorios/gerar_relatorio_eda_hm.py
Gera runs/hm_clientes/relatorios/eda_hm.html
"""

import json
import os

EDA = "runs/hm_clientes/eda"
OUT = "runs/hm_clientes/relatorios/eda_hm.html"

HTML = r"""<title>Base H&amp;M por dentro</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* Layout: relatório em coluna única de leitura (~980px), seções numeradas pelo
   caminho que o dado percorre; gráficos SVG próprios com tooltip. Mesma família
   visual do relatório "rastro do cliente" da base sintética. */
:root {
  --bg: #faf9f5; --surface: #ffffff; --surface-alt: #f1efe6; --border: #e2dfd2;
  --ink: #201f1a; --ink-dim: #6e6c60; --ink-faint: #9a9787;
  --accent: #2b6e5e; --accent-strong: #1e4f44; --accent-soft: #e3eeea;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s1-soft: #cde2fb;
  --band-train: rgba(42,120,214,.07); --band-val: rgba(235,104,52,.10); --band-test: rgba(27,175,122,.10);
  --covid: rgba(208,59,59,.08); --grid: #ece9de;
  --f-display: "IBM Plex Sans", system-ui, sans-serif;
  --f-body: "Source Serif 4", Georgia, serif;
  --f-mono: "JetBrains Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #14181a; --surface: #1b2023; --surface-alt: #20262a; --border: #2b3236;
  --ink: #eae8df; --ink-dim: #a3a094; --ink-faint: #6f6c60;
  --accent: #6fcba9; --accent-strong: #8fe0bf; --accent-soft: #1e2b27;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s1-soft: #1c3557;
  --band-train: rgba(57,135,229,.10); --band-val: rgba(217,89,38,.13); --band-test: rgba(25,158,112,.13);
  --covid: rgba(230,103,103,.12); --grid: #262c30; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #14181a; --surface: #1b2023; --surface-alt: #20262a; --border: #2b3236;
  --ink: #eae8df; --ink-dim: #a3a094; --ink-faint: #6f6c60;
  --accent: #6fcba9; --accent-strong: #8fe0bf; --accent-soft: #1e2b27;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s1-soft: #1c3557;
  --band-train: rgba(57,135,229,.10); --band-val: rgba(217,89,38,.13); --band-test: rgba(25,158,112,.13);
  --covid: rgba(230,103,103,.12); --grid: #262c30; color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--f-body); font-size: 16px; line-height: 1.6; }
.page { max-width: 980px; margin: 0 auto; padding-block: 48px 96px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.eyebrow { font-family: var(--f-display); font-weight: 600; font-size: .72rem; letter-spacing: .09em; text-transform: uppercase; color: var(--accent); margin: 0 0 12px; }
h1 { font-family: var(--f-display); font-weight: 700; font-size: clamp(2rem, 4.2vw, 2.8rem); line-height: 1.08; margin: 0 0 16px; text-wrap: balance; }
h2 { font-family: var(--f-display); font-weight: 700; font-size: 1.45rem; line-height: 1.25; margin: 0 0 10px; text-wrap: balance; }
h3 { font-family: var(--f-display); font-weight: 600; font-size: .95rem; margin: 26px 0 8px; }
p { margin: 0 0 14px; max-width: 70ch; }
.lede { font-size: 1.08rem; color: var(--ink-dim); max-width: 64ch; }
.dim { color: var(--ink-dim); }
.mono, code { font-family: var(--f-mono); font-size: .86em; }
code { background: var(--surface-alt); padding: .05em .35em; border-radius: 4px; }
header { margin-bottom: 56px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 26px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
.tile b { display: block; font-family: var(--f-mono); font-size: 1.35rem; font-weight: 500; color: var(--accent-strong); font-variant-numeric: tabular-nums; }
.tile span { font-family: var(--f-display); font-size: .78rem; color: var(--ink-dim); }
nav.toc { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 22px; font-family: var(--f-display); font-size: .82rem; }
nav.toc a { color: var(--accent-strong); text-decoration: none; border-bottom: 1px solid var(--accent-soft); }
nav.toc a:hover, nav.toc a:focus-visible { border-bottom-color: var(--accent); outline: none; }
section { margin-top: 72px; scroll-margin-top: 16px; }
.chart { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px 14px 8px; margin: 10px 0 6px; position: relative; }
.chart svg { display: block; width: 100%; height: auto; overflow: visible; }
.chart-title { font-family: var(--f-display); font-weight: 600; font-size: .86rem; margin: 0 0 2px; }
.chart-sub { font-family: var(--f-display); font-size: .76rem; color: var(--ink-dim); margin: 0 0 6px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-family: var(--f-display); font-size: .74rem; color: var(--ink-dim); margin: 4px 0 2px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
@media (max-width: 760px) { .grid2 { grid-template-columns: minmax(0, 1fr); } }
.source { font-family: var(--f-mono); font-size: .72rem; color: var(--ink-faint); margin: 4px 0 0; }
.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; background: var(--surface); margin: 10px 0; }
table { border-collapse: collapse; width: 100%; font-family: var(--f-display); font-size: .84rem; }
th { text-align: left; font-weight: 600; font-size: .68rem; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-dim); padding: 10px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }
td { padding: 8px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; font-family: var(--f-mono); font-size: .8rem; white-space: nowrap; }
.pill { font-family: var(--f-mono); font-size: .7rem; padding: 1px 8px; border-radius: 999px; white-space: nowrap; }
.p-train { background: var(--band-train); color: var(--s1); border: 1px solid var(--s1); }
.p-val { background: var(--band-val); color: var(--s2); border: 1px solid var(--s2); }
.p-test { background: var(--band-test); color: var(--s3); border: 1px solid var(--s3); }
.callout { background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 6px; padding: 14px 18px; margin: 18px 0; }
.callout b.t { display: block; font-family: var(--f-display); font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; color: var(--accent); margin-bottom: 6px; }
.callout p { font-family: var(--f-display); font-size: .88rem; color: var(--ink-dim); margin: 0 0 6px; }
.callout p:last-child { margin: 0; }
ol.decisoes { padding-left: 1.2em; max-width: 72ch; }
ol.decisoes li { margin: 0 0 12px; }
ol.decisoes b { font-family: var(--f-display); }
#tip { position: fixed; pointer-events: none; z-index: 20; background: var(--ink); color: var(--bg); font-family: var(--f-display); font-size: .76rem; line-height: 1.4; padding: 6px 9px; border-radius: 5px; max-width: 260px; opacity: 0; transition: opacity .08s; }
#tip b { font-family: var(--f-mono); font-weight: 500; }
.axis text, .lbl { font-family: var(--f-display); font-size: 11px; fill: var(--ink-dim); }
.lbl-strong { font-family: var(--f-mono); font-size: 11px; fill: var(--ink); }
.gridline { stroke: var(--grid); stroke-width: 1; }
.baseline { stroke: var(--ink-faint); stroke-width: 1; }
.hit { fill: transparent; cursor: default; }
footer { margin-top: 90px; padding-top: 18px; border-top: 1px solid var(--border); font-family: var(--f-mono); font-size: .74rem; color: var(--ink-faint); }
@media (prefers-reduced-motion: reduce) { #tip { transition: none; } }
</style>

<div class="page">
<header>
  <p class="eyebrow">Customer Foundation Model · experimento H&amp;M · análise exploratória</p>
  <h1>A base H&amp;M por dentro</h1>
  <p class="lede">Dois anos de compras de uma varejista de moda: o que tem em cada tabela, como os clientes compram,
  o que o calendário e a pandemia fizeram com os números, e quais escolhas de modelagem saíram daqui.</p>
  <div class="tiles" id="tiles"></div>
  <nav class="toc" aria-label="Seções">
    <a href="#estrutura">Tabelas</a><a href="#tempo">Tempo e COVID</a><a href="#clientes">Como os clientes compram</a>
    <a href="#catalogo">Catálogo e preço</a><a href="#cadastro">Cadastro</a><a href="#rotulos">Rótulos</a>
    <a href="#sequencias">Sequências</a><a href="#comparacao">As três bases</a><a href="#decisoes">Decisões</a>
  </nav>
</header>

<section id="estrutura">
  <p class="eyebrow">Tabelas de origem</p>
  <h2>Três tabelas, uma linha por item vendido</h2>
  <p>O dataset é o da competição H&amp;M Personalized Fashion Recommendations (Kaggle, 2022), lido do espelho no
  HuggingFace <code>einrafh/hnm-fashion-recommendations-data</code>. Cada linha de <code>transactions</code> é um item
  comprado; não há id de pedido, então tratamos <b>cliente + dia</b> como a cesta de compra.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>Tabela</th><th class="num">Linhas</th><th>Colunas</th><th>Como entra no modelo</th></tr></thead>
    <tbody id="t-estrutura"></tbody>
  </table></div>
  <p class="source">dados_hm/*.parquet · adapters/adapter_hm_clientes.py</p>
</section>

<section id="tempo">
  <p class="eyebrow">Tempo</p>
  <h2>Sazonalidade forte e um teste que cai em plena pandemia</h2>
  <p>O volume semanal tem dois picos de verão (junho de 2019 e de 2020) e um vale em fevereiro. Os splits são por
  data: treino até 26/12/2019, validação até 25/03/2020 e teste de 26/03/2020 ao fim da base.</p>
  <div class="chart" id="c-semanal"></div>
  <div class="chart" id="c-online"></div>
  <div class="callout"><b class="t">O que isso muda</b>
    <p>Entre 23/03 e 03/05/2020 <b>100% das vendas foram online</b>: as lojas fecharam. O teste começa três dias
    depois. A tarefa "próxima cesta é online" tem taxa de 58% no treino e 62% no teste, e a de churn em 90 dias cai
    de 30% na validação para 16% no teste. Isso é deslocamento de distribuição real, e foi mantido de propósito:
    é o tipo de choque que um modelo em produção enfrenta.</p>
  </div>
  <p class="source">runs/hm_clientes/eda/eda_relatorio.json · semanal</p>
</section>

<section id="clientes">
  <p class="eyebrow">Comportamento de compra</p>
  <h2>Muitos clientes ocasionais, uma cauda que compra muito</h2>
  <p id="p-clientes"></p>
  <div class="grid2">
    <div class="chart" id="c-itens-cliente"></div>
    <div class="chart" id="c-cestas-cliente"></div>
  </div>
  <div class="grid2">
    <div class="chart" id="c-itens-cesta"></div>
    <div class="chart" id="c-gap"></div>
  </div>
  <div class="callout"><b class="t">Por que a âncora é a cesta</b>
    <p>A cesta média tem 3,5 itens. Se cada item fosse um exemplo, todo item seguido de outro da mesma cesta teria
    "próxima compra em 0 dias", um rótulo trivial que infla qualquer métrica. Por isso rótulos e exemplos
    downstream ficam no último item de cada cliente + dia, e o "próximo" é a próxima cesta.</p>
    <p>O horizonte de churn saiu do histograma ao lado: 15% dos intervalos entre cestas passam de 90 dias.
    Com N=90 a taxa de churn fica em 24,7% no treino, nem rara nem trivial.</p>
  </div>
</section>

<section id="catalogo">
  <p class="eyebrow">Catálogo e preço</p>
  <h2>Moda feminina domina; o catálogo tem cauda longa</h2>
  <div class="grid2">
    <div class="chart" id="c-index"></div>
    <div class="chart" id="c-concentracao"></div>
  </div>
  <div class="chart" id="c-preco"></div>
  <p>O preço vem normalizado na fonte (a H&amp;M não publicou valores em moeda). Como a faixa de preço muda muito
  por grupo, o token de preço é o decil <b>dentro do grupo de produto</b>, calculado só no treino: uma meia cara
  e um casaco barato não são comparáveis em valor absoluto.</p>
  <p class="source">runs/hm_clientes/eda/eda_relatorio.json · preco_por_grupo, concentracao_artigos</p>
</section>

<section id="cadastro">
  <p class="eyebrow">Dados cadastrais</p>
  <h2>Idade com dois picos, poucos campos e muito nulo</h2>
  <div class="grid2">
    <div class="chart" id="c-idade"></div>
    <div>
      <div class="table-wrap"><table>
        <thead><tr><th>Campo</th><th>Valores</th><th class="num">Nulo</th></tr></thead>
        <tbody id="t-cadastro"></tbody>
      </table></div>
      <p class="source">dados_hm/customers_compacto.parquet</p>
    </div>
  </div>
  <p>O cadastro entra só na fusão e no GBDT, nunca nos tokens. FN e Active são nulos em ~65% dos clientes; pela
  convenção do dataset, nulo significa "não". O CEP vem hasheado e tem 352,9 mil valores, a maioria com até 3
  clientes. Na fusão ele vira top-1.000 + "OUTROS"; no GBDT, a contagem de clientes com o mesmo CEP.</p>
</section>

<section id="rotulos">
  <p class="eyebrow">Tarefas downstream</p>
  <h2>Quatro perguntas de negócio sobre a próxima cesta</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Split</th><th class="num">Cestas (âncoras)</th><th class="num">Churn 90d</th><th class="num">Próxima online</th></tr></thead>
    <tbody id="t-rotulos"></tbody>
  </table></div>
  <div class="grid2">
    <div class="chart" id="c-nextcat"></div>
    <div class="chart" id="c-ltv"></div>
  </div>
  <p>A faixa 2 do LTV concentra 31% das cestas porque reúne todos os clientes sem compra nos 90 dias seguintes
  (o valor zero ocupa os três primeiros decis). As demais faixas têm ~10% cada, como esperado de decis.</p>
  <p class="source">runs/hm_clientes/rotulos_downstream/rotulos.parquet</p>
</section>

<section id="sequencias">
  <p class="eyebrow">Sequências de tokens</p>
  <h2>351 milhões de tokens, 2.428 palavras no vocabulário</h2>
  <p>Cada item vira ~11 tokens: <code>EVT</code>, índice, grupo, tipo de produto, cor, linha, canal, artigo
  (top-2000 ou <code>OUTROS</code>), faixa de preço, recência e mês. Um cliente mediano tem 100 tokens; o p99, 2.058.</p>
  <div class="grid2">
    <div class="chart" id="c-tokens"></div>
    <div>
      <div class="table-wrap"><table>
        <thead><tr><th class="num">Contexto</th><th class="num">Clientes inteiros</th><th class="num">Eventos na janela</th></tr></thead>
        <tbody id="t-contexto"></tbody>
      </table></div>
      <p class="dim" style="font-size:.88rem">O contexto de 1.024 tokens é a referência. As âncoras que ficam antes
      da janela principal recebem uma janela própria na extração, então todas têm embedding.</p>
    </div>
  </div>
</section>

<section id="comparacao">
  <p class="eyebrow">Comparação</p>
  <h2>As três bases que o motor já processou</h2>
  <div class="table-wrap"><table>
    <thead><tr><th></th><th class="num">Sintética (agro)</th><th class="num">Olist (vendedores)</th><th class="num">H&amp;M (clientes)</th></tr></thead>
    <tbody id="t-comp"></tbody>
  </table></div>
  <p>A H&amp;M tem 318 vezes mais eventos que a base sintética e 254 vezes mais tokens de treino. É a primeira em
  que o backbone tem dados suficientes para ~12M de parâmetros (20 tokens de treino por parâmetro).</p>
</section>

<section id="decisoes">
  <p class="eyebrow">Implicações</p>
  <h2>O que a análise decidiu na modelagem</h2>
  <ol class="decisoes">
    <li><b>Âncora por cesta.</b> 9,08M cestas de 3,5 itens em média; evita rótulos triviais de gap zero.</li>
    <li><b>Churn com N=90 e teste de 180 dias.</b> 15% dos intervalos passam de 90 dias; a janela de teste precisa ter pelo menos o dobro do horizonte para o rótulo fechar.</li>
    <li><b>Preço em decis por grupo de produto.</b> As faixas de preço de cada grupo mal se sobrepõem.</li>
    <li><b>Artigo como top-2000 + OUTROS.</b> O top-2000 cobre 27% das vendas; o tipo de produto (130 valores) carrega o resto do sinal sem explodir o vocabulário.</li>
    <li><b>Contexto de 1.024 tokens com janelas extras.</b> Cobre 95% dos clientes inteiros; os demais não perdem âncoras.</li>
    <li><b>Backbone de ~12M parâmetros.</b> 227M tokens de treino pela regra de 20 tokens por parâmetro.</li>
    <li><b>FN/Active nulo = 0, idade nula = média do treino.</b> Convenção do dataset e imputação neutra.</li>
    <li><b>COVID mantida no teste.</b> O deslocamento é documentado e medido, não corrigido.</li>
  </ol>
</section>

<footer>Gerado por pipeline/relatorios/gerar_relatorio_eda_hm.py a partir de runs/hm_clientes/eda/*.json · branch luiz_g_product_exp</footer>
</div>
<div id="tip" role="tooltip"></div>

<script>
const E = __EDA__;
const R = __REL__;
const nf = (v, d = 0) => v.toLocaleString("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 1) => nf(v * 100, d) + "%";
const mil = v => v >= 1e6 ? nf(v / 1e6, v >= 1e7 ? 1 : 2) + " mi" : v >= 1e3 ? nf(v / 1e3, v >= 1e4 ? 0 : 1) + " mil" : nf(v);
const tip = document.getElementById("tip");
function showTip(ev, html) { tip.innerHTML = html; tip.style.opacity = 1;
  const x = Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 8), y = Math.max(8, ev.clientY - tip.offsetHeight - 10);
  tip.style.left = x + "px"; tip.style.top = y + "px"; }
function hideTip() { tip.style.opacity = 0; }
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function head(box, title, sub) { box.innerHTML = `<p class="chart-title">${title}</p>` + (sub ? `<p class="chart-sub">${sub}</p>` : ""); }
function niceMax(v) { const p = Math.pow(10, Math.floor(Math.log10(v))); for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p; return 10 * p; }

/* barras verticais (histograma / categorias) */
function colunas(id, title, sub, itens, o = {}) {
  const box = document.getElementById(id); head(box, title, sub);
  const W = Math.max(280, box.clientWidth - 28), H = o.h || 210, m = { l: 46, r: 8, t: 10, b: o.rot ? 62 : 34 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title }, box);
  const max = niceMax(Math.max(...itens.map(d => d.v))), iw = W - m.l - m.r, ih = H - m.t - m.b;
  const bw = iw / itens.length, gap = Math.min(6, bw * 0.18);
  for (let k = 0; k <= 4; k++) { const y = m.t + ih - ih * k / 4;
    el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: k ? "gridline" : "baseline" }, svg);
    el("text", { x: m.l - 6, y: y + 4, "text-anchor": "end", class: "lbl" }, svg).textContent = (o.fy || mil)(max * k / 4); }
  itens.forEach((d, i) => {
    const h = ih * d.v / max, x = m.l + i * bw + gap / 2, w = bw - gap, y = m.t + ih - h;
    const r = Math.min(4, w / 2, h);
    el("path", { d: `M${x},${m.t + ih} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${m.t + ih} Z`,
      fill: d.cor || "var(--s1)" }, svg);
    const t = el("text", { x: x + w / 2, y: H - m.b + 14, "text-anchor": o.rot ? "end" : "middle", class: "lbl",
      transform: o.rot ? `rotate(-35 ${x + w / 2} ${H - m.b + 14})` : "" }, svg); t.textContent = d.l;
    const hit = el("rect", { x: m.l + i * bw, y: m.t, width: bw, height: ih, class: "hit" }, svg);
    hit.addEventListener("mousemove", ev => showTip(ev, `${d.tip || d.l}<br><b>${(o.ft || (v => nf(v)))(d.v)}</b>` + (d.extra ? `<br>${d.extra}` : "")));
    hit.addEventListener("mouseleave", hideTip);
  });
  if (o.marca) { const i = o.marca.i, x = m.l + i * bw + (o.marca.centro ? bw / 2 : 0);
    el("line", { x1: x, x2: x, y1: m.t - 4, y2: m.t + ih, stroke: "var(--s2)", "stroke-width": 2, "stroke-dasharray": "4 3" }, svg);
    el("text", { x: x + 5, y: m.t + 8, class: "lbl-strong" }, svg).textContent = o.marca.t; }
}
function histItens(h, fmtL) {
  const b = h.bordas, c = h.contagens, tot = c.reduce((a, x) => a + x, 0);
  return c.map((v, i) => ({ v, l: fmtL(b[i], b[i + 1]), tip: fmtL(b[i], b[i + 1], true), extra: pct(v / tot) + " do total" }));
}
const faixa = (a, b, longo) => (b - a === 1 ? `${nf(a)}` : (longo ? `${nf(a)} a ${nf(b - 1)}` : `${nf(a)}–${nf(b - 1)}`));

/* barras horizontais (categorias) */
function barrasH(id, title, sub, itens, o = {}) {
  const box = document.getElementById(id); head(box, title, sub);
  const W = Math.max(280, box.clientWidth - 28), rowH = 24, m = { l: o.ml || 130, r: 56, t: 4, b: 6 }, H = m.t + m.b + rowH * itens.length;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title }, box);
  const max = Math.max(...itens.map(d => d.v)), iw = W - m.l - m.r;
  itens.forEach((d, i) => { const y = m.t + i * rowH + 4, w = Math.max(2, iw * d.v / max), h = rowH - 8;
    el("text", { x: m.l - 8, y: y + h - 3, "text-anchor": "end", class: "lbl" }, svg).textContent = d.l;
    const r = Math.min(4, h / 2);
    el("path", { d: `M${m.l},${y} H${m.l + w - r} Q${m.l + w},${y} ${m.l + w},${y + r} V${y + h - r} Q${m.l + w},${y + h} ${m.l + w - r},${y + h} H${m.l} Z`, fill: "var(--s1)" }, svg);
    el("text", { x: m.l + w + 6, y: y + h - 3, class: "lbl-strong" }, svg).textContent = (o.ft || pct)(d.v);
    const hit = el("rect", { x: 0, y: m.t + i * rowH, width: W, height: rowH, class: "hit" }, svg);
    hit.addEventListener("mousemove", ev => showTip(ev, `${d.l}<br><b>${(o.ft || pct)(d.v)}</b>` + (d.extra ? `<br>${d.extra}` : "")));
    hit.addEventListener("mouseleave", hideTip); });
}

/* linha temporal com faixas de split e crosshair */
function linha(id, title, sub, xs, ys, o = {}) {
  const box = document.getElementById(id); head(box, title, sub);
  if (o.legenda) box.insertAdjacentHTML("beforeend", `<div class="legend">${o.legenda}</div>`);
  const W = Math.max(300, box.clientWidth - 28), H = o.h || 230, m = { l: 50, r: 10, t: 12, b: 26 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title }, box);
  const t = xs.map(s => new Date(s + "T00:00:00").getTime()), t0 = t[0], t1 = t[t.length - 1];
  const iw = W - m.l - m.r, ih = H - m.t - m.b, X = v => m.l + iw * (v - t0) / (t1 - t0);
  const ymax = o.ymax || niceMax(Math.max(...ys)), Y = v => m.t + ih - ih * v / ymax;
  (o.faixas || []).forEach(f => { const a = X(new Date(f.de).getTime()), b = X(new Date(f.ate).getTime());
    el("rect", { x: a, y: m.t, width: Math.max(0, b - a), height: ih, fill: f.cor }, svg);
    if (f.rot) el("text", { x: a + 4, y: m.t + 12, class: "lbl" }, svg).textContent = f.rot; });
  for (let k = 0; k <= 4; k++) { const v = ymax * k / 4, y = Y(v);
    el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: k ? "gridline" : "baseline" }, svg);
    el("text", { x: m.l - 6, y: y + 4, "text-anchor": "end", class: "lbl" }, svg).textContent = (o.fy || mil)(v); }
  ["2018-10-01", "2019-01-01", "2019-04-01", "2019-07-01", "2019-10-01", "2020-01-01", "2020-04-01", "2020-07-01"].forEach(d => {
    const x = X(new Date(d + "T00:00:00").getTime()); if (x < m.l || x > W - m.r) return;
    el("text", { x, y: H - 8, "text-anchor": "middle", class: "lbl" }, svg).textContent = d.slice(5, 7) + "/" + d.slice(2, 4); });
  el("path", { d: ys.map((v, i) => (i ? "L" : "M") + X(t[i]).toFixed(1) + "," + Y(v).toFixed(1)).join(""), fill: "none", stroke: "var(--s1)", "stroke-width": 2, "stroke-linejoin": "round" }, svg);
  const cross = el("line", { y1: m.t, y2: m.t + ih, stroke: "var(--ink-faint)", "stroke-width": 1, opacity: 0 }, svg);
  const dot = el("circle", { r: 4.5, fill: "var(--s1)", stroke: "var(--surface)", "stroke-width": 2, opacity: 0 }, svg);
  const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, class: "hit" }, svg);
  hit.addEventListener("mousemove", ev => { const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
    let i = 0, best = 1e18; t.forEach((v, k) => { const d = Math.abs(X(v) - px); if (d < best) { best = d; i = k; } });
    cross.setAttribute("x1", X(t[i])); cross.setAttribute("x2", X(t[i])); cross.setAttribute("opacity", 1);
    dot.setAttribute("cx", X(t[i])); dot.setAttribute("cy", Y(ys[i])); dot.setAttribute("opacity", 1);
    showTip(ev, `semana de ${new Date(t[i]).toLocaleDateString("pt-BR")}<br><b>${(o.ft || nf)(ys[i])}</b>`); });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("opacity", 0); dot.setAttribute("opacity", 0); hideTip(); });
}

/* faixa p25–p75 com mediana */
function faixasPreco(id, title, sub, rows) {
  const box = document.getElementById(id); head(box, title, sub);
  const W = Math.max(300, box.clientWidth - 28), rowH = 24, m = { l: 140, r: 16, t: 4, b: 26 }, H = m.t + m.b + rowH * rows.length;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title }, box);
  const max = 0.06, iw = W - m.l - m.r, X = v => m.l + iw * Math.min(v, max) / max;
  for (let k = 0; k <= 6; k++) { const v = max * k / 6, x = X(v);
    el("line", { x1: x, x2: x, y1: m.t, y2: H - m.b, class: k ? "gridline" : "baseline" }, svg);
    el("text", { x, y: H - 8, "text-anchor": "middle", class: "lbl" }, svg).textContent = nf(v, 2); }
  rows.forEach((d, i) => { const y = m.t + i * rowH + rowH / 2;
    el("text", { x: m.l - 8, y: y + 4, "text-anchor": "end", class: "lbl" }, svg).textContent = d.grupo;
    el("line", { x1: X(d.p25), x2: X(d.p75), y1: y, y2: y, stroke: "var(--s1-soft)", "stroke-width": 8, "stroke-linecap": "round" }, svg);
    el("circle", { cx: X(d.p50), cy: y, r: 5, fill: "var(--s1)", stroke: "var(--surface)", "stroke-width": 2 }, svg);
    const hit = el("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH, class: "hit" }, svg);
    hit.addEventListener("mousemove", ev => showTip(ev, `${d.grupo}<br>p25 <b>${nf(d.p25, 4)}</b> · mediana <b>${nf(d.p50, 4)}</b> · p75 <b>${nf(d.p75, 4)}</b><br>${mil(d.n)} itens vendidos`));
    hit.addEventListener("mouseleave", hideTip); });
}

function render() {
  document.querySelectorAll(".chart").forEach(c => c.innerHTML = "");
  const S = R.semanal, fx = [
    { de: "2018-09-17", ate: "2019-12-27", cor: "var(--band-train)", rot: "treino" },
    { de: "2019-12-27", ate: "2020-03-26", cor: "var(--band-val)", rot: "val" },
    { de: "2020-03-26", ate: "2020-09-21", cor: "var(--band-test)", rot: "teste" }];
  const leg = `<span><i style="background:var(--band-train);border:1px solid var(--s1)"></i>treino</span><span><i style="background:var(--band-val);border:1px solid var(--s2)"></i>validação</span><span><i style="background:var(--band-test);border:1px solid var(--s3)"></i>teste</span>`;
  linha("c-semanal", "Itens vendidos por semana", "A primeira e a última semana são parciais.", S.semana, S.itens, { faixas: fx, legenda: leg, ft: v => nf(v) + " itens" });
  linha("c-online", "Participação do canal online", "Fração dos itens vendidos online em cada semana; a faixa vermelha marca as lojas fechadas.", S.semana, S.share_online,
    { ymax: 1, fy: v => pct(v, 0), ft: v => pct(v), faixas: [...fx, { de: "2020-03-23", ate: "2020-05-04", cor: "var(--covid)", rot: "lojas fechadas" }], legenda: leg });

  colunas("c-itens-cliente", "Itens por cliente", "Clientes por faixa de itens comprados em 2 anos.", histItens(R.hist_itens_cliente, faixa), { rot: true, ft: v => nf(v) + " clientes" });
  colunas("c-cestas-cliente", "Cestas por cliente", "33% compraram em um único dia.", histItens(R.hist_cestas_cliente, faixa), { rot: true, ft: v => nf(v) + " clientes" });
  colunas("c-itens-cesta", "Itens por cesta", "Cesta = cliente + dia.", histItens(R.hist_itens_cesta, faixa), { rot: true, ft: v => nf(v) + " cestas" });
  colunas("c-gap", "Dias até a próxima cesta", "Intervalo entre cestas consecutivas do mesmo cliente.", histItens(R.hist_gap_dias, faixa), { rot: true, ft: v => nf(v) + " intervalos", marca: { i: 6, t: "N = 90" } });

  const ig = Object.entries(R.share_index_group), tg = ig.reduce((a, [, v]) => a + v, 0);
  barrasH("c-index", "Itens vendidos por linha (index group)", null, ig.map(([l, v]) => ({ l, v: v / tg, extra: mil(v) + " itens" })), { ml: 104 });
  const cc = R.concentracao_artigos;
  colunas("c-concentracao", "Concentração das vendas no catálogo", "Fração das vendas coberta pelos N artigos mais vendidos.",
    cc.top_n.slice(0, 9).map((n, i) => ({ v: cc.share[i], l: n >= 1000 ? nf(n / 1000) + " mil" : nf(n), tip: `top ${nf(n)} artigos` })),
    { fy: v => pct(v, 0), ft: v => pct(v) + " das vendas", rot: true, marca: { i: 4, t: "top-2000 (token)", centro: true } });
  faixasPreco("c-preco", "Preço por grupo de produto", "Faixa entre p25 e p75; o ponto é a mediana. Preço normalizado pela H&M.", R.preco_por_grupo.slice(0, 9));

  colunas("c-idade", "Idade dos clientes", "1,2% sem idade informada.", histItens(R.hist_idade, (a, b, l) => l ? `${a} a ${b - 1} anos` : `${a}–${b - 1}`), { rot: true, ft: v => nf(v) + " clientes" });

  const nc = Object.entries(R.rotulos.next_category);
  barrasH("c-nextcat", "Próxima categoria (alvo)", "Grupo dominante da próxima cesta; 10 classes.", nc.map(([l, v]) => ({ l, v })), { ml: 128 });
  const lb = Object.entries(R.rotulos.ltv_buckets);
  colunas("c-ltv", "LTV em 90 dias (alvo)", "Faixa do gasto nos 90 dias seguintes à cesta.", lb.map(([k, v]) => ({ v, l: "faixa " + parseInt(k), tip: "faixa " + parseInt(k) + (parseInt(k) === 2 ? " (inclui gasto zero)" : "") })),
    { fy: v => pct(v, 0), ft: v => pct(v) + " das cestas" });
  colunas("c-tokens", "Tokens por cliente", "Comprimento da sequência completa (seq_full).", histItens(R.hist_tokens_cliente, (a, b, l) => l ? `${nf(a)} a ${nf(b)} tokens` : `${a >= 1000 ? nf(a / 1000) + "k" : a}–${b >= 1000 ? nf(b / 1000) + "k" : b}`),
    { rot: true, ft: v => nf(v) + " clientes", marca: { i: 5, t: "1.024" } });
}

function tabelas() {
  const C = R.comparacao_bases, h = C.hm_clientes;
  document.getElementById("tiles").innerHTML = [
    [mil(E.transacoes), "itens vendidos"], [mil(E.clientes_com_compra), "clientes com compra"], [mil(E.cestas), "cestas (cliente + dia)"],
    [mil(E.artigos_vendidos), "artigos vendidos"], [nf(E.dias), "dias (set/2018 a set/2020)"], [mil(h.tokens), "tokens serializados"]]
    .map(([b, s]) => `<div class="tile"><b>${b}</b><span>${s}</span></div>`).join("");
  document.getElementById("t-estrutura").innerHTML = [
    ["transactions", E.transacoes, "t_dat, customer_id, article_id, price, sales_channel_id", "Cada linha é um evento (item). Data → splits, recência e mês; preço → faixa; canal → token."],
    ["articles", E.artigos_cadastro, "25 colunas: tipo, grupo, cor, departamento, índice, seção, linha, descrição", "Índice, grupo, tipo, cor e linha viram tokens do item; artigo vira top-2000 + OUTROS."],
    ["customers", E.clientes_cadastro, "FN, Active, club_member_status, fashion_news_frequency, age, postal_code", "Só na fusão e no GBDT (não tokenizado)."]]
    .map(r => `<tr><td><code>${r[0]}</code></td><td class="num">${nf(r[1])}</td><td>${r[2]}</td><td>${r[3]}</td></tr>`).join("");
  const cd = R.cadastro, clube = cd.club_member_status, nl = cd.fashion_news_frequency, tc = Object.values(clube).reduce((a, b) => a + b, 0);
  document.getElementById("t-cadastro").innerHTML = [
    ["Idade", "16 a 99 anos; mediana 32", pct(cd.idade_nula)],
    ["Clube", Object.entries(clube).filter(([k]) => k !== "None").map(([k, v]) => `${k} ${pct(v / tc, 1)}`).join(" · "), pct((clube.None || 0) / tc)],
    ["Newsletter", Object.entries(nl).filter(([k]) => k !== "None").map(([k, v]) => `${k} ${pct(v / tc, 1)}`).join(" · "), pct((nl.None || 0) / tc)],
    ["FN / Active", "1 ou nulo (nulo = não)", pct(cd.FN_nulo, 0)],
    ["CEP", `${mil(cd.cep_distintos)} valores; mediana de ${nf(cd.cep_mediana_clientes)} clientes por CEP`, "0%"]]
    .map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td class="num">${r[2]}</td></tr>`).join("");
  const RT = R.rotulos, cls = { train: "p-train", val: "p-val", test: "p-test" }, nome = { train: "treino", val: "validação", test: "teste" };
  document.getElementById("t-rotulos").innerHTML = ["train", "val", "test"].map(s =>
    `<tr><td><span class="pill ${cls[s]}">${nome[s]}</span></td><td class="num">${nf(RT.cestas_por_split[s])}</td><td class="num">${pct(RT.churn_taxa[s])}</td><td class="num">${pct(RT.online_taxa[s])}</td></tr>`).join("");
  document.getElementById("t-contexto").innerHTML = [[256, .721, .514], [512, .860, .707], [1024, .953, .873], [2048, .990, .964], [4096, .999, .994]]
    .map(r => `<tr${r[0] === 1024 ? ' style="font-weight:600"' : ""}><td class="num">${nf(r[0])}</td><td class="num">${pct(r[1])}</td><td class="num">${pct(r[2])}</td></tr>`).join("");
  const S = C.synthetic_agro, O = C.olist_sellers, lin = (rot, f) => `<tr><td>${rot}</td><td class="num">${f(S)}</td><td class="num">${f(O)}</td><td class="num">${f(h)}</td></tr>`;
  document.getElementById("t-comp").innerHTML = [
    lin("Entidade", x => x === S ? "cliente" : x === O ? "vendedor" : "cliente"),
    lin("Eventos", x => nf(x.eventos)), lin("Entidades", x => nf(x.entidades)),
    lin("Eventos por entidade (mediana)", x => nf(x.eventos_por_entidade_mediana)),
    lin("Período", x => x.periodo[0].slice(0, 7) + " a " + x.periodo[1].slice(0, 7)),
    lin("Vocabulário (tokens)", x => nf(x.vocab)), lin("Tokens totais", x => nf(x.tokens)), lin("Tokens de treino", x => nf(x.tokens_treino)),
    lin("Dados", x => x === S ? "sintéticos" : "reais")].join("");
  document.getElementById("p-clientes").innerHTML = `O cliente mediano comprou ${nf(E.itens_por_cliente.p50)} itens em ${nf(E.cestas_por_cliente.p50)} cestas;
    o p99 chega a ${nf(E.itens_por_cliente.p99)} itens. ${pct(E.frac_clientes_1_cesta, 0)} dos clientes compraram em um único dia, e a metade
    das recompras acontece em até ${nf(E.gap_entre_cestas_dias.p50)} dias.`;
}

tabelas(); render();
let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(render, 150); });
</script>
"""


def main():
    with open(os.path.join(EDA, "eda.json"), encoding="utf-8") as f:
        eda = json.load(f)
    with open(os.path.join(EDA, "eda_relatorio.json"), encoding="utf-8") as f:
        rel = json.load(f)
    rel["rotulos"].pop("relatorio", None)
    html = HTML.replace("__EDA__", json.dumps(eda, ensure_ascii=False)).replace("__REL__", json.dumps(rel, ensure_ascii=False))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Gravado {OUT} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
