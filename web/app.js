const state = {
  data: null,
  holdout: null,
  extreme: null,
  signal: null,
  mode: 'experimental',
  year: 'all',
  page: 1,
  pageSize: 40,
  challenge: null,
  revealed: false,
  strictTicket: null,
  coins: Number(localStorage.getItem('ssq_coins') || 0),
  streak: Number(localStorage.getItem('ssq_streak') || 0),
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function pad2(value) {
  return String(value).padStart(2, '0');
}

function pct(value, digits = 1) {
  return `${(Number(value || 0) * 100).toFixed(digits)}%`;
}

function fixed(value, digits = 3) {
  return Number(value || 0).toFixed(digits);
}

function signed(value, digits = 3) {
  const number = Number(value || 0);
  return `${number >= 0 ? '+' : ''}${number.toFixed(digits)}`;
}

function tierName(tier) {
  const names = {
    1: '一等奖',
    2: '二等奖',
    3: '三等奖',
    4: '四等奖',
    5: '五等奖',
    6: '六等奖',
  };
  return names[tier] || '未中奖';
}

function ballHTML(number, color, extra = '') {
  return `<span class="ball ${color} ${extra}">${pad2(number)}</span>`;
}

function ballsHTML(numbers, color, options = {}) {
  const hitSet = new Set(options.hits || []);
  const mutedSet = new Set(options.muted || []);
  const extraClass = options.small ? 'small' : options.tiny ? 'tiny' : '';
  return numbers.map((number) => {
    const classes = [extraClass];
    if (hitSet.has(number)) classes.push('hit');
    if (mutedSet.has(number) && !hitSet.has(number)) classes.push('muted');
    return ballHTML(number, color, classes.filter(Boolean).join(' '));
  }).join('');
}

function singleTicketHTML(reds, blue, options = {}) {
  return [
    ballsHTML([...reds].sort((a, b) => a - b), 'red', options),
    '<span class="ball-sep">+</span>',
    ballsHTML([blue], 'blue', options),
  ].join('');
}

function setBalls(selector, numbers, color, options = {}) {
  const element = $(selector);
  if (element) element.innerHTML = ballsHTML(numbers, color, options);
}

function setSingle(selector, reds, blue, options = {}) {
  const element = $(selector);
  if (element) element.innerHTML = singleTicketHTML(reds, blue, options);
}

function randomTicket() {
  const pool = Array.from({ length: 33 }, (_, index) => index + 1);
  const reds = [];
  while (reds.length < 6) {
    const index = Math.floor(Math.random() * pool.length);
    reds.push(pool.splice(index, 1)[0]);
  }
  reds.sort((a, b) => a - b);
  return { reds, blue: 1 + Math.floor(Math.random() * 16) };
}

function modeData() {
  if (!state.data) return { cases: [], rolling: [], summary: {} };
  if (state.mode === 'strict') {
    return {
      cases: state.data.cases || [],
      rolling: state.data.rolling || [],
      summary: state.data.champion.summary || {},
    };
  }
  return {
    cases: state.data.experimental.cases || [],
    rolling: state.data.experimental.rolling || [],
    summary: state.data.experimental.test_summary || {},
  };
}

function filteredCases() {
  const { cases } = modeData();
  if (state.year === 'all') return cases;
  return cases.filter((item) => String(item.date).startsWith(state.year));
}

function filteredRolling() {
  const { rolling } = modeData();
  if (state.year === 'all') return rolling;
  return rolling.filter((item) => String(item.date).startsWith(state.year));
}

function switchView(name) {
  $$('.tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.view === name));
  $$('.view').forEach((view) => view.classList.toggle('active', view.id === `view-${name}`));
  if (name === 'backtest') requestAnimationFrame(() => renderBacktest());
  if (name === 'holdout2015') requestAnimationFrame(() => renderHoldout());
  if (name === 'extreme') requestAnimationFrame(() => renderExtreme());
  if (name === 'models') requestAnimationFrame(() => renderModels());
}

function renderMeta() {
  const { meta } = state.data;
  $('#dataBadge').textContent = `${meta.rows} 期 · ${meta.first_issue} → ${meta.last_issue}`;
  $('#pillRows').textContent = `${meta.rows} 期官方数据`;
  $('#pillCrosscheck').textContent = meta.pagination_crosscheck?.ok ? '100/50 双分页复核' : '单分页采集';
  $('#pillNoLookahead').textContent = meta.no_lookahead ? '无前视盲测' : '检查回测设置';
  $('#pillChampion').textContent = `严格模式：${meta.champion_label}`;
  $('#pillChampion').className = meta.champion_key === 'random' ? 'pill warn' : 'pill good';
  const orbitNumber = $('.orbit-core span:not(.red-dot)');
  if (orbitNumber) orbitNumber.textContent = String(meta.test_n);
}

function renderStats() {
  const { meta, baseline, champion } = state.data;
  const strict = champion.summary;
  const experimental = state.data.experimental.test_summary;
  const cards = [
    {
      label: '官方真实数据',
      value: `${meta.rows}`,
      sub: `${meta.first_issue} → ${meta.last_issue}`,
      className: '',
    },
    {
      label: '5 年盲测',
      value: `${meta.test_n}`,
      sub: `${meta.test_start} → ${meta.test_end}`,
      className: '',
    },
    {
      label: '单注头奖',
      value: `${experimental.exact_jackpot_count} 次`,
      sub: `严格基线 ${strict.exact_jackpot_count} 次 · 概率 ${pct(baseline.exact_jackpot_rate, 5)}`,
      className: 'bad',
    },
    {
      label: 'Top-10 红球覆盖',
      value: pct(experimental.top10_coverage_rate),
      sub: `理论覆盖 ${pct(baseline.top10_red_coverage_rate)} · 不是中奖率`,
      className: 'good',
    },
    {
      label: '单注任意奖级',
      value: pct(experimental.any_prize_rate),
      sub: `严格基线 ${pct(strict.any_prize_rate)} · 理论 ${pct(baseline.any_prize_rate)}`,
      className: 'warn',
    },
    {
      label: '模拟回报',
      value: pct(experimental.roi),
      sub: `2 元/期，按官方奖级金额`,
      className: 'bad',
    },
  ];
  $('#statsGrid').innerHTML = cards.map((card) => `
    <article class="stat-card">
      <div class="stat-label">${card.label}</div>
      <div class="stat-value ${card.className}">${card.value}</div>
      <div class="stat-sub">${card.sub}</div>
    </article>
  `).join('');
}

function renderPredictions() {
  const champion = state.data.champion;
  const strict = champion.latest_prediction;
  const experimental = state.data.experimental.latest_prediction;
  const strictChip = $('#strictChip');
  const strictNote = $('#strictNote');

  if (champion.key === 'random') {
    strictChip.textContent = '无显著预测力';
    strictChip.className = 'status-chip bad';
    strictNote.textContent = '没有候选模型通过验证段显著性门槛。严格模式不伪造排序，以下为可复现的随机基线；它代表“没有可靠信号”，不是能力不足的占位符。';
  } else {
    strictChip.textContent = '验证通过';
    strictChip.className = 'status-chip';
    strictNote.textContent = `${champion.label} 在验证段相对随机基线达到预设显著性门槛；测试段结果仍需独立观察。`;
  }

  $('#strictIssueLabel').textContent = `${state.data.meta.next_issue} · 单注候选`;
  $('#expIssueLabel').textContent = `${state.data.meta.next_issue} · 单注候选`;

  state.strictTicket = strict.single_ticket;
  setSingle('#strictSingle', strict.single_ticket.reds, strict.single_ticket.blue);
  setBalls('#strictPool', strict.top10_reds, 'red', { small: true });

  const ensemble = state.data.models.find((model) => model.key === 'ensemble');
  const significant = ensemble?.validation_vs_random?.significant_improvement;
  $('#expChip').textContent = significant ? '验证通过' : '未通过门槛';
  $('#expChip').className = significant ? 'status-chip' : 'status-chip warn';
  $('#expNote').textContent = `多因子集成在 5 年测试段的 NDCG@10 为 ${fixed(state.data.experimental.test_summary.mean_ndcg_at_10)}，随机基线为 ${fixed(state.data.models.find((model) => model.key === 'random').test.mean_ndcg_at_10)}；差异没有统计显著性。`;
  setSingle('#expSingle', experimental.single_ticket.reds, experimental.single_ticket.blue);
  setBalls('#expPool', experimental.top10_reds, 'red', { small: true });
  const expPool = $('#expPool');
  expPool.insertAdjacentHTML('beforeend', '<span class="ball-sep">+</span>');
  expPool.insertAdjacentHTML('beforeend', ballsHTML(experimental.top4_blues, 'blue', { small: true }));
}

function pointsFor(caseItem) {
  return caseItem.red_hits * 10
    + (caseItem.blue_hit ? 20 : 0)
    + (caseItem.tier ? 100 : 0)
    + (caseItem.top10_coverage ? 50 : 0);
}

function saveScore() {
  localStorage.setItem('ssq_coins', String(state.coins));
  localStorage.setItem('ssq_streak', String(state.streak));
  $('#coins').textContent = String(state.coins);
  $('#streak').textContent = String(state.streak);
}

function newChallenge() {
  const cases = state.data.experimental.cases || [];
  if (!cases.length) return;
  let next = cases[Math.floor(Math.random() * cases.length)];
  if (state.challenge && cases.length > 1) {
    while (next.issue === state.challenge.issue) {
      next = cases[Math.floor(Math.random() * cases.length)];
    }
  }
  state.challenge = next;
  state.revealed = false;
  renderChallenge();
}

function renderChallenge() {
  const container = $('#challenge');
  if (!state.challenge) {
    container.innerHTML = '<div class="challenge-empty">正在准备盲测案例…</div>';
    return;
  }
  const item = state.challenge;
  const actualRedSet = new Set(item.actual_reds);
  const top10Set = new Set(item.top10_reds);
  const actualHTML = state.revealed
    ? `${ballsHTML(item.actual_reds, 'red', { hits: item.top10_reds, small: true })}<span class="ball-sep">+</span>${ballsHTML([item.actual_blue], 'blue', { hits: item.top4_blues, small: true })}`
    : '<span class="challenge-hidden">● ● ● ● ● ●  +  ●</span>';
  const tierText = item.tier ? tierName(item.tier) : '未中奖';
  const resultHTML = state.revealed
    ? `<div class="challenge-result ${item.top10_coverage ? '' : 'warn'}">
        Top-10 覆盖：<b>${item.top10_hits > 0 ? `命中 ${item.top10_hits} 个红球` : '未覆盖'}</b><br>
        单注红中 <b>${item.red_hits}</b> 个 · 蓝球 <b>${item.blue_hit ? '命中' : '未中'}</b> · <b>${tierText}</b><br>
        本关积分 <b>+${pointsFor(item)}</b>
      </div>`
    : '<div class="challenge-result warn">揭晓后才会显示真实结果。系统候选已经固定，不能根据答案重选。</div>';

  container.innerHTML = `
    <div class="challenge-card">
      <div class="challenge-main">
        <div class="challenge-title">
          <strong>${item.issue}</strong>
          <span>${item.date} · 5 年测试段案例</span>
        </div>
        <div class="block-label">系统单注</div>
        <div class="ball-line">${singleTicketHTML(item.red_ranking.slice(0, 6), item.blue_ranking[0], { small: true })}</div>
        <div class="block-label" style="margin-top:14px">Top-10 红球池</div>
        <div class="ball-line pool">${ballsHTML(item.top10_reds, 'red', { hits: state.revealed ? item.actual_reds : [], small: true })}</div>
        <div class="challenge-actions">
          ${state.revealed
            ? '<button class="btn" data-action="next">下一关</button>'
            : '<button class="btn" data-action="reveal">揭晓开奖</button><button class="btn ghost" data-action="next">换一期</button>'}
        </div>
      </div>
      <div class="challenge-side">
        <div class="block-label">真实开奖</div>
        <div class="ball-line">${actualHTML}</div>
        <div class="block-label" style="margin-top:14px">实际 Top-10 覆盖</div>
        <div class="ball-line">${state.revealed
          ? ballsHTML(item.actual_reds, 'red', { hits: item.top10_reds, small: true })
          : '<span class="challenge-hidden">等待揭晓</span>'}</div>
        ${resultHTML}
      </div>
    </div>
  `;
}

function revealChallenge() {
  if (!state.challenge) return;
  if (!state.revealed) {
    state.revealed = true;
    const points = pointsFor(state.challenge);
    state.coins += points;
    state.streak = state.challenge.top10_coverage ? state.streak + 1 : 0;
    saveScore();
  }
  renderChallenge();
}

function renderBacktest() {
  if (!state.data) return;
  const { summary } = modeData();
  $('#backtestStats').innerHTML = [
    ['Top-10 覆盖', pct(summary.top10_coverage_rate)],
    ['单注任意奖级', pct(summary.any_prize_rate)],
    ['平均红球命中', fixed(summary.avg_red_hits, 3)],
    ['单注头奖', `${summary.exact_jackpot_count} 次`],
    ['模拟 ROI', pct(summary.roi)],
  ].map(([label, value]) => `<div class="mini-stat"><span>${label}</span><b>${value}</b></div>`).join('');
  $('#coverageValue').textContent = pct(summary.top10_coverage_rate);
  $('#prizeValue').textContent = pct(summary.any_prize_rate);
  renderYearFilter();
  renderCaseTable();
  requestAnimationFrame(() => {
    drawRateChart($('#coverageChart'), filteredRolling().map((item) => item.top10_coverage_rate), {
      minY: 0.8,
      maxY: 1.0,
      baseline: state.data.baseline.top10_red_coverage_rate,
      color: '#f5c451',
    });
    drawRateChart($('#prizeChart'), filteredRolling().map((item) => item.any_prize_rate), {
      minY: 0,
      maxY: 0.2,
      baseline: state.data.baseline.any_prize_rate,
      color: '#54a8ff',
    });
  });
}

function renderHoldout() {
  if (!state.holdout) {
    $('#holdoutStats').innerHTML = '';
    $('#holdoutVerdictTitle').textContent = '2015 盲测结果未生成';
    $('#holdoutVerdictText').textContent = '请先运行 make holdout，再重新加载页面。';
    $('#holdoutTableBody').innerHTML = '';
    $('#holdoutCount').textContent = '—';
    return;
  }

  const report = state.holdout;
  const summary = report.random10.summaries.ensemble;
  const baseline = report.random10.summaries.random;
  const full = report.full_2015.summaries.ensemble;
  const lower = summary.top10_coverage_lower_bound_95;
  const upper = summary.top10_coverage_wilson_95[1];

  $('#holdoutStats').innerHTML = [
    ['随机 10 期', `${summary.n} 期`],
    ['Top-10 覆盖', `${summary.top10_coverage_count}/${summary.n}`],
    ['覆盖 95% 下界', pct(lower, 2)],
    ['单注任何奖级', `${summary.any_prize_count}/${summary.n}`],
    ['单注头奖', `${summary.exact_jackpot_count} 次`],
  ].map(([label, value]) => `<div class="mini-stat"><span>${label}</span><b>${value}</b></div>`).join('');

  $('#holdoutVerdictTitle').textContent = summary.meets_90pct_lower_bound
    ? `10 期 Top-10 覆盖 ${summary.top10_coverage_count}/${summary.n}，95% 下界达到 90%`
    : `10 期 Top-10 覆盖 ${summary.top10_coverage_count}/${summary.n}，但 95% 下界只有 ${pct(lower, 2)}`;
  $('#holdoutVerdictText').textContent = `覆盖区间 ${pct(lower, 2)}–${pct(upper, 2)}；单注任何奖级 ${summary.any_prize_count}/${summary.n}，单注头奖 ${summary.exact_jackpot_count} 次。2015 全年 Top-10 覆盖 ${full.top10_coverage_count}/${full.n}（${pct(full.top10_coverage_rate, 2)}，区间 ${pct(full.top10_coverage_wilson_95[0], 2)}–${pct(full.top10_coverage_wilson_95[1], 2)}），随机基线 ${baseline.top10_coverage_count}/${baseline.n}。因此只能说候选池覆盖达到样本值，不能认证未来 90% 下限，更不能推出单注预测能力。`;

  const issues = report.meta.selected_issues;
  $('#holdoutCount').textContent = `种子 ${report.meta.seed} · ${issues.length} 期`;
  $('#holdoutTableBody').innerHTML = issues.map((issue) => {
    const item = report.random10.cases[issue].ensemble;
    const tier = item.tier ? tierName(item.tier) : '未中奖';
    return `
      <tr>
        <td class="issue-cell">${item.issue}</td>
        <td class="date-cell">${item.date}</td>
        <td><div class="tiny-balls">${ballsHTML(item.actual_reds, 'red', { hits: item.top10_reds, tiny: true })}<span class="ball-sep">+</span>${ballsHTML([item.actual_blue], 'blue', { tiny: true })}</div></td>
        <td><div class="tiny-balls">${ballsHTML(item.top10_reds, 'red', { hits: item.actual_reds, tiny: true })}</div></td>
        <td><span class="tag ${item.top10_coverage ? 'good' : 'bad'}">${item.top10_coverage ? `命中 ${item.top10_hits}` : '未覆盖'}</span></td>
        <td><div class="tiny-balls">${singleTicketHTML(item.single_reds, item.single_blue, { tiny: true })}</div></td>
        <td><span class="tag ${item.red_hits ? 'neutral' : 'bad'}">${item.red_hits}</span> / <span class="tag ${item.blue_hit ? 'good' : 'neutral'}">${item.blue_hit ? '中' : '—'}</span></td>
        <td><span class="tag ${item.tier ? 'good' : 'neutral'}">${tier}</span></td>
      </tr>
    `;
  }).join('');
}

function renderSignal() {
  const stats = $('#signalStats');
  const count = $('#signalCount');
  const conclusion = $('#signalConclusion');
  if (!stats || !count || !conclusion) return;

  if (!state.signal) {
    stats.innerHTML = '';
    count.textContent = '未生成';
    conclusion.textContent = '请先运行 make signal 生成信号审计，再重新加载页面。';
    return;
  }

  const report = state.signal;
  const frequency = report.frequency;
  const repeats = report.repeats;
  const transitions = report.transitions;
  const annual = report.annual_frequency_check;
  const folds = report.fold_checks || [];
  const significantFolds = folds.filter((fold) => fold.ndcg_ci95_low > 0).length;
  const strict = report.decision.action === 'keep_random_strict';

  stats.innerHTML = [
    ['红球频率 p', fixed(frequency.red_p_value, 4)],
    ['蓝球频率 p', fixed(frequency.blue_p_value, 4)],
    ['邻期重号 p', fixed(repeats.p_two_sided, 4)],
    ['前后期转移 p', fixed(transitions.p_two_sided, 4)],
    ['年度频率胜负', `${annual.frequency_wins}:${annual.frequency_losses}`],
    ['显著折数', `${significantFolds}/${folds.length}`],
  ].map(([label, value]) => `<div class="mini-stat"><span>${label}</span><b>${value}</b></div>`).join('');

  count.textContent = `${report.meta.data_rows} 期 · ${report.meta.eval_start} → ${report.meta.eval_end}`;
  conclusion.textContent = `${report.conclusion} ${strict ? '严格模式继续保留随机基线。' : '审计要求人工复核，暂不自动提升模型。'}`;
}

function renderExtreme() {
  if (!state.extreme) {
    $('#extremeStats').innerHTML = '';
    $('#extremeVerdictTitle').textContent = '99% 极限审计未生成';
    $('#extremeVerdictText').textContent = '请先运行 make extreme，再重新加载页面。';
    $('#extremeTableBody').innerHTML = '';
    $('#extremeCount').textContent = '—';
    return;
  }

  const report = state.extreme;
  const baseline = report.baselines;
  const requirement = report.single_ticket_99pct;
  const observed = report.observed_ceiling;
  const top3 = requirement.top3_or_better;
  const jackpot = requirement.jackpot;

  $('#extremeStats').innerHTML = [
    ['单注前三等奖', `1 / ${Math.round(baseline.top3_prize_one_in).toLocaleString('zh-CN')}`],
    ['单注头奖', `1 / ${Math.round(baseline.jackpot_one_in).toLocaleString('zh-CN')}`],
    ['20 年最优前三等奖', pct(observed.best_top3_rate, 4)],
    ['99% 前三等奖票数', top3.random_distinct_tickets.toLocaleString('zh-CN')],
    ['票面成本', `¥${top3.cost_yuan.toLocaleString('zh-CN')}`],
  ].map(([label, value]) => `<div class="mini-stat"><span>${label}</span><b>${value}</b></div>`).join('');

  $('#extremeVerdictTitle').textContent = `纯预测单注前三等奖率 ${pct(baseline.top3_prize_rate, 4)}，20 年最优只有 ${pct(observed.best_top3_rate, 4)}`;
  $('#extremeVerdictText').textContent = `${report.meta.eval_draws} 期 walk-forward 中，单注头奖 ${observed.best_jackpot_count} 次；最好模型前三等奖 ${pct(observed.best_top3_rate, 4)}，最好 Top-10+4 候选池 ${pct(observed.best_top10_top4_candidate_rate, 4)}。若要把“至少一次前三等奖”的概率做到 99%，随机不同票约需 ${top3.random_distinct_tickets.toLocaleString('zh-CN')} 张、成本 ¥${top3.cost_yuan.toLocaleString('zh-CN')}；这不是单注预测。`;

  $('#extremeCount').textContent = `${report.meta.eval_start} → ${report.meta.eval_end} · ${report.meta.eval_draws} 期`;
  $('#extremeTableBody').innerHTML = report.models.map((model) => {
    const top3ci = model.top3_wilson_95 || [0, 0];
    const candidateci = model.top10_top4_candidate_wilson_95 || [0, 0];
    return `
      <tr>
        <td>${model.label}</td>
        <td>${model.n}</td>
        <td><span class="tag ${model.jackpot_count ? 'good' : 'neutral'}">${model.jackpot_count} 次</span></td>
        <td><span class="tag ${model.top3_count ? 'warn' : 'neutral'}">${model.top3_count} 次</span></td>
        <td>${pct(model.top3_rate, 4)} <span class="date-cell">[${pct(top3ci[0], 3)}, ${pct(top3ci[1], 3)}]</span></td>
        <td>${model.top10_top4_candidate_count} 次</td>
        <td>${pct(model.top10_top4_candidate_rate, 4)} <span class="date-cell">[${pct(candidateci[0], 3)}, ${pct(candidateci[1], 3)}]</span></td>
      </tr>
    `;
  }).join('');
}

function renderYearFilter() {
  const select = $('#yearFilter');
  if (select.dataset.ready === 'true') return;
  const years = [...new Set(state.data.cases.map((item) => item.date.slice(0, 4)))].sort();
  select.innerHTML = '<option value="all">全部年份</option>' + years.map((year) => `<option value="${year}">${year}</option>`).join('');
  select.dataset.ready = 'true';
}

function renderCaseTable() {
  const cases = filteredCases();
  const totalPages = Math.max(1, Math.ceil(cases.length / state.pageSize));
  state.page = Math.min(Math.max(1, state.page), totalPages);
  const start = (state.page - 1) * state.pageSize;
  const pageCases = cases.slice(start, start + state.pageSize);
  $('#caseTableBody').innerHTML = pageCases.map((item) => {
    const actualSet = new Set(item.actual_reds);
    const top10Set = new Set(item.top10_reds);
    const cover = item.top10_coverage;
    const tier = item.tier ? tierName(item.tier) : '—';
    return `
      <tr>
        <td class="issue-cell">${item.issue}</td>
        <td class="date-cell">${item.date}</td>
        <td><div class="tiny-balls">${ballsHTML(item.actual_reds, 'red', { hits: item.top10_reds, tiny: true })}<span class="ball-sep">+</span>${ballsHTML([item.actual_blue], 'blue', { tiny: true })}</div></td>
        <td><div class="tiny-balls">${ballsHTML(item.top10_reds, 'red', { hits: item.actual_reds, tiny: true })}</div></td>
        <td><span class="tag ${cover ? 'good' : 'bad'}">${cover ? `命中 ${item.top10_hits}` : '未覆盖'}</span></td>
        <td><span class="tag ${item.red_hits >= 3 ? 'good' : item.red_hits > 0 ? 'neutral' : 'bad'}">${item.red_hits}</span></td>
        <td><span class="tag ${item.blue_hit ? 'good' : 'neutral'}">${item.blue_hit ? '中' : '—'}</span></td>
        <td><span class="tag ${item.tier ? 'good' : 'neutral'}">${tier}</span></td>
      </tr>
    `;
  }).join('');
  $('#tableCount').textContent = `共 ${cases.length} 期`;
  $('#pageInfo').textContent = `第 ${state.page} / ${totalPages} 页`;
  $('#prevPage').disabled = state.page <= 1;
  $('#nextPage').disabled = state.page >= totalPages;
}

function prepareCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, rect.width);
  const height = Number(canvas.getAttribute('height')) || 220;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

function drawRateChart(canvas, values, options) {
  if (!canvas || !values.length) return;
  const { ctx, width, height } = prepareCanvas(canvas);
  const pad = { left: 38, right: 12, top: 14, bottom: 22 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const minY = options.minY;
  const maxY = options.maxY;
  const y = (value) => pad.top + (maxY - value) / (maxY - minY) * plotH;
  const x = (index) => pad.left + (values.length <= 1 ? 0 : index / (values.length - 1) * plotW);

  ctx.clearRect(0, 0, width, height);
  ctx.font = '10px -apple-system, BlinkMacSystemFont, sans-serif';
  ctx.lineWidth = 1;
  for (let step = 0; step <= 4; step += 1) {
    const value = minY + (maxY - minY) * step / 4;
    const yPos = y(value);
    ctx.strokeStyle = 'rgba(146, 180, 220, 0.09)';
    ctx.beginPath();
    ctx.moveTo(pad.left, yPos);
    ctx.lineTo(width - pad.right, yPos);
    ctx.stroke();
    ctx.fillStyle = '#65809e';
    ctx.fillText(pct(value, 0), 2, yPos + 3);
  }

  if (typeof options.baseline === 'number') {
    const baselineY = y(options.baseline);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.24)';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(pad.left, baselineY);
    ctx.lineTo(width - pad.right, baselineY);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const gradient = ctx.createLinearGradient(0, pad.top, 0, height - pad.bottom);
  gradient.addColorStop(0, `${options.color}44`);
  gradient.addColorStop(1, `${options.color}00`);
  ctx.beginPath();
  values.forEach((value, index) => {
    const xPos = x(index);
    const yPos = y(value);
    if (index === 0) ctx.moveTo(xPos, yPos);
    else ctx.lineTo(xPos, yPos);
  });
  ctx.lineWidth = 2;
  ctx.strokeStyle = options.color;
  ctx.stroke();
  ctx.lineTo(x(values.length - 1), height - pad.bottom);
  ctx.lineTo(x(0), height - pad.bottom);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.fillStyle = '#65809e';
  ctx.fillText('滚动窗口', pad.left, height - 4);
  ctx.fillText('最近', width - pad.right - 24, height - 4);
}

function drawYearlyChart() {
  const canvas = $('#yearlyChart');
  if (!canvas || !state.data) return;
  const yearly = state.data.experimental.yearly || [];
  const { ctx, width, height } = prepareCanvas(canvas);
  const pad = { left: 40, right: 16, top: 18, bottom: 30 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  ctx.clearRect(0, 0, width, height);
  ctx.font = '10px -apple-system, BlinkMacSystemFont, sans-serif';
  for (let step = 0; step <= 4; step += 1) {
    const value = step / 4;
    const yPos = pad.top + (1 - value) * plotH;
    ctx.strokeStyle = 'rgba(146, 180, 220, 0.09)';
    ctx.beginPath();
    ctx.moveTo(pad.left, yPos);
    ctx.lineTo(width - pad.right, yPos);
    ctx.stroke();
    ctx.fillStyle = '#65809e';
    ctx.fillText(pct(value, 0), 2, yPos + 3);
  }
  const groupWidth = plotW / Math.max(1, yearly.length);
  yearly.forEach((item, index) => {
    const baseX = pad.left + index * groupWidth + groupWidth * 0.16;
    const barW = groupWidth * 0.28;
    const coverageH = item.top10_coverage_rate * plotH;
    const prizeH = item.any_prize_rate * plotH;
    ctx.fillStyle = '#f5c451';
    ctx.fillRect(baseX, pad.top + plotH - coverageH, barW, coverageH);
    ctx.fillStyle = '#54a8ff';
    ctx.fillRect(baseX + barW + 4, pad.top + plotH - prizeH, barW, prizeH);
    ctx.fillStyle = '#8da5c1';
    ctx.fillText(String(item.year), baseX, height - 8);
  });
  ctx.fillStyle = '#f5c451';
  ctx.fillRect(width - 138, 2, 8, 8);
  ctx.fillStyle = '#8da5c1';
  ctx.fillText('Top-10 覆盖', width - 124, 10);
  ctx.fillStyle = '#54a8ff';
  ctx.fillRect(width - 72, 2, 8, 8);
  ctx.fillStyle = '#8da5c1';
  ctx.fillText('任意奖级', width - 58, 10);
}

function renderModels() {
  const champion = state.data.champion;
  $('#verdictTitle').textContent = champion.key === 'random'
    ? '严格结论：没有模型通过显著性门槛'
    : `严格结论：验证段选出 ${champion.label}`;
  $('#verdictText').textContent = state.data.meta.champion_reason
    + ' 测试段只用于评估，不参与选择。NDCG@10 越高越好；若相对随机基线的 95% 区间跨越 0，就不能宣称预测能力。';

  const rows = state.data.models.map((model) => {
    const v = model.validation;
    const t = model.test;
    const sig = model.validation_vs_random?.significant_improvement;
    const diff = model.validation_vs_random?.difference;
    return `
      <tr class="${model.selected ? 'selected' : ''}">
        <td>${model.label}${model.selected ? ' · 严格选择' : ''}</td>
        <td>${fixed(v.mean_ndcg_at_10)}</td>
        <td>${fixed(t.mean_ndcg_at_10)}</td>
        <td>${pct(t.top10_coverage_rate)}</td>
        <td>${pct(t.any_prize_rate)}</td>
        <td>${fixed(t.avg_red_hits, 3)}</td>
        <td>${pct(t.roi)}</td>
        <td><span class="tag ${sig ? 'good' : 'neutral'}">${sig ? '验证显著' : '无显著'}</span>${diff !== undefined && diff !== null ? ` <span class="date-cell">${signed(diff)}</span>` : ''}</td>
      </tr>
    `;
  }).join('');
  $('#modelTableBody').innerHTML = rows;
  requestAnimationFrame(drawYearlyChart);
}

function renderData() {
  const { meta, baseline } = state.data;
  $('#sourceLink').innerHTML = `<a href="${meta.source_page || meta.source}" target="_blank" rel="noreferrer">${meta.source}</a>`;
  $('#fetchedAt').textContent = meta.fetched_at_utc ? new Date(meta.fetched_at_utc).toLocaleString('zh-CN') : '—';
  $('#dataRows').textContent = `${meta.rows} 期`;
  $('#firstIssue').textContent = `${meta.first_issue} · ${meta.first_date}`;
  $('#lastIssue').textContent = `${meta.last_issue} · ${meta.last_date}`;
  $('#csvHash').textContent = meta.csv_sha256 || '—';
  $('#rawHash').textContent = meta.raw_sha256 || '—';
  $('#disclaimerText').textContent = state.data.disclaimer;
  $('#checkList').innerHTML = [
    `${meta.rows} 期官方记录完整`,
    '期号按年份连续，无缺期 / 重复',
    '每期 6 个红球，范围 1–33，不重复',
    '每期 1 个蓝球，范围 1–16',
    '中彩网 100 条/页 + 福彩官网 50 条/页',
    '2091 期双源重叠段红蓝球逐期一致',
    '每期一等奖注数与奖金非空',
    `单注头奖理论概率 ${pct(baseline.exact_jackpot_rate, 6)}`,
  ].map((text) => `<div class="check-item">${text}</div>`).join('');
}

function renderAll() {
  renderMeta();
  renderStats();
  renderPredictions();
  saveScore();
  newChallenge();
  renderBacktest();
  renderHoldout();
  renderSignal();
  renderExtreme();
  renderModels();
  renderData();
}

function bindEvents() {
  $$('.tab').forEach((tab) => tab.addEventListener('click', () => switchView(tab.dataset.view)));

  $$('#modeSwitch .seg').forEach((button) => button.addEventListener('click', () => {
    state.mode = button.dataset.mode;
    state.page = 1;
    $$('#modeSwitch .seg').forEach((item) => item.classList.toggle('active', item === button));
    renderBacktest();
  }));

  $('#yearFilter').addEventListener('change', (event) => {
    state.year = event.target.value;
    state.page = 1;
    renderBacktest();
  });
  $('#prevPage').addEventListener('click', () => {
    state.page -= 1;
    renderCaseTable();
  });
  $('#nextPage').addEventListener('click', () => {
    state.page += 1;
    renderCaseTable();
  });

  $('#strictResample').addEventListener('click', () => {
    const ticket = randomTicket();
    state.strictTicket = ticket;
    setSingle('#strictSingle', ticket.reds, ticket.blue);
    $('#strictNote').textContent = '已重新抽取随机基线。它仍然没有预测优势；严格模式只是拒绝伪造排序。';
  });

  $('#challenge').addEventListener('click', (event) => {
    const action = event.target.closest('[data-action]')?.dataset.action;
    if (action === 'reveal') revealChallenge();
    if (action === 'next') newChallenge();
  });

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if ($('#view-backtest').classList.contains('active')) renderBacktest();
      if ($('#view-models').classList.contains('active')) drawYearlyChart();
    }, 120);
  });
}

async function init() {
  try {
    const [response, holdoutResponse, extremeResponse, signalResponse] = await Promise.all([
      fetch('data/dashboard.json', { cache: 'no-store' }),
      fetch('data/holdout2015.json', { cache: 'no-store' }),
      fetch('data/extreme.json', { cache: 'no-store' }),
      fetch('data/signal_audit.json', { cache: 'no-store' }),
    ]);
    if (!response.ok) throw new Error(`dashboard HTTP ${response.status}`);
    state.data = await response.json();
    state.holdout = holdoutResponse.ok ? await holdoutResponse.json() : null;
    state.extreme = extremeResponse.ok ? await extremeResponse.json() : null;
    state.signal = signalResponse.ok ? await signalResponse.json() : null;
    renderAll();
    bindEvents();
  } catch (error) {
    $('#dataBadge').textContent = '数据加载失败';
    $('#challenge').innerHTML = `<div class="challenge-empty">无法读取 dashboard.json：${error.message}<br>请用 <code>python3 -m lottery_lab.server</code> 启动本地服务。</div>`;
    $('#statsGrid').innerHTML = '<div class="challenge-empty">数据文件加载失败。请从项目根目录运行本地服务器。</div>';
  }
}

init();
