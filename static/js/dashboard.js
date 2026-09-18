const API = {
  overview: () => fetch('/api/overview').then(r => r.json()),
  processes: (params) => fetch('/api/processes?' + new URLSearchParams(params)).then(r => r.json()),
  process: (id) => fetch(`/api/processes/${encodeURIComponent(id)}`).then(r => r.json()),
  ml: () => fetch('/api/ml').then(r => r.json()),
  upload: (file) => {
    const fd = new FormData();
    fd.append('file', file);
    return fetch('/api/upload', { method: 'POST', body: fd }).then(r => r.json());
  },
  reset: () => fetch('/api/reset', { method: 'POST' }).then(r => r.json()),
};

const CHART_FONT = { family: "'IBM Plex Mono', monospace", size: 11 };
const COLORS = {
  amber: '#f2a93c', green: '#45b880', coral: '#e8615a', blue: '#5ba3d9',
  muted: '#8398b0', grid: '#1c2a3a',
};
const LEVEL_COLORS = ['#e8615a', '#f2a93c', '#9bc24a', '#45b880', '#5ba3d9'];

let charts = {};
let activeProcessId = null;

function destroyChart(key) {
  if (charts[key]) { charts[key].destroy(); delete charts[key]; }
}

function baseGridOptions() {
  return {
    scales: {
      x: { ticks: { color: COLORS.muted, font: CHART_FONT }, grid: { color: COLORS.grid } },
      y: { ticks: { color: COLORS.muted, font: CHART_FONT }, grid: { color: COLORS.grid }, beginAtZero: true },
    },
    plugins: { legend: { labels: { color: COLORS.muted, font: CHART_FONT } } },
  };
}

function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('is-visible');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove('is-visible'), 3200);
}

// ---------------- NAV ----------------
document.querySelectorAll('.deck-nav__link').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.deck-nav__link').forEach(b => b.classList.remove('is-active'));
    btn.classList.add('is-active');
    const view = btn.dataset.view;
    document.querySelectorAll('.view').forEach(v => v.classList.add('is-hidden'));
    document.getElementById('view-' + view).classList.remove('is-hidden');
    if (view === 'ml') loadML();
    if (view === 'processes') loadRegistry();
  });
});

// ---------------- OVERVIEW ----------------
async function loadOverview() {
  const d = await API.overview();

  const strip = document.getElementById('stat-strip');
  strip.innerHTML = `
    <div class="stat-card stat-card--amber">
      <div class="stat-card__label">Tracked Processes</div>
      <div class="stat-card__value">${d.total_processes}</div>
    </div>
    <div class="stat-card stat-card--blue">
      <div class="stat-card__label">Avg Maturity Score</div>
      <div class="stat-card__value">${d.avg_maturity_score}</div>
      <div class="stat-card__foot">out of 100</div>
    </div>
    <div class="stat-card stat-card--blue">
      <div class="stat-card__label">Avg Governance Score</div>
      <div class="stat-card__value">${d.avg_governance_score}</div>
      <div class="stat-card__foot">out of 100</div>
    </div>
    <div class="stat-card stat-card--green">
      <div class="stat-card__label">Measured Gate</div>
      <div class="stat-card__value">${d.pam_ready_pct}%</div>
      <div class="stat-card__foot">of fleet passing</div>
    </div>
    <div class="stat-card stat-card--coral">
      <div class="stat-card__label">Optimized Gate</div>
      <div class="stat-card__value">${d.ai_ready_pct}%</div>
      <div class="stat-card__foot">of fleet passing</div>
    </div>`;

  destroyChart('maturity');
  charts.maturity = new Chart(document.getElementById('chart-maturity'), {
    type: 'bar',
    data: {
      labels: ['1 · Initial', '2 · Managed', '3 · Defined', '4 · Measured', '5 · Optimized'],
      datasets: [{
        label: 'Processes',
        data: [1, 2, 3, 4, 5].map(l => d.maturity_level_counts[l] || 0),
        backgroundColor: LEVEL_COLORS,
        borderRadius: 2,
        maxBarThickness: 46,
      }],
    },
    options: { ...baseGridOptions(), plugins: { legend: { display: false } } },
  });

  destroyChart('status');
  const statusOrder = ['Critical', 'Needs Improvement', 'Managed', 'Strong'];
  const statusColors = { Critical: COLORS.coral, 'Needs Improvement': COLORS.amber, Managed: '#9bc24a', Strong: COLORS.green };
  charts.status = new Chart(document.getElementById('chart-status'), {
    type: 'doughnut',
    data: {
      labels: statusOrder,
      datasets: [{
        data: statusOrder.map(s => d.governance_status_counts[s] || 0),
        backgroundColor: statusOrder.map(s => statusColors[s]),
        borderColor: '#141f2b',
        borderWidth: 2,
      }],
    },
    options: { plugins: { legend: { position: 'bottom', labels: { color: COLORS.muted, font: CHART_FONT, padding: 14 } } }, cutout: '62%' },
  });

  destroyChart('radar');
  const dims = d.governance_dimension_avg;
  const dimLabels = { data_completeness: 'Completeness', data_accuracy: 'Accuracy', data_consistency: 'Consistency',
    data_timeliness: 'Timeliness', data_lineage: 'Lineage', data_ownership: 'Ownership',
    metadata_completeness: 'Metadata', data_access_compliance: 'Access/Compliance' };
  charts.radar = new Chart(document.getElementById('chart-radar'), {
    type: 'radar',
    data: {
      labels: Object.keys(dimLabels).map(k => dimLabels[k]),
      datasets: [{
        label: 'Fleet average %',
        data: Object.keys(dimLabels).map(k => dims[k]),
        backgroundColor: 'rgba(242, 169, 60, 0.15)',
        borderColor: COLORS.amber,
        pointBackgroundColor: COLORS.amber,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        r: {
          angleLines: { color: COLORS.grid }, grid: { color: COLORS.grid },
          pointLabels: { color: COLORS.muted, font: { size: 10.5 } },
          ticks: { color: COLORS.muted, backdropColor: 'transparent', font: { size: 9 } },
          suggestedMin: 0, suggestedMax: 100,
        },
      },
    },
  });

  destroyChart('dept');
  const deptEntries = Object.entries(d.avg_maturity_by_department).sort((a, b) => b[1] - a[1]);
  charts.dept = new Chart(document.getElementById('chart-dept'), {
    type: 'bar',
    data: {
      labels: deptEntries.map(e => e[0]),
      datasets: [{ label: 'Avg maturity score', data: deptEntries.map(e => e[1]), backgroundColor: COLORS.blue, borderRadius: 2 }],
    },
    options: { ...baseGridOptions(), indexAxis: 'y', plugins: { legend: { display: false } } },
  });

  const gapsEl = document.getElementById('top-gaps');
  gapsEl.innerHTML = d.top_gaps.map(g => `
    <div class="gap-row">
      <span class="gap-row__code">${g.code}</span>
      <span>${g.name}</span>
      <span class="gap-row__count">${g.count}×</span>
      <span class="gap-row__action">${g.action}</span>
    </div>`).join('') || '<div class="gap-row">No rules currently triggered fleet-wide.</div>';
}

// ---------------- REGISTRY ----------------
async function loadRegistry() {
  const q = document.getElementById('process-search').value;
  const level = document.getElementById('filter-level').value;
  const status = document.getElementById('filter-status').value;
  const params = {};
  if (q) params.q = q;
  if (level) params.level = level;
  if (status) params.status = status;

  const d = await API.processes(params);
  const tbody = document.getElementById('registry-tbody');
  const levelClass = { 1: 'l1', 2: 'l2', 3: 'l3', 4: 'l4', 5: 'l5' };

  tbody.innerHTML = d.processes.map(p => `
    <tr data-id="${p.process_id}" class="${p.process_id === activeProcessId ? 'is-active' : ''}">
      <td>${p.process_name}</td>
      <td class="mono">${p.department}</td>
      <td><span class="pill pill--${levelClass[p.maturity_level]}">L${p.maturity_level} · ${p.maturity_score}</span></td>
      <td class="mono">${p.governance_score}</td>
      <td><span class="gate-dot gate-dot--${p.pam_ready ? 'pass' : 'fail'}"></span></td>
      <td><span class="gate-dot gate-dot--${p.ai_ready ? 'pass' : 'fail'}"></span></td>
      <td class="mono">${p.exception_rate}%</td>
    </tr>`).join('');

  tbody.querySelectorAll('tr').forEach(tr => {
    tr.addEventListener('click', () => openDetail(tr.dataset.id));
  });
}

async function openDetail(id) {
  activeProcessId = id;
  document.querySelectorAll('#registry-tbody tr').forEach(tr => {
    tr.classList.toggle('is-active', tr.dataset.id === id);
  });

  const p = await API.process(id);
  const panel = document.getElementById('detail-panel');

  const dimBars = p.governance.dimensions.map(dm => `
    <div class="dim-bar-row">
      <span class="dim-bar-row__label">${dm.label}</span>
      <span class="dim-bar-track"><span class="dim-bar-fill" style="width:${dm.value}%"></span></span>
      <span class="dim-bar-row__value">${dm.value}%</span>
    </div>`).join('');

  const gaps = p.gaps.length
    ? `<ul class="plain-list">${p.gaps.map(g => `<li>${g}</li>`).join('')}</ul>`
    : '<div class="target-callout">No open gaps — this process is fully compliant with tracked rules.</div>';

  const rules = p.triggered_rules.length
    ? p.triggered_rules.map(r => `
      <div class="rule-card">
        <div class="rule-card__head"><span>${r.code} — ${r.name}</span></div>
        <div class="rule-card__desc">${r.description}</div>
        <div class="rule-card__action">→ ${r.action}</div>
      </div>`).join('')
    : '<div class="target-callout">No rules currently triggered.</div>';

  const recs = p.recommendations.length
    ? `<ul class="plain-list">${p.recommendations.map(r => `<li>${r}</li>`).join('')}</ul>`
    : '';

  panel.innerHTML = `
    <div class="detail-head">
      <div class="detail-head__id">${p.process_id}</div>
      <div class="detail-head__name">${p.process_name}</div>
      <div class="detail-head__dept">${p.department}</div>
    </div>

    <div class="detail-scorerow">
      <div class="detail-score">
        <div class="detail-score__label">Governance</div>
        <div class="detail-score__value">${p.governance.score}<span style="font-size:12px;color:var(--text-faint)"> /100</span></div>
        <div class="stat-card__foot">${p.governance.status}</div>
      </div>
      <div class="detail-score">
        <div class="detail-score__label">Maturity</div>
        <div class="detail-score__value">${p.maturity.score}<span style="font-size:12px;color:var(--text-faint)"> /100</span></div>
        <div class="stat-card__foot">Level ${p.maturity.level} — ${p.maturity.label}</div>
      </div>
    </div>

    <div class="detail-section">
      <h3>Data Governance Dimensions</h3>
      ${dimBars}
    </div>

    <div class="detail-section">
      <h3>Readiness Gates</h3>
      <div class="gate-line"><span>Measured</span><span class="gate-status gate-status--${p.gates.pam_ready ? 'pass' : 'fail'}">${p.gates.pam_ready ? 'PASS' : 'FAIL'}</span></div>
      <div class="gate-line"><span>Optimized</span><span class="gate-status gate-status--${p.gates.ai_ready ? 'pass' : 'fail'}">${p.gates.ai_ready ? 'PASS' : 'FAIL'}</span></div>
    </div>

    <div class="detail-section">
      <h3>Performance Indicators</h3>
      <div class="gate-line"><span>Cycle time</span><span class="mono">${p.indicators.cycle_time_hours} h</span></div>
      <div class="gate-line"><span>Waiting time</span><span class="mono">${p.indicators.waiting_time_hours} h</span></div>
      <div class="gate-line"><span>Exception rate</span><span class="mono">${p.indicators.exception_rate}%</span></div>
      <div class="gate-line"><span>Rework rate</span><span class="mono">${p.indicators.rework_rate}%</span></div>
      <div class="gate-line"><span>Manual handling</span><span class="mono">${p.indicators.manual_handling_rate}%</span></div>
      <div class="gate-line"><span>Handoffs</span><span class="mono">${p.indicators.handoff_count}</span></div>
    </div>

    <div class="detail-section">
      <h3>Key Gaps</h3>
      ${gaps}
    </div>

    <div class="detail-section">
      <h3>Triggered Rules</h3>
      ${rules}
    </div>

    ${recs ? `<div class="detail-section"><h3>Recommendations</h3>${recs}</div>` : ''}

    <div class="detail-section">
      <h3>Target / Next State</h3>
      <div class="target-callout">→ ${p.maturity.target_next_state}</div>
    </div>
  `;
}

document.getElementById('process-search').addEventListener('input', debounce(loadRegistry, 250));
document.getElementById('filter-level').addEventListener('change', loadRegistry);
document.getElementById('filter-status').addEventListener('change', loadRegistry);

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// ---------------- ML INSIGHTS ----------------
async function loadML() {
  const d = await API.ml();
  const m = d.metrics;

  document.getElementById('ml-stat-strip').innerHTML = `
    <div class="stat-card stat-card--amber">
      <div class="stat-card__label">Model</div>
      <div class="stat-card__value" style="font-size:16px">RandomForest</div>
      <div class="stat-card__foot">300 trees, depth 10</div>
    </div>
    <div class="stat-card stat-card--green">
      <div class="stat-card__label">Test Accuracy</div>
      <div class="stat-card__value">${(m.accuracy * 100).toFixed(1)}%</div>
    </div>
    <div class="stat-card stat-card--blue">
      <div class="stat-card__label">Train / Test Split</div>
      <div class="stat-card__value" style="font-size:16px">${m.train_size} / ${m.test_size}</div>
    </div>
    <div class="stat-card stat-card--blue">
      <div class="stat-card__label">Target</div>
      <div class="stat-card__value" style="font-size:16px">Maturity Level</div>
      <div class="stat-card__foot">5 classes</div>
    </div>`;

  destroyChart('importance');
  charts.importance = new Chart(document.getElementById('chart-importance'), {
    type: 'bar',
    data: {
      labels: d.feature_importance.map(f => f.feature),
      datasets: [{ label: 'Importance', data: d.feature_importance.map(f => f.importance), backgroundColor: COLORS.amber, borderRadius: 2 }],
    },
    options: { ...baseGridOptions(), indexAxis: 'y', plugins: { legend: { display: false } } },
  });

  const table = document.getElementById('ml-table');
  const labels = d.maturity_labels;
  const rows = Object.keys(m.report).filter(k => !['accuracy', 'macro avg', 'weighted avg'].includes(k));
  table.innerHTML = `
    <thead><tr><th>Level</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
    <tbody>
      ${rows.map(k => {
        const r = m.report[k];
        return `<tr>
          <td>Level ${k} — ${labels[k] || ''}</td>
          <td class="mono">${r.precision.toFixed(2)}</td>
          <td class="mono">${r.recall.toFixed(2)}</td>
          <td class="mono">${r['f1-score'].toFixed(2)}</td>
          <td class="mono">${r.support}</td>
        </tr>`;
      }).join('')}
    </tbody>`;
}

// ---------------- DATASET UPLOAD ----------------
document.getElementById('csv-upload').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  document.getElementById('dataset-status').textContent = 'Processing…';
  const res = await API.upload(file);
  if (res.error) {
    toast('Upload failed: ' + res.error);
    document.getElementById('dataset-status').textContent = 'Upload failed';
    return;
  }
  document.getElementById('dataset-status').textContent = `${res.rows} rows loaded`;
  toast(`Dataset replaced — ${res.rows} processes loaded and model retrained.`);
  refreshAll();
});

document.getElementById('reset-dataset').addEventListener('click', async () => {
  document.getElementById('dataset-status').textContent = 'Resetting…';
  const res = await API.reset();
  document.getElementById('dataset-status').textContent = `${res.rows} rows loaded`;
  toast('Reset to bundled sample dataset.');
  refreshAll();
});

function refreshAll() {
  activeProcessId = null;
  loadOverview();
  loadRegistry();
  document.getElementById('detail-panel').innerHTML = '<div class="detail-panel__empty">Select a process from the registry to open its control-tower readout.</div>';
}

// ---------------- INIT ----------------
loadOverview();
