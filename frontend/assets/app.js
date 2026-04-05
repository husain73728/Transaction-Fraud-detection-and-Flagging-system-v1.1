const API_BASE = '';

const q = (id) => document.getElementById(id);

function setStatus(message, tone = 'muted') {
  const el = q('upload-status');
  if (!el) return;
  const colors = {
    muted: 'text-slate-400',
    error: 'text-rose-400',
    success: 'text-emerald-400',
  };
  el.className = `text-sm ${colors[tone] || colors.muted}`;
  el.textContent = message;
}

async function postPrediction(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('/api/predict', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || 'Upload failed');
  }

  return res.json();
}

function wireUploadPage() {
  const form = q('upload-form');
  const fileInput = q('file-input');
  const drop = q('drop-zone');
  const button = q('upload-button');
  const fileLabel = q('selected-file-name');

  if (!form || !fileInput) return;

  const handleFiles = (files) => {
    if (!files || !files.length) return;
    fileInput.files = files;
    if (fileLabel) fileLabel.textContent = files[0].name;
  };

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = fileInput.files?.[0];
    if (!file) {
        setStatus('Select a CSV to start.');
        return;
    }
    setStatus('Scoring in progress…');
    button.disabled = true;
    button.classList.add('opacity-70');
    try {
      const data = await postPrediction(file);
      localStorage.setItem('lastJobId', data.job_id);
      localStorage.setItem('lastSummary', JSON.stringify(data.summary));
      window.location.href = `/results?job_id=${data.job_id}`;
    } catch (err) {
      setStatus(err.message || 'Could not score file', 'error');
    } finally {
      button.disabled = false;
      button.classList.remove('opacity-70');
    }
  });

  drop?.addEventListener('dragover', (e) => {
    e.preventDefault();
    drop.classList.add('ring-2', 'ring-primary');
  });

  drop?.addEventListener('dragleave', () => {
    drop.classList.remove('ring-2', 'ring-primary');
  });

  drop?.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('ring-2', 'ring-primary');
    const files = e.dataTransfer?.files;
    handleFiles(files);
  });

  drop?.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

  // Fill quick summary if available
  const quick = q('last-run');
  const stored = localStorage.getItem('lastSummary');
  if (quick && stored) {
    try {
      const summary = JSON.parse(stored);
      quick.querySelector('[data-job]').textContent = summary.job_id || 'latest';
      quick.querySelector('[data-rows]').textContent = summary.rows ?? '—';
      quick.querySelector('[data-flagged]').textContent = summary.flagged ?? '—';
      quick.classList.remove('hidden');
    } catch (e) {
      // ignore parse errors
    }
  }
}

async function fetchJob(jobId) {
  const res = await fetch(`/api/jobs/${jobId}`);
  if (!res.ok) throw new Error('Job not found');
  return res.json();
}

function renderResults(summary) {
  const set = (id, value) => {
    const el = q(id);
    if (el) el.textContent = value;
  };
  set('result-file-name', summary.file_name);
  set('result-processed-at', new Date(summary.processed_at).toLocaleString());
  set('result-total-rows', summary.rows);
  set('result-high-risk', summary.high_risk);
  set('result-medium-risk', summary.medium_risk);
  set('result-low-risk', summary.low_risk);
  set('result-flagged', summary.flagged);
  set('result-avg-risk', `${summary.avg_risk_percent}%`);
  set('result-max-risk', `${summary.max_risk_percent}%`);
  const download = q('download-link');
  if (download && summary.download_url) {
    download.href = summary.download_url;
  }

  const ignored = new Set(['risk_percent', 'fraud_flag', 'risk_label', 'paysim_prob', 'kartik_prob', 'final_score']);
  const pickField = (row, preferred, used) => {
    for (const key of preferred) {
      if (row[key] !== undefined && row[key] !== null && row[key] !== '') {
        used.add(key);
        return { key, value: row[key] };
      }
    }
    for (const [k, v] of Object.entries(row)) {
      if (ignored.has(k) || used.has(k)) continue;
      if (v !== undefined && v !== null && v !== '') {
        used.add(k);
        return { key: k, value: v };
      }
    }
    return { key: null, value: 'N/A' };
  };

  const table = q('top-cases');
  if (table && summary.top_cases) {
    table.innerHTML = '';
    summary.top_cases.forEach((row, idx) => {
      const used = new Set();
      const merch = pickField(row, ['merchant', 'category', 'Customer Id', 'customer_id', 'Company', 'company', 'name'], used).value;
      const amount = pickField(row, ['amount', 'Amount', 'amt', 'transaction_amount'], used).value;
      const type = pickField(row, ['type', 'Type', 'payment_type', 'txn_type'], used).value;

      const riskPercent = typeof row.risk_percent === 'number'
        ? row.risk_percent.toFixed(2)
        : row.risk_percent
        ? Number(row.risk_percent).toFixed(2)
        : '?';

      const riskClass = row.risk_label === 'HIGH RISK'
        ? 'text-rose-400'
        : row.risk_label === 'MEDIUM RISK'
        ? 'text-amber-300'
        : 'text-emerald-300';

      const safe = (v) => (v === undefined || v === null || v === '') ? 'N/A' : v;

      const tr = document.createElement('tr');
      tr.className = 'border-b border-white/5 last:border-0';
      tr.innerHTML = `
        <td class="py-2 text-sm text-slate-400">#${idx + 1}</td>
        <td class="py-2 text-sm">${safe(merch)}</td>
        <td class="py-2 text-sm">${safe(amount)}</td>
        <td class="py-2 text-sm">${safe(type)}</td>
        <td class="py-2 text-sm">${riskPercent}%</td>
        <td class="py-2 text-sm font-semibold ${riskClass}">${row.risk_label}</td>
      `;
      table.appendChild(tr);
    });
  }
}

function wireResultsPage() {
  const params = new URLSearchParams(window.location.search);
  const requestedId = params.get('job_id');
  const fallbackId = localStorage.getItem('lastJobId');
  const jobId = requestedId || fallbackId;

  if (!jobId) {
    const msg = q('results-status');
    if (msg) msg.textContent = 'Upload a CSV to see results.';
    return;
  }

  fetchJob(jobId)
    .then((summary) => {
      renderResults(summary);
      localStorage.setItem('lastJobId', jobId);
      localStorage.setItem('lastSummary', JSON.stringify(summary));
    })
    .catch(() => {
      const stored = localStorage.getItem('lastSummary');
      if (stored) {
        renderResults(JSON.parse(stored));
        const msg = q('results-status');
        if (msg) msg.textContent = 'Showing last saved run (network unavailable).';
      } else if (q('results-status')) {
        q('results-status').textContent = 'No results available yet.';
      }
    });
}

(function init() {
  if (q('upload-page')) wireUploadPage();
  if (q('results-page')) wireResultsPage();
})();
