// 技术总览页面逻辑：搜索、列表、详情弹窗、管理员编辑/删除
(function () {
  const searchInput = document.getElementById('searchInput');
  const searchClear = document.getElementById('searchClear');
  const techList = document.getElementById('techList');
  const techCount = document.getElementById('techCount');
  const emptyState = document.getElementById('emptyState');
  const adminHint = document.getElementById('adminHint');
  const modal = document.getElementById('detailModal');
  const dTitle = document.getElementById('dTitle');
  const detailBody = document.getElementById('detailBody');
  let isAdmin = false;
  let debounceT = null;
  let lastItems = [];

  // 检查管理员状态
  fetch('/api/auth/check').then(r => r.json()).then(d => {
    isAdmin = !!d.authed;
    if (isAdmin) adminHint.hidden = false;
  }).catch(() => {});

  function load(q) {
    const url = q ? `/api/techniques?q=${encodeURIComponent(q)}` : '/api/techniques';
    fetch(url).then(r => r.json()).then(d => {
      lastItems = d.items || [];
      render(lastItems);
    }).catch(() => { render([]); });
  }
  load('');

  searchInput.addEventListener('input', e => {
    const v = e.target.value.trim();
    searchClear.hidden = !v;
    clearTimeout(debounceT);
    debounceT = setTimeout(() => load(v), 280);
  });
  searchClear.addEventListener('click', () => {
    searchInput.value = ''; searchClear.hidden = true; load('');
  });

  function render(items) {
    techCount.textContent = items.length ? `共 ${items.length} 个技名` : '';
    if (!items.length) {
      techList.innerHTML = '';
      emptyState.hidden = false;
      return;
    }
    emptyState.hidden = true;
    techList.innerHTML = items.map(it => `
      <div class="tech-card" data-id="${it.move_id}">
        <div class="tc-name">${esc(it.move_name)}</div>
        ${it.japanese_name ? `<div class="tc-jp">${esc(it.japanese_name)}</div>` : ''}
        ${it.category ? `<span class="tc-cat">${esc(it.category)}</span>` : ''}
      </div>`).join('');
    techList.querySelectorAll('.tech-card').forEach(el =>
      el.addEventListener('click', () => openDetail(el.dataset.id)));
  }

  async function openDetail(id) {
    const r = await fetch(`/api/techniques/${id}`);
    if (!r.ok) { alert('加载失败'); return; }
    const it = await r.json();
    const admin = isAdmin || it.is_admin;
    dTitle.textContent = it.move_name;
    detailBody.innerHTML = `
      ${admin ? '' : `<div class="alert ok" style="margin-bottom:14px">用户视图：仅查看。登录管理员可修改详细信息。</div>`}
      <div class="detail-row">
        <label>技名</label>
        ${admin ? `<input id="eName" value="${esc(it.move_name)}"/>` : `<div class="dv">${esc(it.move_name)}</div>`}
      </div>
      <div class="detail-row">
        <label>日语名</label>
        ${admin ? `<input id="eJp" value="${esc(it.japanese_name || '')}"/>` : `<div class="dv">${esc(it.japanese_name || '—')}</div>`}
      </div>
      <div class="detail-row">
        <label>分类</label>
        ${admin ? `<input id="eCat" value="${esc(it.category || '')}"/>` : `<div class="dv">${esc(it.category || '—')}</div>`}
      </div>
      <div class="detail-row">
        <label>B站链接 / BV号</label>
        ${admin ? `<input id="eBili" value="${esc(it.bilibili || '')}"/>`
                : `<div class="dv">${it.bilibili ? `<a href="${bilibiliUrl(it.bilibili)}" target="_blank">${esc(it.bilibili)}</a>` : '—'}</div>`}
      </div>
      <div class="detail-row">
        <label>描述</label>
        ${admin ? `<textarea id="eDesc" rows="3">${esc(it.description || '')}</textarea>` : `<div class="dv">${esc(it.description || '—')}</div>`}
      </div>
      <div class="detail-row">
        <label>来源视频</label>
        <div class="dv muted">${esc(it.source_video || '—')}</div>
      </div>
      <div class="detail-row">
        <label>元数据</label>
        <div class="dv muted">${it.frame_count || 0} 帧 · ${it.duration ? Number(it.duration).toFixed(1) : 0}s · 收录于 ${esc((it.created_at || '').slice(0, 10))}</div>
      </div>
      ${admin ? `<div class="modal-actions">
          <button class="btn danger" id="delBtn">删除该技名</button>
          <button class="btn primary" id="saveBtn">保存修改</button>
        </div>` : ''}
    `;
    modal.hidden = false;
    if (admin) {
      document.getElementById('saveBtn').addEventListener('click', () => saveDetail(id));
      document.getElementById('delBtn').addEventListener('click', () => delDetail(id, it.move_name));
    }
  }

  async function saveDetail(id) {
    const body = {
      move_name: val('eName'), japanese_name: val('eJp'), category: val('eCat'),
      bilibili: val('eBili'), description: val('eDesc'),
    };
    const r = await fetch(`/api/techniques/${id}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const d = await r.json();
    if (r.ok && d.success) {
      closeModal(); load(searchInput.value.trim());
    } else { alert(d.error || '保存失败'); }
  }

  async function delDetail(id, name) {
    if (!confirm(`确定删除技名「${name}」？该操作不可撤销。`)) return;
    const r = await fetch(`/api/techniques/${id}`, { method: 'DELETE' });
    const d = await r.json();
    if (r.ok && d.success) { closeModal(); load(searchInput.value.trim()); }
    else alert(d.error || '删除失败');
  }

  function val(id) { const el = document.getElementById(id); return el ? el.value.trim() : ''; }
  function closeModal() { modal.hidden = true; }
  modal.querySelectorAll('[data-close]').forEach(el => el.addEventListener('click', closeModal));

  function bilibiliUrl(s) {
    if (/^https?:\/\//i.test(s)) return s;
    if (/^BV/i.test(s)) return `https://www.bilibili.com/video/${s}`;
    return `https://www.bilibili.com/video/${s}`;
  }
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
})();
