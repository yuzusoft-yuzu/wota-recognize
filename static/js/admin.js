// 管理员界面逻辑：登录 + 标准动作上传入库
(function () {
  const loginPanel = document.getElementById('loginPanel');
  const adminPanel = document.getElementById('adminPanel');
  const loginBtn = document.getElementById('loginBtn');
  const loginError = document.getElementById('loginError');
  const logoutBtn = document.getElementById('logoutBtn');

  const adminDrop = document.getElementById('adminDrop');
  const adminFile = document.getElementById('adminFile');
  const adminFileChosen = document.getElementById('adminFileChosen');
  const uploadBtn = document.getElementById('uploadBtn');
  const adminProgress = document.getElementById('adminProgress');
  const adminProgLabel = document.getElementById('adminProgLabel');
  const adminProgBar = document.getElementById('adminProgBar');
  const adminResult = document.getElementById('adminResult');
  const adminIdle = document.getElementById('adminIdle');
  const adminCount = document.getElementById('adminCount');

  let selectedFile = null;

  const STATUS_LABEL = {
    queued: '排队中…', processing: '正在读取视频…', extracting: '骨光融合特征提取中…',
    storing: '写入数据库…', finalizing: '生成预览…', done: '入库完成', error: '出错', merging: '合并视频…',
  };

  // 初始化：检查登录状态
  fetch('/api/auth/check').then(r => r.json()).then(d => {
    if (d.authed) showAdmin(); else showLogin();
  }).catch(showLogin);

  function showLogin() { loginPanel.hidden = false; adminPanel.hidden = true; }
  function showAdmin() {
    loginPanel.hidden = true; adminPanel.hidden = false;
    refreshCount();
  }

  loginBtn.addEventListener('click', async () => {
    loginError.hidden = true;
    const username = document.getElementById('loginUser').value.trim();
    const password = document.getElementById('loginPass').value;
    if (!username || !password) { loginError.textContent = '请输入账号和密码'; loginError.hidden = false; return; }
    try {
      const r = await fetch('/api/auth/login', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      const d = await r.json();
      if (r.ok && d.success) showAdmin();
      else { loginError.textContent = d.error || '登录失败'; loginError.hidden = false; }
    } catch (e) { loginError.textContent = e.message; loginError.hidden = false; }
  });

  document.getElementById('loginPass').addEventListener('keydown', e => { if (e.key === 'Enter') loginBtn.click(); });

  logoutBtn.addEventListener('click', async () => {
    await fetch('/api/auth/logout', { method: 'POST' });
    showLogin();
  });

  // 文件选择
  adminDrop.addEventListener('click', () => adminFile.click());
  ['dragenter', 'dragover'].forEach(ev =>
    adminDrop.addEventListener(ev, e => { e.preventDefault(); adminDrop.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach(ev =>
    adminDrop.addEventListener(ev, e => { e.preventDefault(); adminDrop.classList.remove('drag'); }));
  adminDrop.addEventListener('drop', e => { if (e.dataTransfer.files.length) pick(e.dataTransfer.files[0]); });
  adminFile.addEventListener('change', e => { if (e.target.files.length) pick(e.target.files[0]); });

  function pick(f) {
    if (!f.type.startsWith('video/') && !/\.(mp4|mov|webm|avi|mkv|flv|m4v)$/i.test(f.name)) {
      alert('请选择视频文件'); return;
    }
    selectedFile = f;
    const mb = (f.size / 1024 / 1024).toFixed(1);
    adminFileChosen.hidden = false;
    adminFileChosen.textContent = `已选择：${f.name}（${mb} MB）`;
    validate();
  }

  function validate() {
    const name = document.getElementById('fName').value.trim();
    uploadBtn.disabled = !(selectedFile && name);
  }
  ['fName', 'fJp', 'fCat', 'fBili', 'fDesc'].forEach(id =>
    document.getElementById(id).addEventListener('input', validate));

  uploadBtn.addEventListener('click', async () => {
    const name = document.getElementById('fName').value.trim();
    if (!name || !selectedFile) return;
    if (selectedFile.size > 50 * 1024 * 1024) {
      alert('视频需在 50MB 以内'); return;
    }
    uploadBtn.disabled = true;
    adminIdle.hidden = true;
    adminResult.innerHTML = '';
    adminProgress.hidden = false;
    setProg(2, 'queued');
    const fd = new FormData();
    fd.append('video', selectedFile);
    fd.append('name', name);
    fd.append('japanese_name', val('fJp'));
    fd.append('category', val('fCat'));
    fd.append('bilibili', val('fBili'));
    fd.append('description', val('fDesc'));
    try {
      const r = await fetch('/api/admin/upload', { method: 'POST', body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || '上传失败');
      poll(d.task_id);
    } catch (e) {
      adminProgress.hidden = true;
      adminResult.innerHTML = `<div class="alert error">${esc(e.message)}</div>`;
      uploadBtn.disabled = false;
    }
  });

  async function poll(taskId) {
    const t = setInterval(async () => {
      try {
        const r = await fetch(`/api/admin/upload/${taskId}`);
        const d = await r.json();
        setProg(d.progress, d.status);
        if (d.status === 'done') { clearInterval(t); renderOk(d.result); }
        else if (d.status === 'error') { clearInterval(t); renderErr(d.error); }
      } catch (e) { clearInterval(t); renderErr(e.message); }
    }, 1200);
  }

  function setProg(p, status) {
    adminProgBar.style.width = (p || 0) + '%';
    adminProgLabel.textContent = STATUS_LABEL[status] || status || '处理中…';
  }

  function renderOk(res) {
    adminProgress.hidden = true;
    adminResult.innerHTML = `
      <div class="alert ok">✅ 入库成功！</div>
      <div class="topmatch">
        <div class="muted">已收录技名</div>
        <div class="tm-name">${esc(res.move_name)}</div>
        <div class="muted" style="margin-top:8px">当前数据库共 <b>${res.total_count}</b> 个技名</div>
        <div class="muted" style="margin-top:4px">${res.query_frames} 帧 · ${res.duration}s · ${res.used_skeleton ? '骨骼+光棒融合' : '纯光棒模式'}</div>
      </div>
      ${res.preview_url ? `<img class="preview-img" src="${res.preview_url}" alt="预览"/><div class="preview-cap">骨光融合预览</div>` : ''}
      <div class="modal-actions" style="margin-top:14px">
        <a class="btn ghost" href="/techniques">前往技术总览查看 →</a>
      </div>`;
    // 清空表单，便于继续上传
    document.getElementById('fName').value = '';
    document.getElementById('fJp').value = '';
    document.getElementById('fCat').value = '';
    document.getElementById('fBili').value = '';
    document.getElementById('fDesc').value = '';
    adminFileChosen.hidden = true;
    selectedFile = null;
    refreshCount();
    validate();
  }

  function renderErr(msg) {
    adminProgress.hidden = true;
    adminResult.innerHTML = `<div class="alert error">分析入库失败：${esc(msg)}</div>`;
    uploadBtn.disabled = false;
  }

  function refreshCount() {
    fetch('/api/health').then(r => r.json()).then(d => {
      adminCount.textContent = `当前数据库共 ${d.technique_count} 个技名 · ${d.mediapipe_available ? '骨骼融合已启用' : '纯光棒模式'}`;
    }).catch(() => {});
  }

  function val(id) { const el = document.getElementById(id); return el ? el.value.trim() : ''; }
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
})();
