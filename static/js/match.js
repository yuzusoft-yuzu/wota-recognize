// B站视频溯源匹配 - 首页内嵌卡片逻辑
(function () {
  const dz = document.getElementById('matchDropzone');
  if (!dz) return;  // 首页没有该区块时静默退出
  const fileInput = document.getElementById('matchFileInput');
  const fileChosen = document.getElementById('matchFileChosen');
  const submitBtn = document.getElementById('matchSubmitBtn');
  const progressBox = document.getElementById('matchProgressBox');
  const progressLabel = document.getElementById('matchProgressLabel');
  const progressBar = document.getElementById('matchProgressBar');
  const resultBox = document.getElementById('matchResultBox');
  let selectedFile = null;

  const STATUS_LABEL = {
    queued: '排队中…', processing: '正在读取视频…', extracting: '提取关键帧特征中…',
    searching: '在B站视频库中检索…', done: '完成', error: '出错',
  };

  dz.addEventListener('click', () => fileInput.click());
  ['dragenter', 'dragover'].forEach(ev =>
    dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach(ev =>
    dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', e => { if (e.dataTransfer.files.length) pickFile(e.dataTransfer.files[0]); });
  fileInput.addEventListener('change', e => { if (e.target.files.length) pickFile(e.target.files[0]); });

  function pickFile(f) {
    if (!f.type.startsWith('video/') && !/\.(mp4|mov|webm|avi|mkv|flv|m4v)$/i.test(f.name)) {
      alert('请选择视频文件'); return;
    }
    selectedFile = f;
    const mb = (f.size / 1024 / 1024).toFixed(1);
    fileChosen.hidden = false;
    fileChosen.textContent = `已选择：${f.name}（${mb} MB）`;
    submitBtn.disabled = false;
  }

  submitBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    submitBtn.disabled = true;
    resultBox.hidden = true;
    progressBox.hidden = false;
    setProgress(2, 'queued');
    const fd = new FormData();
    fd.append('video', selectedFile);
    try {
      const r = await fetch('/api/match-video', { method: 'POST', body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || '上传失败');
      poll(d.task_id);
    } catch (e) {
      fail(e.message);
    }
  });

  async function poll(taskId) {
    const t = setInterval(async () => {
      try {
        const r = await fetch(`/api/match-video/${taskId}`);
        const d = await r.json();
        setProgress(d.progress, d.status);
        if (d.status === 'done') {
          clearInterval(t);
          renderResult(d.result);
        } else if (d.status === 'error') {
          clearInterval(t);
          fail(d.error);
        }
      } catch (e) { clearInterval(t); fail(e.message); }
    }, 1200);
  }

  function setProgress(p, status) {
    progressBar.style.width = (p || 0) + '%';
    progressLabel.textContent = STATUS_LABEL[status] || status || '处理中…';
  }

  function fail(msg) {
    progressBox.hidden = true;
    resultBox.hidden = false;
    resultBox.innerHTML = `<div class="alert error">溯源失败：${esc(msg)}</div>`;
    submitBtn.disabled = false;
  }

  function renderResult(res) {
    progressBox.hidden = true;
    resultBox.hidden = false;
    if (!res.db_ready) {
      resultBox.innerHTML = `<div class="alert info">${esc(res.message || 'B站视频特征库尚未建立，请等待数据爬取完成后再试。')}</div>`;
      submitBtn.disabled = false;
      return;
    }
    let html = `<div class="muted" style="margin-bottom:8px">已比对 ${res.video_count} 个B站wota艺视频 · 提取 ${res.query_frames} 个关键帧</div>`;
    if (!res.matches || res.matches.length === 0) {
      html += `<div class="alert error">未找到相似的B站视频。可能该视频尚未被收录，或视频内容差异较大。</div>`;
    } else {
      const top = res.matches[0];
      html += `<div class="topmatch">
        <div class="muted">最相似的B站视频</div>
        <div class="tm-name">${esc(top.title)}</div>
        <div class="tm-jp">UP主：${esc(top.up_name || '未知')} · 播放 ${fmtPlay(top.play_count)}</div>
        <div class="tm-pct">${top.similarity}<small>%</small></div>
        <div class="muted">内容相似度 · 命中 ${top.vote}/${res.query_frames} 帧</div>
        <div style="margin-top:8px"><a class="btn ghost" href="${esc(top.link)}" target="_blank">在B站观看 →</a></div>
      </div>`;
      if (res.matches.length > 1) {
        html += `<div class="pred-list">`;
        res.matches.slice(1).forEach((m, i) => {
          html += `<div class="pred">
            <div class="pred-rank">${i + 2}</div>
            <div><div class="pred-name">${esc(m.title)}</div>
              <div class="pred-meta">UP主：${esc(m.up_name || '未知')} · 播放 ${fmtPlay(m.play_count)}</div></div>
            <div class="pred-bar"><i style="width:${m.similarity}%"></i></div>
            <div class="pred-pct">${m.similarity}%</div>
          </div>`;
        });
        html += `</div>`;
      }
    }
    resultBox.innerHTML = html;
    submitBtn.disabled = false;
  }

  function fmtPlay(n) {
    n = Number(n || 0);
    if (n >= 10000) return (n / 10000).toFixed(1) + '万';
    return String(n);
  }
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
})();
