// 动作识别页面逻辑
(function () {
  const dz = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const fileChosen = document.getElementById('fileChosen');
  const submitBtn = document.getElementById('submitBtn');
  const progressBox = document.getElementById('progressBox');
  const progressLabel = document.getElementById('progressLabel');
  const progressBar = document.getElementById('progressBar');
  const resultBox = document.getElementById('resultBox');
  let selectedFile = null;

  const STATUS_LABEL = {
    queued: '排队中…', processing: '正在读取视频…', extracting: '骨光融合特征提取中…',
    matching: '与数据库 DTW 比对中…', finalizing: '生成预览…', done: '完成', error: '出错',
    merging: '合并视频…',
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
      const r = await fetch('/api/recognize', { method: 'POST', body: fd });
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
        const r = await fetch(`/api/recognize/${taskId}`);
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
    resultBox.innerHTML = `<div class="alert error">识别失败：${esc(msg)}</div>`;
    submitBtn.disabled = false;
  }

  function renderResult(res) {
    progressBox.hidden = true;
    resultBox.hidden = false;
    if (!res.has_db || res.db_count === 0) {
      resultBox.innerHTML = `<div class="alert error">数据库为空，尚无标准动作可比对。请先由管理员在「管理员界面」上传标准动作视频。</div>`;
      submitBtn.disabled = false;
      return;
    }
    let html = '';
    if (res.predictions.length === 0) {
      html = `<div class="alert error">未找到匹配的技名（匹配度过低）。可尝试上传更清晰、更完整的动作视频。</div>`;
    } else {
      const top = res.predictions[0];
      html += `<div class="topmatch">
        <div class="muted">识别结果</div>
        <div class="tm-name">${esc(top.move_name)}</div>
        ${top.japanese_name ? `<div class="tm-jp">${esc(top.japanese_name)}</div>` : ''}
        <div class="tm-pct">${top.match}<small>%</small></div>
        <div class="muted">匹配度${top.category ? ' · 分类 ' + esc(top.category) : ''}</div>
        ${top.bilibili ? `<div style="margin-top:8px"><a class="btn ghost" href="${bilibiliUrl(top.bilibili)}" target="_blank">观看 B站参考 →</a></div>` : ''}
      </div>`;
      if (res.predictions.length > 1) {
        html += `<div class="pred-list">`;
        res.predictions.slice(1).forEach((p, i) => {
          html += `<div class="pred">
            <div class="pred-rank">${i + 2}</div>
            <div><div class="pred-name">${esc(p.move_name)}</div>
              <div class="pred-meta">${esc(p.japanese_name || '')}${p.category ? ' · ' + esc(p.category) : ''}</div></div>
            <div class="pred-bar"><i style="width:${p.match}%"></i></div>
            <div class="pred-pct">${p.match}%</div>
          </div>`;
        });
        html += `</div>`;
      }
    }
    if (res.preview_url) {
      html += `<img class="preview-img" src="${res.preview_url}" alt="预览"/>
        <div class="preview-cap">骨光融合预览（黄点=骨骼关键点，红/品红圆=光棒光斑）· 共 ${res.query_frames} 帧 · ${res.duration}s · ${res.used_skeleton ? '骨骼+光棒融合' : '纯光棒模式'}</div>`;
    }
    resultBox.innerHTML = html;
    submitBtn.disabled = false;
  }

  function bilibiliUrl(s) {
    if (/^https?:\/\//i.test(s)) return s;
    if (/^BV/i.test(s)) return `https://www.bilibili.com/video/${s}`;
    return `https://www.bilibili.com/video/${s}`;
  }
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
})();
