/* 网文作品资源配置助手 —— 零构建前端。
 *
 * 用原生 SSE 而不是引框架：这个页面的复杂度全在「把 Agent 的多步推理展示清楚」上，
 * 而不在状态管理上。引 React 反而要多一层构建链路。
 */

const SESSION_KEY = 'lcc_session_id';
let sessionId = localStorage.getItem(SESSION_KEY);
if (!sessionId) {
  sessionId = 'web-' + Math.random().toString(36).slice(2, 10);
  localStorage.setItem(SESSION_KEY, sessionId);
}

const $ = (id) => document.getElementById(id);
const chat = $('chat');
const traceBody = $('traceBody');

const EXAMPLES = [
  { tag: '解释', q: '加更活动要怎么才能开起来？' },
  { tag: '解释', q: '保底分成的书，作者分成给 60% 行吗' },
  { tag: '推荐', q: '给 W1001 的新书 B001 配一套冷启动方案' },
  { tag: '优化', q: 'B003 想冲榜，在现有配置上怎么调' },
  { tag: '优化', q: '把 B004 的付费起点改到第 30 章，再开个完本奖' },
  { tag: '解释', q: '作者问为什么这个月没拿到全勤奖' },
];

// ---------------------------------------------------------------- 初始化
EXAMPLES.forEach(({ tag, q }) => {
  const b = document.createElement('button');
  b.innerHTML = `<span class="tag">${tag}</span>${q}`;
  b.onclick = () => { $('input').value = q; send(); };
  $('examples').appendChild(b);
});

fetch('/api/health').then(r => r.json()).then(h => {
  const b = h.backends;
  // 把降级状态显式标出来 —— 降级是静默发生的，不标出来很容易
  // 出现「以为在用 Neo4j，其实跑的是内存图」这种误判
  const pill = (v, fallbackVals) =>
    `<span class="pill ${fallbackVals.includes(v) ? 'fallback' : ''}">${v}</span>`;
  $('health').innerHTML = `
    <div>向量 ${pill(b.vector, ['numpy'])}　关键词 ${pill(b.keyword, ['bm25'])}</div>
    <div>图谱 ${pill(b.graph, ['networkx'])}　会话 ${pill(b.session, ['inproc'])}</div>
    <div>Embedding ${pill(b.embedding.backend, ['tfidf'])} <b>${b.embedding.dim}</b>d</div>
    <div>LLM ${pill(b.llm.provider, ['mock'])} <b>${b.llm.model}</b></div>
    <div style="margin-top:6px">索引：<b>${h.index.chunks}</b> chunk ·
      <b>${h.index.entities}</b> 实体 · <b>${h.index.config_items}</b> 配置项</div>`;
}).catch(() => { $('health').textContent = '后端未就绪'; });

$('resetBtn').onclick = async () => {
  await fetch(`/api/session/${sessionId}`, { method: 'DELETE' });
  chat.innerHTML = '';
  traceBody.innerHTML = '<div class="empty">会话已清空。</div>';
};

const input = $('input');
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 160) + 'px';
});
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
$('composer').onsubmit = (e) => { e.preventDefault(); send(); };

// ---------------------------------------------------------------- 发送
let busy = false;

async function send() {
  const text = input.value.trim();
  if (!text || busy) return;
  busy = true;
  $('sendBtn').disabled = true;
  input.value = '';
  input.style.height = 'auto';

  addMsg('user', text);
  traceBody.innerHTML = '';
  const bot = addMsg('bot', '');
  const body = bot.querySelector('.bubble');
  body.innerHTML = '<div class="typing">正在识别意图…</div>';

  let answer = '';
  try {
    const resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        message: text,
        retrieval_mode: $('mode').value,
        use_agent: $('useAgent').checked,
      }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      // SSE 以空行分隔事件；最后一段可能不完整，留在缓冲区等下次
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const raw of parts) {
        const evt = parseEvent(raw);
        if (!evt) continue;
        if (evt.event === 'token') {
          answer += evt.data.text;
          body.innerHTML = renderMarkdown(answer);
          chat.scrollTop = chat.scrollHeight;
        } else {
          handleEvent(evt, bot, body);
        }
      }
    }
  } catch (err) {
    body.innerHTML = `<div class="verdict bad">请求失败：${err.message}</div>`;
  } finally {
    busy = false;
    $('sendBtn').disabled = false;
    if (!answer) body.querySelector('.typing')?.remove();
  }
}

function parseEvent(raw) {
  let event = null, data = null;
  for (const line of raw.split('\n')) {
    if (line.startsWith('event: ')) event = line.slice(7).trim();
    else if (line.startsWith('data: ')) {
      try { data = JSON.parse(line.slice(6)); } catch { /* 分片，忽略 */ }
    }
  }
  return event && data !== null ? { event, data } : null;
}

function handleEvent(evt, bot, body) {
  switch (evt.event) {
    case 'intent': {
      const i = evt.data.intent;
      const who = bot.querySelector('.who');
      who.innerHTML = `助手 <span class="badge ${i}">${intentLabel(i)}</span>` +
        `<span style="color:#5d6472">置信 ${(evt.data.confidence * 100).toFixed(0)}%` +
        (evt.data.fallback_used ? ' · 低置信兜底' : '') + '</span>';
      body.innerHTML = '<div class="typing">正在检索领域知识…</div>';
      break;
    }
    case 'followup': {
      // 追问识别结果：是追问才带上相关的历史轮次
      const who = bot.querySelector('.who');
      const tip = evt.data.is_followup
        ? `追问（${(evt.data.reasons || []).join('、')}）` : '新话题';
      who.insertAdjacentHTML('beforeend', `<span style="color:#5d6472"> · ${tip}</span>`);
      break;
    }
    case 'retrieval':
      renderRetrieval(evt.data);
      body.innerHTML = '<div class="typing">正在推理…</div>';
      break;
    case 'agent_step':
      renderAgentStep(evt.data);
      break;
    case 'validation':
      renderValidation(evt.data);
      break;
    case 'error':
      body.innerHTML = `<div class="verdict bad">服务异常：${evt.data.message}</div>`;
      break;
  }
}

function intentLabel(i) {
  return { explain: '解释类', recommend: '推荐类', optimize: '优化类',
           out_of_scope: '越界' }[i] || i;
}

// ---------------------------------------------------------------- 轨迹渲染
function renderRetrieval(d) {
  const sec = document.createElement('div');
  sec.className = 'sec';
  const lat = d.latency_ms || {};

  let html = `<h3>① 实体链接 → 图谱扩展</h3><div>`;
  html += d.seed_entities.map(e =>
    `<span class="chip seed">${e.name} <b>${e.score.toFixed(2)}</b></span>`).join('') ||
    '<span class="chip">未命中种子实体（本轮退化为 hybrid 召回）</span>';
  html += '</div>';

  if (d.expanded?.length) {
    html += `<div style="margin-top:6px">`;
    html += d.expanded.map(e =>
      `<span class="chip hop${e.hop}">${e.name}<sup>${e.hop}跳</sup></span>`).join('');
    html += `</div>`;
  }
  html += `<div class="m" style="color:#5d6472;font-size:10.5px;margin-top:5px">
    子图 ${d.subgraph_size} 节点 · ${d.triples.length} 条关系` +
    (d.pruned_count ? ` · 上下文缓存剪掉 ${d.pruned_count} 个已扩展实体` : '') +
    ` · ${(lat.stage1_graph || 0).toFixed(0)}ms</div>`;

  if (d.triples?.length) {
    html += `<h3 style="margin-top:12px">② 注入上下文的关系三元组</h3>`;
    html += d.triples.map(t => `<div class="triple">${escapeHtml(t)}</div>`).join('');
  }

  html += `<h3 style="margin-top:12px">③ 混合检索命中文本段</h3>`;
  html += d.chunks.map(c => `
    <div class="doc">
      <div class="t">${escapeHtml(c.section || c.doc)}</div>
      <div class="m"><span class="src">${c.source}</span>
        score ${c.score.toFixed(3)} · ${escapeHtml(c.preview)}…</div>
    </div>`).join('');
  html += `<div style="color:#5d6472;font-size:10.5px">检索总耗时 ${(lat.total || 0).toFixed(0)}ms</div>`;

  sec.innerHTML = html;
  traceBody.appendChild(sec);
}

function renderAgentStep(s) {
  const d = document.createElement('details');
  d.className = 'step' + (s.ok === false ? ' fail' : '');
  d.open = false;
  d.innerHTML = `
    <summary>
      <span class="n">${s.step}</span>
      <span class="tool">${s.tool || 'final_answer'}</span>
      <span class="ms">${s.latency_ms ? s.latency_ms.toFixed(0) + 'ms' : ''}</span>
    </summary>
    <div class="body">
      ${s.thought ? `<div class="thought">${escapeHtml(s.thought)}</div>` : ''}
      ${s.arguments && Object.keys(s.arguments).length
        ? `<pre>${escapeHtml(JSON.stringify(s.arguments, null, 2))}</pre>` : ''}
      ${s.observation ? `<pre style="margin-top:6px">${escapeHtml(s.observation)}</pre>` : ''}
    </div>`;
  traceBody.appendChild(d);
  traceBody.scrollTop = traceBody.scrollHeight;
}

function renderValidation(v) {
  const el = document.createElement('div');
  el.className = 'sec';
  const ok = v.passed && !(v.violations || []).length;
  el.innerHTML = `<h3>④ 规则引擎兜底校验</h3>
    <div class="verdict ${ok ? 'ok' : 'bad'}">` +
    (ok ? '✅ 依赖链完整、无互斥冲突、R001~R004 全部满足'
        : (v.violations || []).map(x =>
            `${x.severity === 'error' ? '❌' : '⚠️'} [${x.rule_id}] ${escapeHtml(x.message)}` +
            (x.suggestion ? `<br><span style="opacity:.8">建议：${escapeHtml(x.suggestion)}</span>` : '')
          ).join('<hr style="border-color:#3a2226;margin:6px 0">')) +
    `</div>`;
  traceBody.appendChild(el);
}

// ---------------------------------------------------------------- 工具
function addMsg(role, text) {
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  el.innerHTML = `<div class="who">${role === 'user' ? '我' : '助手'}</div>
                  <div class="bubble">${role === 'user' ? escapeHtml(text) : ''}</div>`;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"]/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

/* 极简 markdown 渲染。只支持助手实际会输出的子集（标题/表格/列表/强调/代码/引用），
 * 不引 marked.js —— 一个 40KB 的依赖换这几个语法不划算，而且输出格式是我们
 * 自己在提示词里规定的，可控。 */
function renderMarkdown(md) {
  const lines = escapeHtml(md).split('\n');
  const out = [];
  let inTable = false, inList = null;

  const closeList = () => { if (inList) { out.push(`</${inList}>`); inList = null; } };
  const closeTable = () => { if (inTable) { out.push('</tbody></table>'); inTable = false; } };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (/^\s*\|(.+)\|\s*$/.test(line)) {
      const cells = line.trim().slice(1, -1).split('|').map(c => c.trim());
      if (/^[-:\s|]+$/.test(line.replace(/\|/g, ''))) continue;  // 分隔行
      if (!inTable) {
        closeList();
        out.push('<table><thead><tr>' + cells.map(c => `<th>${inline(c)}</th>`).join('') +
                 '</tr></thead><tbody>');
        inTable = true;
      } else {
        out.push('<tr>' + cells.map(c => `<td>${inline(c)}</td>`).join('') + '</tr>');
      }
      continue;
    }
    closeTable();

    let m;
    if ((m = line.match(/^(#{2,4})\s+(.*)$/))) {
      closeList();
      const lvl = Math.min(m[1].length, 3);
      out.push(`<h${lvl}>${inline(m[2])}</h${lvl}>`);
    } else if (/^\s*[-*]\s+/.test(line)) {
      if (inList !== 'ul') { closeList(); out.push('<ul>'); inList = 'ul'; }
      out.push(`<li>${inline(line.replace(/^\s*[-*]\s+/, ''))}</li>`);
    } else if (/^\s*\d+\.\s+/.test(line)) {
      if (inList !== 'ol') { closeList(); out.push('<ol>'); inList = 'ol'; }
      out.push(`<li>${inline(line.replace(/^\s*\d+\.\s+/, ''))}</li>`);
    } else if (/^\s*>\s?/.test(line)) {
      closeList();
      out.push(`<blockquote>${inline(line.replace(/^\s*>\s?/, ''))}</blockquote>`);
    } else if (/^\s*---+\s*$/.test(line)) {
      closeList(); out.push('<hr>');
    } else if (line.trim() === '') {
      closeList();
    } else {
      closeList();
      out.push(`<p>${inline(line)}</p>`);
    }
  }
  closeList(); closeTable();
  return out.join('');
}

function inline(s) {
  return s
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}
