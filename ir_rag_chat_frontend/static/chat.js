const chatListEl = document.getElementById('chatList');
const messagesEl = document.getElementById('messages');
const chatMetaEl = document.getElementById('chatMeta');
const questionInputEl = document.getElementById('questionInput');
const submitBtnEl = document.getElementById('submitBtn');
const newChatBtnEl = document.getElementById('newChatBtn');
const clearChatBtnEl = document.getElementById('clearChatBtn');
const saveChatBtnEl = document.getElementById('saveChatBtn');

let currentChatId = null;

function escapeHtml(text) {
  return (text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function setMeta(text) {
  chatMetaEl.textContent = text;
}

function autoGrow() {
  questionInputEl.style.height = 'auto';
  questionInputEl.style.height = `${questionInputEl.scrollHeight}px`;
}
questionInputEl.addEventListener('input', autoGrow);

autoGrow();

function formatTime(text) {
  if (!text) return '';
  return text.replace('T', ' ');
}

async function requestJson(url, options = {}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || '请求失败');
  }
  return res.json();
}

function renderUserMessage(content) {
  const node = document.getElementById('userMessageTpl').content.firstElementChild.cloneNode(true);
  node.querySelector('.bubble').textContent = content;
  messagesEl.appendChild(node);
}

function renderAssistantMessage(payload) {
  const node = document.getElementById('assistantMessageTpl').content.firstElementChild.cloneNode(true);
  const answerEl = node.querySelector('.answer');
  const refsEl = node.querySelector('.references');
  const imagesEl = node.querySelector('.images');
  const debugEl = node.querySelector('.debug');

  answerEl.textContent = payload.answer || '未返回 answer';

  const citedDocs = payload.cited_docs || [];
  if (citedDocs.length) {
    refsEl.innerHTML = `<div class="section-title">引用文本如下</div>`;
    citedDocs.forEach((doc, index) => {
      const meta = doc.metadata || {};
      const docNo = doc.rank ?? index + 1;
      const pageNo = meta.orig_page_no ?? meta.page_no ?? '未知';
      const source = meta.source || '';
      const div = document.createElement('div');
      div.className = 'ref-card';
      div.innerHTML = `
        <div class="ref-doc-id">文档编号：${escapeHtml(String(docNo))}</div>
        <div class="ref-meta">页码：${escapeHtml(String(pageNo))}${source ? ` ｜ 来源：${escapeHtml(source)}` : ''}</div>
        <div>${escapeHtml(doc.page_content || '')}</div>
      `;
      refsEl.appendChild(div);
    });
  }

  const relatedImages = payload.related_images || [];
  if (relatedImages.length) {
    const grid = document.createElement('div');
    grid.className = 'image-grid';
    const title = document.createElement('div');
    title.className = 'section-title';
    title.textContent = '图表信息';
    imagesEl.appendChild(title);

    relatedImages.forEach((img) => {
      if (!img.image_path) return;
      const card = document.createElement('div');
      card.className = 'image-card';
      const src = `/api/images?path=${encodeURIComponent(img.image_path)}`;
      card.innerHTML = `
        <img src="${src}" alt="图表图片" loading="lazy" />
        <div class="caption">
          ${escapeHtml(img.caption_label || '')}
          ${img.caption_text ? `<div>${escapeHtml(img.caption_text)}</div>` : ''}
          <div>路径：${escapeHtml(img.image_path || '')}</div>
        </div>
      `;
      grid.appendChild(card);
    });
    imagesEl.appendChild(grid);
  }

  if (payload.debug) {
    debugEl.textContent = `query chars=${payload.debug.query_chars} ｜ context chars=${payload.debug.context_chars} ｜ first visible token latency=${payload.debug.first_visible_token_latency ?? '-'}s ｜ llm total time=${payload.debug.llm_total_time}s`;
  }

  messagesEl.appendChild(node);
}

function renderChat(chat) {
  messagesEl.innerHTML = '';
  currentChatId = chat.id;
  setMeta(`聊天ID：${chat.id} ｜ 创建时间：${formatTime(chat.created_at)} ｜ 更新时间：${formatTime(chat.updated_at)}`);

  if (!chat.messages || chat.messages.length === 0) {
    messagesEl.innerHTML = '<div class="empty-state">开始一个新问题吧。</div>';
    return;
  }

  chat.messages.forEach((msg) => {
    if (msg.role === 'user') {
      renderUserMessage(msg.content || '');
    } else if (msg.role === 'assistant') {
      renderAssistantMessage(msg.payload || { answer: msg.content || '' });
    }
  });
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function refreshChatList(selectId = null) {
  const chats = await requestJson('/api/chats');
  chatListEl.innerHTML = '';

  chats.forEach((chat) => {
    const btn = document.createElement('button');
    btn.className = 'chat-item';
    if (chat.id === selectId) btn.classList.add('active');
    btn.innerHTML = `
      <div class="chat-item-title">${escapeHtml(chat.title || '新对话')}</div>
      <div class="chat-item-meta">${escapeHtml(formatTime(chat.updated_at) || '')} ｜ ${chat.message_count || 0} 条消息</div>
    `;
    btn.addEventListener('click', async () => {
      const data = await requestJson(`/api/chats/${chat.id}`);
      await refreshChatList(chat.id);
      renderChat(data);
    });
    chatListEl.appendChild(btn);
  });
}

async function createChat() {
  const chat = await requestJson('/api/chats', { method: 'POST', body: '{}' });
  await refreshChatList(chat.id);
  renderChat(chat);
}

async function ensureChat() {
  if (currentChatId) return currentChatId;
  const chat = await requestJson('/api/chats', { method: 'POST', body: '{}' });
  currentChatId = chat.id;
  await refreshChatList(chat.id);
  renderChat(chat);
  return currentChatId;
}

async function askQuestion() {
  const query = questionInputEl.value.trim();
  if (!query) return;
  const chatId = await ensureChat();

  if (messagesEl.querySelector('.empty-state')) {
    messagesEl.innerHTML = '';
  }
  renderUserMessage(query);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  questionInputEl.value = '';
  autoGrow();
  submitBtnEl.disabled = true;
  submitBtnEl.textContent = '提交中...';

  try {
    const payload = await requestJson(`/api/chats/${chatId}/ask`, {
      method: 'POST',
      body: JSON.stringify({ query }),
    });
    renderAssistantMessage(payload);
    const chat = await requestJson(`/api/chats/${chatId}`);
    renderChat(chat);
    await refreshChatList(chatId);
  } catch (err) {
    renderAssistantMessage({ answer: `请求失败：${err.message}` });
  } finally {
    submitBtnEl.disabled = false;
    submitBtnEl.textContent = '提交';
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

submitBtnEl.addEventListener('click', askQuestion);
questionInputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    askQuestion();
  }
});
newChatBtnEl.addEventListener('click', createChat);
clearChatBtnEl.addEventListener('click', async () => {
  if (!currentChatId) return;
  const chat = await requestJson(`/api/chats/${currentChatId}/clear`, { method: 'POST', body: '{}' });
  renderChat(chat);
  await refreshChatList(currentChatId);
});
saveChatBtnEl.addEventListener('click', async () => {
  if (!currentChatId) return;
  const resp = await requestJson(`/api/chats/${currentChatId}/save`, { method: 'POST', body: '{}' });
  alert(`已保存到：${resp.path}`);
  const chat = await requestJson(`/api/chats/${currentChatId}`);
  renderChat(chat);
  await refreshChatList(currentChatId);
});

(async function init() {
  const chats = await requestJson('/api/chats');
  if (chats.length) {
    const first = await requestJson(`/api/chats/${chats[0].id}`);
    await refreshChatList(chats[0].id);
    renderChat(first);
  } else {
    await createChat();
  }
})();
