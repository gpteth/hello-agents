async function errorMessage(res, fallback) {
  const text = await res.text().catch(() => '')
  let detail = ''
  try {
    const data = JSON.parse(text)
    detail = data.detail || data.error || ''
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg || JSON.stringify(d)).join('; ')
  } catch {
    detail = ''
  }
  if (detail) return String(detail)
  if (res.status === 504) return `${fallback}：服务端处理超时（504），请把问题拆小后重试`
  if (res.status === 404) return `${fallback}：接口不存在（404），请确认后端已部署`
  const snippet = text.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 160)
  return `${fallback}（HTTP ${res.status}${snippet ? `：${snippet}` : ''}）`
}

export async function api(path, options = {}) {
  const res = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  if (!res.ok) {
    throw new Error(await errorMessage(res, '请求失败'))
  }
  return res.json().catch(() => ({}))
}

function parseSSE(block) {
  let event = 'message'
  const data = []
  for (const line of block.split('\n')) {
    if (!line || line.startsWith(':')) continue
    const sep = line.indexOf(':')
    const field = sep === -1 ? line : line.slice(0, sep)
    const value = sep === -1 ? '' : line.slice(sep + 1).replace(/^ /, '')
    if (field === 'event') event = value
    else if (field === 'data') data.push(value)
  }
  if (!data.length) return null
  try {
    return { type: event, ...JSON.parse(data.join('\n') || '{}') }
  } catch {
    return null
  }
}

async function syncChat(message, sessionId, onEvent, signal) {
  const res = await fetch('/api/chat/send/sync', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal,
  })
  if (!res.ok) throw new Error(await errorMessage(res, '对话请求失败'))
  const data = await res.json()
  if (data.session_id) onEvent({ type: 'session', session_id: data.session_id })
  if (data.error) {
    onEvent({ type: 'error', error: data.error })
    return
  }
  onEvent({ type: 'chunk', content: data.content || '' })
  onEvent({ type: 'done', content: data.content || '', session_id: data.session_id })
}

export async function streamChat(message, sessionId, onEvent, signal) {
  let res
  try {
    res = await fetch('/api/chat/send/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({ message, session_id: sessionId }),
      signal,
    })
  } catch (err) {
    if (err.name === 'AbortError') throw err
    // Nothing reached the server's agent yet, so retrying without streaming is safe.
    return syncChat(message, sessionId, onEvent, signal)
  }
  if (!res.ok) {
    throw new Error(await errorMessage(res, '流式请求失败'))
  }
  if (!res.body || !(res.headers.get('content-type') || '').includes('text/event-stream')) {
    return syncChat(message, sessionId, onEvent, signal)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let finished = false
  const dispatch = (block) => {
    const event = parseSSE(block)
    if (!event) return
    if (event.type === 'done' || event.type === 'error') finished = true
    onEvent(event)
  }
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n?/g, '\n')
      let index
      while ((index = buffer.indexOf('\n\n')) !== -1) {
        dispatch(buffer.slice(0, index))
        buffer = buffer.slice(index + 2)
      }
    }
    buffer += decoder.decode()
    if (buffer.trim()) dispatch(buffer)
  } catch (err) {
    if (err.name === 'AbortError') throw err
    if (!finished) throw new Error(`连接中断：${err.message || '网络错误'}，请重试`)
  }
  if (!finished) {
    throw new Error('回答未完成，连接已被服务端关闭（可能超过了函数最长执行时间），请把问题拆小后重试')
  }
}
