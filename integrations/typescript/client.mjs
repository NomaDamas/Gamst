/** Browser-compatible HumanSense WebSocket RPC client. No build dependencies. */
export class HumanSenseClient {
  constructor(url = 'ws://127.0.0.1:8765', { WebSocketImpl = globalThis.WebSocket, timeoutMs = 30000 } = {}) {
    if (!WebSocketImpl) throw new Error('A WebSocket implementation is required');
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('timeoutMs must be positive');
    this.socket = new WebSocketImpl(url);
    this.timeoutMs = timeoutMs;
    this.nextId = 1;
    this.pending = new Map();
    this.listeners = new Set();
    this.readyPromise = new Promise((resolve, reject) => {
      this.socket.addEventListener('open', resolve, { once: true });
      this.socket.addEventListener('error', () => reject(new Error('WebSocket connection failed')), { once: true });
      this.socket.addEventListener('close', () => reject(new Error('WebSocket closed before opening')), { once: true });
    });
    // Consumers may choose to attach ready() later; avoid unhandled rejections.
    this.readyPromise.catch(() => {});
    this.socket.addEventListener('message', event => this.receive(event));
    this.socket.addEventListener('close', () => {
      for (const task of this.pending.values()) {
        clearTimeout(task.timer); task.reject(new Error('WebSocket closed'));
      }
      this.pending.clear();
    });
  }
  async ready() { await this.readyPromise; }
  send(message) {
    if (this.socket.readyState !== 1) throw new Error('WebSocket is not open');
    this.socket.send(JSON.stringify(message));
  }
  receive(event) {
    let message;
    try { message = JSON.parse(String(event.data)); } catch { return; }
    if (!message || typeof message !== 'object') return;
    if (Object.hasOwn(message, 'id') && this.pending.has(message.id)) {
      const task = this.pending.get(message.id);
      this.pending.delete(message.id); clearTimeout(task.timer);
      if (message.error) task.reject(new Error(message.error.message || 'RPC failed'));
      else task.resolve(message.result);
    }
    for (const listener of this.listeners) {
      try { listener(message); } catch { /* One observer cannot break other subscribers. */ }
    }
  }
  async request(method, params = {}) {
    await this.ready();
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id); reject(new Error('HumanSense RPC timed out'));
      }, this.timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      try { this.send({ jsonrpc: '2.0', id, method, params }); }
      catch (error) { clearTimeout(timer); this.pending.delete(id); reject(error); }
    });
  }
  async callTool(name, args = {}) {
    const result = await this.request('tools/call', { name, arguments: args });
    const content = result?.content?.find(item => item.type === 'text')?.text;
    if (result?.isError) throw new Error(content || 'Tool failed');
    if (typeof content !== 'string') return result;
    try { return JSON.parse(content); } catch { return content; }
  }
  subscribe(types = ['event']) { return this.request('humansense/subscribe', { types }); }
  onMessage(callback) { this.listeners.add(callback); return () => this.listeners.delete(callback); }
  close() { this.socket.close(); }
}
