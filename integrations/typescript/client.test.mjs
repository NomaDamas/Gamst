import test from 'node:test';
import assert from 'node:assert/strict';
import { HumanSenseClient } from './client.mjs';

class Socket extends EventTarget {
  readyState = 0;
  sent = [];
  constructor() { super(); queueMicrotask(() => { this.readyState = 1; this.dispatchEvent(new Event('open')); }); }
  send(data) { this.sent.push(JSON.parse(data)); }
  receive(data) { this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(data) })); }
  close() { this.readyState = 3; this.dispatchEvent(new Event('close')); }
}

test('RPC responses match IDs while event notifications remain separate', async () => {
  const c = new HumanSenseClient('ws://test', { WebSocketImpl: Socket });
  await c.ready();
  const messages = [];
  c.onMessage(message => messages.push(message));
  const one = c.callTool('get_state');
  const two = c.callTool('get_context');
  await new Promise(resolve => setTimeout(resolve, 0));
  const [a, b] = c.socket.sent;
  c.socket.receive({ jsonrpc: '2.0', method: 'notifications/humansense/event', params: { event: 'speech.started' } });
  c.socket.receive({ jsonrpc: '2.0', id: b.id, result: { content: [{ type: 'text', text: '{"context":true}' }] } });
  c.socket.receive({ jsonrpc: '2.0', id: a.id, result: { content: [{ type: 'text', text: '{"state":true}' }] } });
  assert.deepEqual(await one, { state: true });
  assert.deepEqual(await two, { context: true });
  assert.equal(messages.length, 3);
  c.close();
});

test('tool failures and connection close reject callers', async () => {
  const c = new HumanSenseClient('ws://test', { WebSocketImpl: Socket });
  await c.ready();
  const failed = c.callTool('start_capture'); await new Promise(resolve => setTimeout(resolve, 0));
  c.socket.receive({ jsonrpc: '2.0', id: c.socket.sent[0].id, result: { isError: true, content: [{ type: 'text', text: 'consent required' }] } });
  await assert.rejects(failed, /consent required/);
  const pending = c.request('ping'); await new Promise(resolve => setTimeout(resolve, 0)); c.close();
  await assert.rejects(pending, /closed/);
});

test('timeout clears pending requests and malformed messages are ignored', async () => {
  const c = new HumanSenseClient('ws://test', { WebSocketImpl: Socket, timeoutMs: 10 });
  await c.ready();
  c.socket.dispatchEvent(new MessageEvent('message', { data: 'not-json' }));
  await assert.rejects(c.request('ping'), /timed out/);
  assert.equal(c.pending.size, 0); c.close();
});
