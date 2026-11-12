# Browser / TypeScript client

Use the build-free ES module `client.mjs`, with TypeScript declarations `client.d.mts` (or the typed `client.ts` re-export).

```js
import { HumanSenseClient } from './client.mjs';
const client = new HumanSenseClient('ws://127.0.0.1:8765');
await client.ready();
client.onMessage(message => {
  if (message.method?.startsWith('notifications/humansense/')) console.log(message.params);
});
await client.subscribe(['event', 'inference']);
const result = await client.callTool('push_text', { text: 'Please clarify' });
const state = await client.callTool('get_state');
client.close();
```

Start `python -m humansense.server --origin http://localhost:3000` for that exact browser app origin. Default subscriptions deliver events only; `humansense/subscribe` is a HumanSense WebSocket extension, not an MCP protocol method. Signal/inference subscriptions are opt-in and use bounded runtime queues. The promise client matches response IDs, separates notifications, rejects tool/RPC failures and closes/times out pending requests.

Run `node integrations/typescript/client.test.mjs`. Tests use an in-process fake WebSocket to verify response ordering, event delivery, errors, close and timeouts. Actual network transport and TypeScript compiler checks remain unverified in the restricted development environment.
