export type OutputType = 'event' | 'signal' | 'inference';
export interface ClientOptions {
  WebSocketImpl?: typeof WebSocket;
  timeoutMs?: number;
}
export declare class HumanSenseClient {
  readonly socket: WebSocket;
  constructor(url?: string, options?: ClientOptions);
  ready(): Promise<void>;
  send(message: Record<string, unknown>): void;
  request(method: string, params?: Record<string, unknown>): Promise<unknown>;
  callTool<T = unknown>(name: string, args?: Record<string, unknown>): Promise<T>;
  subscribe(types?: OutputType[]): Promise<{ types: OutputType[] }>;
  onMessage(callback: (message: unknown) => void): () => void;
  close(): void;
}
