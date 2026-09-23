import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
export type ProviderKind = 'openai' | 'nvidia' | 'openrouter' | 'ollama' | 'custom';
export interface ProviderOptions {
  kind: ProviderKind;
  model: string;
  apiKey?: string;
  baseUrl?: string;
}
export interface ToolOutput {
  content: string;
  metadata?: Json;
}
export interface Tool {
  name: string;
  description: string;
  parameters: Json;
  parallelSafe?: boolean;
  execute(arguments_: Json): Promise<string | ToolOutput> | string | ToolOutput;
}
export interface Message { role: string; [key: string]: unknown }
export interface Session { id: string; messages: Message[] }
export interface AgentEvent { type: string; [key: string]: unknown }
export interface RunOptions {
  provider: ProviderOptions;
  systemPrompt: string;
  prompt: string;
  session: Session;
  tools?: Tool[];
  maxTurns?: number;
  preferStreaming?: boolean;
  onEvent?: (event: AgentEvent) => void;
  signal?: AbortSignal;
}
export interface RunResult { output: string; session: Session }

/** Create a fresh conversation; pass the returned session into the next run. */
export function createSession(id: string): Session { return { id, messages: [] }; }

function binaryPath(): string {
  const platform = `${process.platform}-${process.arch}`;
  const supported = ['linux-x64', 'linux-arm64', 'darwin-x64', 'darwin-arm64', 'win32-x64'];
  if (!supported.includes(platform)) {
    throw new Error(`Unsupported platform ${platform}; supported: ${supported.join(', ')}`);
  }
  const filename = process.platform === 'win32' ? 'unravel-agent-node.exe' : 'unravel-agent-node';
  return fileURLToPath(new URL(`../bin/${platform}/${filename}`, import.meta.url));
}

export class AgentRunError extends Error {
  constructor(message: string, readonly session?: Session) {
    super(message);
    this.name = 'AgentRunError';
  }
}

/** Run the Rust agent loop with JavaScript tool callbacks and return its transcript. */
export async function run(options: RunOptions): Promise<RunResult> {
  if (options.signal?.aborted) throw new AgentRunError('Run aborted');
  const tools = new Map((options.tools ?? []).map(tool => [tool.name, tool]));
  if (tools.size !== (options.tools ?? []).length) throw new Error('Duplicate tool name');
  const child = spawn(binaryPath(), [], { stdio: ['pipe', 'pipe', 'pipe'] });
  // A tool may finish after the child exits; writing to its closed pipe is benign.
  child.stdin.on('error', () => {});
  // Only data sent over the pipe can contain credentials; stderr is diagnostics.
  let stderr = '';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk: string) => { stderr = (stderr + chunk).slice(-4096); });
  let finished = false;
  let currentSession: Session | undefined;
  const send = (value: object) => {
    if (!child.stdin.destroyed) child.stdin.write(JSON.stringify(value) + '\n');
  };
  const abort = () => send({ type: 'stop' });
  options.signal?.addEventListener('abort', abort, { once: true });
  const completion = new Promise<RunResult>((resolve, reject) => {
    const fail = (message: string, session?: Session) => {
      if (!finished) { finished = true; reject(new AgentRunError(message, session)); }
    };
    child.on('error', (error) => fail(error.message));
    child.on('close', (code) => fail(`Rust bridge exited (${code})${stderr ? ': ' + stderr : ''}`, currentSession));
    const lines = createInterface({ input: child.stdout });
    lines.on('line', (line) => {
      let data: Record<string, unknown>;
      try { data = JSON.parse(line); }
      catch { fail('Invalid response from Rust bridge'); child.kill(); return; }
      if (data.type === 'event') {
        try { options.onEvent?.(data.event as AgentEvent); }
        catch (error) { fail(String(error)); child.kill(); }
      } else if (data.type === 'tool_request') {
        const tool = tools.get(data.name as string);
        if (!tool) { send({ type: 'tool_result', id: data.id, error: 'Tool is not registered' }); return; }
        Promise.resolve().then(() => tool.execute(data.arguments as Json)).then(value => {
          const output = typeof value === 'string' ? { content: value } : value;
          send({ type: 'tool_result', id: data.id, content: output.content, metadata: output.metadata ?? null });
        }, error => send({ type: 'tool_result', id: data.id, error: String(error) }));
      } else if (data.type === 'result') {
        currentSession = data.session as Session;
        if (!finished) { finished = true; resolve({ output: data.output as string, session: currentSession }); }
      } else if (data.type === 'error') {
        currentSession = data.session as Session | undefined;
        fail(data.error as string, currentSession);
      }
    });
  });
  send({
    provider: options.provider, systemPrompt: options.systemPrompt, prompt: options.prompt,
    session: options.session, maxTurns: options.maxTurns, preferStreaming: options.preferStreaming,
    tools: (options.tools ?? []).map(({ name, description, parameters, parallelSafe }) =>
      ({ name, description, parameters, parallelSafe: parallelSafe ?? false })),
  });
  try { return await completion; }
  finally { options.signal?.removeEventListener('abort', abort); child.stdin.end(); }
}
