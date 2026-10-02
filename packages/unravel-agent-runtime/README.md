# @unravelai/unravel-agent-runtime for Node.js

Run the Rust agent loop from Node.js with TypeScript types and JavaScript tool
callbacks. Each `run` launches a local Rust bridge process. The package
includes executables for Linux x64/arm64, macOS x64/arm64, and Windows x64;
Node.js 20 or later is required.

## Install and run

Install the npm package, then configure a provider and any tools you need.
You can supply `apiKey` directly or set the provider's API key environment
variable, such as `OPENAI_API_KEY`.

```sh
npm install @unravelai/unravel-agent-runtime@0.2.0
```

```ts
import { createSession, run } from '@unravelai/unravel-agent-runtime';

const result = await run({
  provider: { kind: 'openai', model: 'gpt-4o-mini' },
  systemPrompt: 'Help the user.',
  prompt: 'Greet Ada.',
  session: createSession('conversation-1'),
  tools: [{
    name: 'greet',
    description: 'Greet someone by name',
    parameters: {
      type: 'object',
      properties: { name: { type: 'string' } },
      required: ['name'],
    },
    execute: (arguments_) => `Hello, ${(arguments_ as { name: string }).name}!`,
  }],
});

console.log(result.output);
// Pass result.session to the next run in this conversation.
```

`run` returns the assistant's final text and the full session history. Pass
the returned session into a later `run` to continue the conversation. Tool
callbacks may be synchronous or asynchronous and return text or an object
with `content` and optional `metadata`. Tools execute sequentially by
default. Set `preferStreaming: true` to receive model deltas through
`onEvent`; the runtime only dispatches complete tool calls.

Version 0.2.0 keeps the Node callback contract text/metadata-only. Ephemeral
tool observations are available through the Rust API; returning an image from
a JavaScript tool does not implicitly forward it to the model.

For a local OpenAI-compatible endpoint, use `kind: 'custom'` with a `baseUrl`
that includes `/v1`. The `ollama`, `nvidia`, and `openrouter` kinds use the
built-in Rust provider settings.

When a run fails, `AgentRunError.session` contains the available conversation
history. Resolve any `tool_unknown` entries before resuming: an interrupted
tool may already have produced a side effect. An `AbortSignal` requests the
Rust loop to stop; the promise rejects with its current session when it stops.

## License

This package and the included Rust bridge are licensed under AGPL-3.0-only.
See `LICENSE`, `NOTICE`, and `PROVIDERS-NOTICE` in the package for attribution.
