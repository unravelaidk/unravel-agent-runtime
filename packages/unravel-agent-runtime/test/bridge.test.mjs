import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { once } from 'node:events';
import { run, createSession, AgentRunError } from '../dist/index.js';

test('Rust agent runs a JavaScript tool and returns resumable session history', async () => {
  const requests = [];
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    requests.push(JSON.parse(Buffer.concat(chunks).toString()));
    response.setHeader('Content-Type', 'application/json');
    response.end(JSON.stringify(requests.length === 1 ? {
      choices: [{ message: { content: '', tool_calls: [{
        id: 'call-1', type: 'function', function: { name: 'greet', arguments: '{"name":"Ada"}' },
      }] }, finish_reason: 'tool_calls' }],
    } : { choices: [{ message: { content: 'Done.' }, finish_reason: 'stop' }] }));
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  try {
    const events = [];
    const result = await run({
      provider: { kind: 'custom', model: 'mock', baseUrl: `http://127.0.0.1:${server.address().port}/v1` },
      systemPrompt: 'Help the user.', prompt: 'Say hello', session: createSession('test'),
      onEvent: event => events.push(event.type),
      tools: [{ name: 'greet', description: 'Greeting', parameters: { type: 'object' },
        execute: ({ name }) => `Hello, ${name}!` }],
    });
    assert.equal(result.output, 'Done.');
    assert.equal(requests.length, 2);
    assert.equal(requests[1].messages.at(-1).content, 'Hello, Ada!');
    assert.equal(result.session.messages.at(-1).content, 'Done.');
    assert.ok(events.includes('tool_completed'));
  } finally { server.close(); }
});

test('Rust errors return session history to the caller', async () => {
  await assert.rejects(run({
    provider: { kind: 'custom', model: 'mock', baseUrl: 'http://127.0.0.1:1/v1' },
    systemPrompt: 'Help', prompt: 'Hi', session: createSession('failed'), maxTurns: 0,
  }), error => error instanceof AgentRunError && error.session?.messages.length === 2);
});
