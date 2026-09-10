const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../app/web/app.js'), 'utf8');
const code = source.slice(source.indexOf('function streamAnswer('), source.indexOf('function connectEvents('));
const frames = new Map();
let frameId = 0, stream, finalRenders = 0, rendered;
const copy = {textContent: '', classList: {add() {}}};
const shell = {isConnected: true, querySelector() {return copy;}};
class EventSource {
  constructor() {this.handlers = {}; stream = this;}
  addEventListener(name, handler) {this.handlers[name] = handler;}
  close() {this.closed = true;}
  emit(name, value) {this.handlers[name]({data: JSON.stringify(value)});}
}
const context = vm.createContext({EventSource, Array, Promise,
  requestAnimationFrame(fn) {frames.set(++frameId, fn); return frameId;},
  cancelAnimationFrame(id) {frames.delete(id);},
  renderDraft(_, payload) {rendered = payload; if (payload.draft.content) finalRenders++;},
  setStatus() {}, scrollToBottom() {}, getDraft() {throw Error('Unexpected fallback');},
});
vm.runInContext(code, context);
const answer = '中文😀ABC'.repeat(20);
const promise = context.streamAnswer(shell, 'session', 'request');
stream.emit('answer_start', {draft: {content: '', title: 'Test'}});
stream.emit('answer_delta', {text: answer, offset: Array.from(answer).length});
stream.emit('answer_done', {});
assert.equal(finalRenders, 0, 'Network completion must not flush all text');
let now = 0;
function tick() {
  now += 16;
  const pending = [...frames.values()]; frames.clear();
  pending.forEach(fn => fn(now));
}
tick();
tick();
assert.ok(copy.textContent.length > 0 && copy.textContent.length < answer.length);
for (let i = 0; frames.size && i < 1000; i++) tick();
assert.equal(finalRenders, 1);
assert.equal(rendered.draft.content, answer);
assert.equal(copy.textContent, answer);
assert.ok(now > 1000, 'The answer should be visibly progressive');
assert.equal(context.streamAnswer(shell, 'session', 'request'), promise);
console.log('PASS: chunk burst is rendered progressively; completion waits; Unicode and deduplication preserved.');

async function testFallback() {
  const fallbackShell = {isConnected: true, querySelector() {return copy;}};
  copy.textContent = '';
  context.api = async () => ({draft: {content: answer, title: 'Fallback'}});
  context.showRunError = (_, message) => {throw Error(message);};
  const before = finalRenders;
  const pending = context.streamAnswer(fallbackShell, 'session', 'fallback');
  await stream.onerror();
  await stream.onerror();
  await stream.onerror();
  assert.equal(finalRenders, before, 'Fallback must not render the full answer immediately');
  tick(); tick();
  assert.ok(copy.textContent.length > 0 && copy.textContent.length < answer.length);
  for (let i = 0; frames.size && i < 1000; i++) tick();
  await pending;
  assert.equal(rendered.draft.content, answer);
  assert.equal(finalRenders, before + 1);
  console.log('PASS: repeated SSE failure also renders fallback progressively.');
}
testFallback().catch(error => {console.error(error); process.exitCode = 1;});
