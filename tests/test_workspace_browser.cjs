const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('app/web/app.js', 'utf8');
const stored = new Map();
let value = 'bad json';
const context = vm.createContext({
  sessionStorage: { getItem: key => stored.get(key) ?? value, setItem: (key, v) => stored.set(key, v) },
  renderConversationList() {},
});
vm.runInContext(source.slice(0, source.indexOf('const LOCAL_DEMO_SCOPE')), context);
vm.runInContext('loadConversations()', context);
assert.equal(vm.runInContext('state.conversations.length', context), 0);
value = '{"not":"an array"}';
vm.runInContext('loadConversations()', context);
assert.equal(vm.runInContext('state.conversations.length', context), 0);
value = '[null,{"sessionId":"s","title":"Title"},{"title":17}]';
vm.runInContext('loadConversations()', context);
assert.equal(vm.runInContext('state.conversations.length', context), 1);
stored.set('easyteaching-conversations:alice', '[{"sessionId":"alice-session","title":"Alice"}]');
stored.set('easyteaching-conversations:bob', '[]');
vm.runInContext('state.user = {user_id:"alice"}; loadConversations()', context);
assert.equal(vm.runInContext('state.conversations[0].sessionId', context), 'alice-session');
vm.runInContext('state.user = {user_id:"bob"}; loadConversations()', context);
assert.equal(vm.runInContext('state.conversations.length', context), 0);
context.sessionStorage.getItem = () => { throw Error('Storage blocked'); };
vm.runInContext('loadConversations()', context);
assert.equal(vm.runInContext('state.conversations.length', context), 0);
console.log('PASS: damaged or blocked browser storage cannot prevent startup; account histories are isolated.');
