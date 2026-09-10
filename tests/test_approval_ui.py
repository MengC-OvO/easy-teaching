"""Execute approval UI logic without a browser or backend side effects."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("terminal", ["completed", "failed"])
def test_approval_ui_waits_for_execution_before_showing_success(terminal):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the frontend behavior check")
    source = str(Path("app/web/app.js").resolve())
    script = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const code = fs.readFileSync(process.argv[1], 'utf8');
const functionCode = code.slice(code.indexOf('async function submitApproval('), code.indexOf('function showRunError('));
const terminal = process.argv[2], toasts = [], renders = [], buttons = [{disabled:false}];
const progress = {textContent:''};
const shell = {dataset:{sessionId:'s',requestId:'r'},isConnected:true,
  querySelector:()=>({querySelector:()=>progress,querySelectorAll:()=>buttons})};
let calls=0;
const context={setBusy:()=>{},setStatus:()=>{},showToast:t=>toasts.push(t),
  renderDraft:(_,payload)=>renders.push(payload),getDraft:()=>{throw Error('premature completion');},
  setTimeout:resolve=>{assert.equal(toasts.length,0);resolve();},
  api:async()=>{calls++;assert.equal(toasts.length,0);return {status:calls<3?'running':terminal};}};
vm.createContext(context);vm.runInContext(functionCode,context);
(async()=>{
  await context.submitApproval(shell,'approve');
  assert.equal(calls,3);assert.equal(renders.length,1);
  assert.equal(renders[0].status,terminal);
  assert.equal(toasts.length,terminal==='completed'?1:0);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    subprocess.run([node, "-e", script, source, terminal], check=True, capture_output=True, text=True)
