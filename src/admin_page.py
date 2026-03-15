"""Admin dashboard HTML generator.

Follows the same pattern as ``landing_page.py``: a single function
returns a self-contained HTML string with inline CSS/JS and CDN
dependencies (Alpine.js, Pico CSS, CodeMirror).
"""


def build_admin_page() -> str:
    """Build the admin dashboard HTML."""
    return """<!DOCTYPE html>
<html lang="ko" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Admin - Claude Code Gateway</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2.0.6/css/pico.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.8/dist/cdn.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/codemirror@5.65.18/lib/codemirror.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/codemirror@5.65.18/theme/material-darker.min.css">
<script src="https://cdn.jsdelivr.net/npm/codemirror@5.65.18/lib/codemirror.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/codemirror@5.65.18/mode/markdown/markdown.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/codemirror@5.65.18/mode/javascript/javascript.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/codemirror@5.65.18/mode/yaml/yaml.min.js"></script>
<style>
:root { --accent: #16a34a; }
[data-theme="dark"] {
  --card-bg: #1e293b; --subtle-bg: #334155; --border: #475569; --page-bg: #0f172a;
  --text: #e2e8f0; --text-muted: #94a3b8;
}
[data-theme="light"] {
  --card-bg: #fff; --subtle-bg: #f1f5f9; --border: #e2e8f0; --page-bg: #f8fafc;
  --text: #1e293b; --text-muted: #64748b;
}
body { background: var(--page-bg); font-size: 14px; }
.container { max-width: 1200px; margin: 0 auto; padding: 1rem; }
.card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem; }
.card h3 { margin-top: 0; font-size: 1rem; color: var(--text-muted); }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }
.stat { text-align: center; }
.stat .value { font-size: 2rem; font-weight: bold; color: var(--accent); }
.stat .label { font-size: 0.85rem; color: var(--text-muted); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
.badge-ok { background: #16a34a22; color: #16a34a; }
.badge-warn { background: #f59e0b22; color: #f59e0b; }
.badge-err { background: #ef444422; color: #ef4444; }
nav.tabs { display: flex; gap: 0; border-bottom: 2px solid var(--border); margin-bottom: 1rem; }
nav.tabs button { background: none; border: none; padding: 0.75rem 1.25rem; color: var(--text-muted);
  cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -2px; font-size: 0.9rem; }
nav.tabs button.active { color: var(--accent); border-bottom-color: var(--accent); }
nav.tabs button:hover { color: var(--text); }
.sidebar { display: flex; gap: 1rem; }
.sidebar .file-tree { width: 260px; flex-shrink: 0; }
.sidebar .editor-area { flex: 1; min-width: 0; }
.file-item { padding: 6px 12px; cursor: pointer; border-radius: 4px; font-size: 0.85rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.file-item:hover { background: var(--subtle-bg); }
.file-item.active { background: var(--accent); color: #fff; }
.file-item .icon { margin-right: 6px; }
.CodeMirror { height: auto; min-height: 300px; max-height: 70vh; border: 1px solid var(--border); border-radius: 4px; font-size: 13px; }
.editor-toolbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
.editor-toolbar .path { font-family: monospace; font-size: 0.85rem; color: var(--text-muted); }
.btn { display: inline-block; padding: 6px 16px; border-radius: 6px; border: none;
  cursor: pointer; font-size: 0.85rem; font-weight: 500; }
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover { opacity: 0.9; }
.btn-sm { padding: 4px 10px; font-size: 0.8rem; }
.btn-ghost { background: transparent; border: 1px solid var(--border); color: var(--text); }
.login-box { max-width: 400px; margin: 4rem auto; }
table { width: 100%; font-size: 0.85rem; }
table th { color: var(--text-muted); font-weight: 600; text-align: left; }
table td, table th { padding: 8px 12px; border-bottom: 1px solid var(--border); }
.toast { position: fixed; bottom: 1.5rem; right: 1.5rem; padding: 12px 20px; border-radius: 8px;
  font-size: 0.85rem; z-index: 100; transition: opacity 0.3s; }
.toast-ok { background: #16a34a; color: #fff; }
.toast-err { background: #ef4444; color: #fff; }
.dirty-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #f59e0b; margin-left: 6px; }
.config-key { font-family: monospace; font-size: 0.8rem; color: var(--accent); }
.redacted { color: var(--text-muted); font-style: italic; }
@media (max-width: 768px) { .grid-2, .grid-3 { grid-template-columns: 1fr; } .sidebar { flex-direction: column; } .sidebar .file-tree { width: 100%; } }
</style>
</head>
<body>

<div x-data="adminApp()" x-init="init()" class="container">

  <!-- Toast -->
  <div x-show="toast.show" x-transition.opacity :class="'toast toast-' + toast.type" x-text="toast.msg"></div>

  <!-- Login -->
  <template x-if="!authenticated">
    <div class="login-box card">
      <h2 style="margin-top:0">Admin Login</h2>
      <p style="color:var(--text-muted)">ADMIN_API_KEY required</p>
      <form @submit.prevent="doLogin()">
        <input type="password" x-model="loginKey" placeholder="Admin API Key"
          style="width:100%; margin-bottom:1rem" required>
        <button class="btn btn-primary" style="width:100%" type="submit">Login</button>
      </form>
      <p x-show="loginError" style="color:#ef4444; margin-top:0.5rem; font-size:0.85rem" x-text="loginError"></p>
    </div>
  </template>

  <!-- Main UI -->
  <template x-if="authenticated">
    <div>
      <!-- Header -->
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem">
        <h1 style="margin:0; font-size:1.5rem">Claude Code Gateway Admin</h1>
        <div style="display:flex; gap:0.5rem; align-items:center">
          <button class="btn btn-sm btn-ghost" @click="refreshAll()">Refresh</button>
          <button class="btn btn-sm btn-ghost" @click="toggleTheme()">Theme</button>
          <button class="btn btn-sm btn-ghost" @click="doLogout()">Logout</button>
        </div>
      </div>

      <!-- Tabs -->
      <nav class="tabs">
        <button :class="{ active: tab === 'dashboard' }" @click="tab='dashboard'">Dashboard</button>
        <button :class="{ active: tab === 'files' }" @click="tab='files'; loadFiles()">Workspace</button>
        <button :class="{ active: tab === 'sessions' }" @click="tab='sessions'; loadSummary()">Sessions</button>
        <button :class="{ active: tab === 'config' }" @click="tab='config'; loadConfig()">Config</button>
      </nav>

      <!-- Dashboard Tab -->
      <div x-show="tab==='dashboard'">
        <div class="grid-3">
          <div class="card stat">
            <div class="value" x-text="summary.sessions?.active ?? '-'"></div>
            <div class="label">Active Sessions</div>
          </div>
          <div class="card stat">
            <div class="value" x-text="summary.models?.length ?? '-'"></div>
            <div class="label">Available Models</div>
          </div>
          <div class="card stat">
            <div class="value" x-text="Object.keys(summary.health?.backends ?? {}).length || '-'"></div>
            <div class="label">Backends</div>
          </div>
        </div>

        <div class="grid-2">
          <!-- Backends -->
          <div class="card">
            <h3>Backends</h3>
            <table>
              <thead><tr><th>Name</th><th>Status</th><th>Auth</th></tr></thead>
              <tbody>
                <template x-for="(status, name) in (summary.health?.backends ?? {})" :key="name">
                  <tr>
                    <td x-text="name"></td>
                    <td><span class="badge badge-ok" x-text="status"></span></td>
                    <td>
                      <span :class="summary.auth?.[name]?.status?.valid ? 'badge badge-ok' : 'badge badge-err'"
                        x-text="summary.auth?.[name]?.status?.valid ? 'valid' : 'invalid'"></span>
                    </td>
                  </tr>
                </template>
              </tbody>
            </table>
          </div>

          <!-- Models -->
          <div class="card">
            <h3>Models</h3>
            <table>
              <thead><tr><th>Model</th><th>Backend</th></tr></thead>
              <tbody>
                <template x-for="m in (summary.models ?? [])" :key="m.id">
                  <tr><td x-text="m.id"></td><td><span class="badge badge-ok" x-text="m.backend"></span></td></tr>
                </template>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Workspace Tab -->
      <div x-show="tab==='files'">
        <div class="sidebar">
          <div class="file-tree card">
            <h3>Files</h3>
            <template x-for="f in files" :key="f.path">
              <div class="file-item" :class="{ active: editor.path === f.path }" @click="openFile(f.path)">
                <span class="icon" x-text="f.path.endsWith('.json') ? '{...}' : f.path.endsWith('.md') ? '#' : '~'"></span>
                <span x-text="f.path.split('/').pop()"></span>
              </div>
            </template>
            <div x-show="files.length === 0" style="color:var(--text-muted); font-size:0.85rem; padding:8px 12px">
              No files found
            </div>
          </div>
          <div class="editor-area card">
            <template x-if="!editor.path">
              <div style="color:var(--text-muted); padding:2rem; text-align:center">
                Select a file to edit
              </div>
            </template>
            <template x-if="editor.path">
              <div>
                <div class="editor-toolbar">
                  <span class="path" x-text="editor.path"></span>
                  <div style="display:flex; gap:0.5rem; align-items:center">
                    <span x-show="editor.dirty" class="dirty-dot" title="Unsaved changes"></span>
                    <button class="btn btn-sm btn-primary" @click="saveFile()" :disabled="!editor.dirty">Save</button>
                  </div>
                </div>
                <textarea x-ref="editorArea" style="display:none"></textarea>
              </div>
            </template>
          </div>
        </div>
      </div>

      <!-- Sessions Tab -->
      <div x-show="tab==='sessions'">
        <div class="card">
          <h3>Active Sessions</h3>
          <table>
            <thead><tr><th>Session ID</th><th>Messages</th><th>Last Active</th><th>Expires</th><th></th></tr></thead>
            <tbody>
              <template x-for="s in (summary.sessions?.sessions ?? [])" :key="s.session_id">
                <tr>
                  <td style="font-family:monospace; font-size:0.8rem" x-text="s.session_id?.substring(0,16) + '...'"></td>
                  <td x-text="s.message_count ?? '-'"></td>
                  <td x-text="formatTime(s.last_accessed)"></td>
                  <td x-text="formatTime(s.expires_at)"></td>
                  <td><button class="btn btn-sm btn-ghost" @click="deleteSession(s.session_id)">Delete</button></td>
                </tr>
              </template>
            </tbody>
          </table>
          <div x-show="(summary.sessions?.sessions ?? []).length === 0"
            style="color:var(--text-muted); padding:1rem; text-align:center">No active sessions</div>
        </div>
      </div>

      <!-- Config Tab -->
      <div x-show="tab==='config'">
        <div class="grid-2">
          <div class="card">
            <h3>Runtime</h3>
            <table>
              <tbody>
                <template x-for="(v, k) in (config.runtime ?? {})" :key="k">
                  <tr><td class="config-key" x-text="k"></td><td x-text="v"></td></tr>
                </template>
              </tbody>
            </table>
          </div>
          <div class="card">
            <h3>Rate Limits <span style="font-size:0.75rem; color:var(--text-muted)">(req/min)</span></h3>
            <table>
              <tbody>
                <template x-for="(v, k) in (config.rate_limits ?? {})" :key="k">
                  <tr><td class="config-key" x-text="k"></td><td x-text="v"></td></tr>
                </template>
              </tbody>
            </table>
          </div>
        </div>
        <div class="card">
          <h3>Environment</h3>
          <table>
            <tbody>
              <template x-for="(v, k) in (config.environment ?? {})" :key="k">
                <tr>
                  <td class="config-key" x-text="k"></td>
                  <td :class="{ redacted: v === '***REDACTED***' || v === '(not set)' }" x-text="v"></td>
                </tr>
              </template>
            </tbody>
          </table>
          <p style="font-size:0.8rem; color:var(--text-muted); margin-top:0.5rem"
            x-text="config._note || ''"></p>
        </div>
        <template x-if="config.mcp_servers">
          <div class="card">
            <h3>MCP Servers</h3>
            <template x-for="s in config.mcp_servers" :key="s">
              <span class="badge badge-ok" style="margin-right:0.5rem" x-text="s"></span>
            </template>
          </div>
        </template>
      </div>

    </div>
  </template>
</div>

<script>
function adminApp() {
  return {
    authenticated: false,
    loginKey: '',
    loginError: '',
    tab: 'dashboard',
    summary: {},
    files: [],
    config: {},
    editor: { path: null, content: '', etag: null, dirty: false },
    cm: null,
    toast: { show: false, msg: '', type: 'ok' },
    pollTimer: null,

    async init() {
      // Check if already authenticated (cookie-based)
      try {
        const r = await this.api('/admin/api/summary');
        if (r.ok) { this.authenticated = true; this.summary = await r.json(); this.startPolling(); }
      } catch(e) {}
    },

    async doLogin() {
      this.loginError = '';
      try {
        const r = await fetch('/admin/api/login', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ api_key: this.loginKey })
        });
        if (r.ok) {
          this.authenticated = true;
          this.loginKey = '';
          await this.loadSummary();
          this.startPolling();
        } else {
          const d = await r.json();
          this.loginError = d.detail || 'Login failed';
        }
      } catch(e) { this.loginError = 'Connection error'; }
    },

    async doLogout() {
      await fetch('/admin/api/logout', { method: 'POST' });
      this.authenticated = false;
      this.stopPolling();
    },

    api(url, opts) { return fetch(url, { ...opts, credentials: 'same-origin' }); },

    async loadSummary() {
      try {
        const r = await this.api('/admin/api/summary');
        if (r.ok) this.summary = await r.json();
        else if (r.status === 401) { this.authenticated = false; this.stopPolling(); }
      } catch(e) {}
    },

    async loadFiles() {
      try {
        const r = await this.api('/admin/api/files');
        if (r.ok) { const d = await r.json(); this.files = d.files || []; }
      } catch(e) {}
    },

    async loadConfig() {
      try {
        const r = await this.api('/admin/api/config');
        if (r.ok) this.config = await r.json();
      } catch(e) {}
    },

    async openFile(path) {
      if (this.editor.dirty && !confirm('Unsaved changes will be lost. Continue?')) return;
      try {
        const r = await this.api('/admin/api/files/' + encodeURI(path));
        if (r.ok) {
          const d = await r.json();
          this.editor = { path: d.path, content: d.content, etag: d.etag, dirty: false };
          this.$nextTick(() => this.setupEditor());
        } else {
          const d = await r.json();
          this.showToast(d.error || 'Failed to load file', 'err');
        }
      } catch(e) { this.showToast('Connection error', 'err'); }
    },

    setupEditor() {
      const ta = this.$refs.editorArea;
      if (!ta) return;
      if (this.cm) { this.cm.toTextArea(); this.cm = null; }
      ta.value = this.editor.content;
      const ext = this.editor.path.split('.').pop();
      const mode = ext === 'json' ? { name: 'javascript', json: true }
        : ext === 'yaml' || ext === 'yml' ? 'yaml' : 'markdown';
      this.cm = CodeMirror.fromTextArea(ta, {
        mode, theme: 'material-darker', lineNumbers: true, lineWrapping: true, tabSize: 2
      });
      this.cm.on('change', () => {
        this.editor.dirty = this.cm.getValue() !== this.editor.content;
      });
    },

    async saveFile() {
      if (!this.editor.path || !this.cm) return;
      const newContent = this.cm.getValue();
      try {
        const r = await this.api('/admin/api/files/' + encodeURI(this.editor.path), {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: newContent, etag: this.editor.etag })
        });
        const d = await r.json();
        if (r.ok) {
          this.editor.content = newContent;
          this.editor.etag = d.etag;
          this.editor.dirty = false;
          this.showToast('Saved', 'ok');
        } else {
          this.showToast(d.error || 'Save failed', 'err');
          if (r.status === 409) {
            if (confirm('File was modified externally. Reload?')) this.openFile(this.editor.path);
          }
        }
      } catch(e) { this.showToast('Connection error', 'err'); }
    },

    async deleteSession(id) {
      if (!confirm('Delete session ' + id.substring(0,16) + '...?')) return;
      try {
        const r = await this.api('/admin/api/sessions/' + id, { method: 'DELETE' });
        if (r.ok) { this.showToast('Session deleted', 'ok'); await this.loadSummary(); }
        else { const d = await r.json(); this.showToast(d.error || 'Delete failed', 'err'); }
      } catch(e) { this.showToast('Failed to delete', 'err'); }
    },

    async refreshAll() {
      await Promise.all([this.loadSummary(), this.loadFiles(), this.loadConfig()]);
      this.showToast('Refreshed', 'ok');
    },

    startPolling() { this.pollTimer = setInterval(() => this.loadSummary(), 15000); },
    stopPolling() { if (this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null; } },

    showToast(msg, type) {
      this.toast = { show: true, msg, type };
      setTimeout(() => { this.toast.show = false; }, 2500);
    },

    formatTime(t) {
      if (!t) return '-';
      try { return new Date(t).toLocaleString('ko-KR', { hour12: false }); }
      catch(e) { return t; }
    },

    toggleTheme() {
      const el = document.documentElement;
      el.dataset.theme = el.dataset.theme === 'dark' ? 'light' : 'dark';
    }
  };
}
</script>
</body>
</html>"""
