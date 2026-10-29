import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';
import {
  connectAppearanceMessages,
  EMBED_READY_EVENT,
  EMBED_THEME_EVENT,
  parseThemeMessage,
} from '../src/embedAppearance.js';

const rootDir = path.resolve(import.meta.dirname, '..');
const html = fs.readFileSync(path.join(rootDir, 'index.html'), 'utf8');
const bootstrap = html.match(/<script>\s*([\s\S]*?)\s*<\/script>/)?.[1];

function embedded(referrer = 'https://webstudio.omnibioai.org/workbench') {
  const listeners = new Map();
  const sent = [];
  const parent = { postMessage: (message, origin) => sent.push([message, origin]) };
  const win = {
    parent,
    addEventListener: (type, handler) => listeners.set(type, handler),
    removeEventListener: type => listeners.delete(type),
  };
  const root = { dataset: {}, style: { cssText: 'untouched' }, innerHTML: 'untouched' };
  return { win, doc: { referrer, documentElement: root }, parent, root, listeners, sent };
}

test('pre-paint bootstrap validates values, marks embedding, and strips only private launcher data', () => {
  assert.ok(bootstrap);
  assert.ok(html.indexOf('<script>') < html.indexOf('<link'));
  const location = {
    pathname: '/_svc/modelregistry/',
    search: '?obai_theme=markup&obai_accent=purple&obai_density=compact&obai_motion=reduce&obai_owner=x&token=x&author=Jane&username=reader',
    hash: '#models',
  };
  let replaced;
  const root = { dataset: {}, style: { cssText: 'untouched' }, innerHTML: 'untouched' };
  const parent = {};
  const windowObject = {
    parent,
    location,
    matchMedia: () => ({ matches: false }),
    history: { state: null, replaceState: (_state, _title, url) => { replaced = url; } },
  };
  vm.runInNewContext(bootstrap, {
    window: windowObject,
    document: { documentElement: root },
    URLSearchParams,
    Object,
    Set,
    Array,
  });
  assert.deepEqual({ ...root.dataset }, { theme: 'light', accent: 'purple', density: 'compact', motion: 'reduce', embedded: 'true' });
  const clean = new URL(replaced, 'https://registry.example');
  assert.equal(clean.searchParams.has('obai_owner'), false);
  assert.equal(clean.searchParams.has('token'), false);
  assert.equal(clean.searchParams.get('author'), 'Jane');
  assert.equal(clean.searchParams.get('username'), 'reader');
  assert.equal(clean.hash, '#models');
  assert.equal(root.style.cssText, 'untouched');
  assert.equal(root.innerHTML, 'untouched');
});

test('theme messages require trusted parent origin, parent source, exact keys, and valid values', () => {
  const context = embedded();
  const stop = connectAppearanceMessages(context.win, context.doc);
  assert.deepEqual(context.sent, [[{ type: EMBED_READY_EVENT }, 'https://webstudio.omnibioai.org']]);
  const listener = context.listeners.get('message');
  const valid = { type: EMBED_THEME_EVENT, theme: 'light', accent: 'purple', density: 'compact', motion: 'reduce' };
  const deliver = (origin, source, data) => listener({ origin, source, data });

  deliver('https://evil.example', context.parent, valid);
  deliver('https://webstudio.omnibioai.org', {}, valid);
  deliver('https://webstudio.omnibioai.org', context.parent, { ...valid, extra: true });
  deliver('https://webstudio.omnibioai.org', context.parent, { ...valid, accent: 'javascript' });
  assert.deepEqual(context.root.dataset, {});
  assert.equal(parseThemeMessage({ ...valid, username: 'reader' }), null);

  deliver('https://webstudio.omnibioai.org', context.parent, valid);
  assert.deepEqual(context.root.dataset, { theme: 'light', accent: 'purple', density: 'compact', motion: 'reduce' });
  assert.equal(context.root.style.cssText, 'untouched');
  assert.equal(context.root.innerHTML, 'untouched');
  stop();
  assert.equal(context.listeners.has('message'), false);
});

test('development parent is disabled by default and accepted only when explicitly exact', () => {
  const disabled = embedded('http://localhost:5174/workbench');
  connectAppearanceMessages(disabled.win, disabled.doc);
  assert.deepEqual(disabled.sent, []);
  assert.equal(disabled.listeners.has('message'), false);

  const enabled = embedded('http://localhost:5174/workbench');
  connectAppearanceMessages(enabled.win, enabled.doc, 'http://localhost:5174');
  assert.deepEqual(enabled.sent, [[{ type: EMBED_READY_EVENT }, 'http://localhost:5174']]);

  const lookalike = embedded('http://localhost:5174/workbench');
  connectAppearanceMessages(lookalike.win, lookalike.doc, 'http://localhost:5174.attacker.example');
  assert.deepEqual(lookalike.sent, []);
});

test('nginx CSP is always applied and vendored package is exact and tooling-free', () => {
  const dockerfile = fs.readFileSync(path.resolve(rootDir, '..', '..', 'Dockerfile'), 'utf8');
  assert.match(dockerfile, /add_header Content-Security-Policy "frame-ancestors '\\\''self'\\\'' https:\/\/webstudio\.omnibioai\.org" always;/);
  const vendor = path.join(rootDir, 'vendor', 'omnibioai-design-tokens');
  const metadata = JSON.parse(fs.readFileSync(path.join(vendor, 'package.json'), 'utf8'));
  assert.equal(metadata.version, '1.1.0');
  assert.equal(metadata.scripts, undefined);
  assert.equal(metadata.devDependencies, undefined);
  assert.equal(fs.existsSync(path.join(vendor, 'package-lock.json')), false);
});
