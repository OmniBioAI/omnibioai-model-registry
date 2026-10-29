import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

const root = path.resolve(import.meta.dirname, '..');
const literal = /#[0-9a-f]{3,8}\b|rgba?\([^)]*\)/gi;
const extensions = new Set(['.css', '.html', '.js', '.jsx', '.svg', '.ts', '.tsx']);
// Explicit exceptions: upstream vendored token definitions and authored SVG artwork.
const exceptions = [
  `${path.sep}vendor${path.sep}omnibioai-design-tokens${path.sep}`,
  `${path.sep}public${path.sep}favicon.svg`,
  `${path.sep}public${path.sep}icons.svg`,
  `${path.sep}src${path.sep}assets${path.sep}react.svg`,
  `${path.sep}src${path.sep}assets${path.sep}vite.svg`,
];

function files(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) return entry.name === 'node_modules' || entry.name === 'dist' ? [] : files(file);
    return extensions.has(path.extname(file)) ? [file] : [];
  });
}

test('application colors are tokens outside the single marked mapping block', () => {
  const failures = [];
  for (const file of files(root)) {
    if (exceptions.some(exception => file.includes(exception))) continue;
    const source = fs.readFileSync(file, 'utf8');
    const scrubbed = source.replace(/\/\* color-token-mapping:start[\s\S]*?color-token-mapping:end \*\//g, '');
    for (const match of scrubbed.matchAll(literal)) {
      const line = scrubbed.slice(0, match.index).split('\n').length;
      failures.push(`${path.relative(root, file)}:${line}: ${match[0]}`);
    }
  }
  assert.deepEqual(failures, []);
});
