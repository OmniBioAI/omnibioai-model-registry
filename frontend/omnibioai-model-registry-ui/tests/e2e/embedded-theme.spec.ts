import { expect, test, type Frame } from '@playwright/test';

const models = [
  { task: 'classification', model_name: 'cell-type-model', version: '1.2.0', created_at: '2026-10-01T12:00:00Z' },
];

async function mockApi(page) {
  await page.route('**/modelregistry/v1/models', route => route.fulfill({ json: models }));
  await page.route('**/modelregistry/v1/aliases**', route => route.fulfill({ json: { ok: true, model_name: 'cell-type-model', aliases: [{ alias: 'production', version: '1.2.0' }] } }));
}

async function contrastFailures(frame: Frame) {
  return frame.locator('body').evaluate(root => {
    const parse = (value: string) => (value.match(/[\d.]+/g) || []).map(Number);
    const composite = (front: number[], back: number[]) => front.length > 3 && front[3] < 1
      ? front.slice(0, 3).map((channel, index) => channel * front[3] + back[index] * (1 - front[3]))
      : front.slice(0, 3);
    const luminance = ([r, g, b]: number[]) => {
      const channels = [r, g, b].map(value => { const s = value / 255; return s <= .04045 ? s / 12.92 : ((s + .055) / 1.055) ** 2.4; });
      return .2126 * channels[0] + .7152 * channels[1] + .0722 * channels[2];
    };
    const contrast = (a: number[], b: number[]) => { const [light, dark] = [luminance(a), luminance(b)].sort((x, y) => y - x); return (light + .05) / (dark + .05); };
    return [...root.querySelectorAll('*')].flatMap(element => {
      const style = getComputedStyle(element);
      const hasText = [...element.childNodes].some(node => node.nodeType === Node.TEXT_NODE && node.textContent?.trim());
      if (!hasText || style.display === 'none' || style.visibility === 'hidden') return [];
      const layers: Element[] = [];
      for (let layer: Element | null = element; layer; layer = layer.parentElement) layers.push(layer);
      let backdrop = parse(getComputedStyle(document.body).backgroundColor).slice(0, 3);
      for (const layer of layers.reverse()) {
        const color = parse(getComputedStyle(layer).backgroundColor);
        if (color.length >= 3 && !(color[0] === 0 && color[1] === 0 && color[2] === 0 && color[3] === 0)) backdrop = composite(color, backdrop);
      }
      const ratio = contrast(parse(style.color), backdrop);
      return ratio < 4.5 ? [{ text: element.textContent?.trim().slice(0, 40), ratio }] : [];
    });
  });
}

for (const scenario of [
  { theme: 'light', accent: 'purple', width: 1440, height: 1000 },
  { theme: 'dark', accent: 'teal', width: 390, height: 844 },
]) {
  test(`embedded ${scenario.theme}+${scenario.accent} at ${scenario.width}px is themed and AA`, async ({ page }) => {
    await page.setViewportSize({ width: scenario.width, height: scenario.height });
    await mockApi(page);
    await page.goto('/_svc/modelregistry/');
    await page.setContent(`<iframe title="Model Registry" src="http://127.0.0.1:4176/_svc/modelregistry/?obai_theme=${scenario.theme}&obai_accent=${scenario.accent}" style="width:100%;height:${scenario.height - 20}px;border:0"></iframe>`);
    const iframe = page.locator('iframe');
    const frame = page.frames().find(candidate => candidate !== page.mainFrame());
    expect(frame).toBeTruthy();
    await expect(iframe).toBeVisible();
    await expect(frame!.getByText('cell-type-model')).toBeVisible();
    await expect(frame!.locator('.app-topbar')).toHaveCSS('display', 'none');
    await expect(frame!.locator('html')).toHaveAttribute('data-theme', scenario.theme);
    await expect(frame!.locator('html')).toHaveAttribute('data-accent', scenario.accent);
    expect(await contrastFailures(frame!)).toEqual([]);

    if (scenario.theme === 'light') {
      const active = frame!.getByRole('button', { name: 'Models', exact: true });
      const sidebarActive = frame!.getByRole('button', { name: /All tasks/ });
      const colors = await active.evaluate(element => {
        const probe = document.createElement('span');
        probe.style.color = 'var(--color-accent)';
        document.body.appendChild(probe);
        const accent = getComputedStyle(probe).color;
        probe.remove();
        return { active: getComputedStyle(element).color, accent };
      });
      const sidebarColor = await sidebarActive.evaluate(element => getComputedStyle(element).color);
      const channels = (value: string) => (value.match(/[\d.]+/g) || []).map(Number);
      expect(channels(colors.active).slice(0, 3)).toEqual([112, 48, 160]);
      expect(channels(sidebarColor).slice(0, 3)).toEqual([112, 48, 160]);
      expect(channels(colors.accent).slice(0, 3)).toEqual([112, 48, 160]);
    }

    await page.screenshot({ path: `/tmp/n11a-${scenario.theme}-${scenario.accent}-${scenario.width}.png`, fullPage: true });
  });
}
