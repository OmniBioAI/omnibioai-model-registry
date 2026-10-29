export const EMBED_READY_EVENT: 'omnibioai.embed.ready.v1';
export const EMBED_THEME_EVENT: 'omnibioai.theme.set.v1';

export type Appearance = {
  theme: 'light' | 'dark';
  accent: 'teal' | 'blue' | 'purple' | 'orange';
  density: 'comfortable' | 'compact';
  motion: 'system' | 'reduce';
};

export function normalizeAppearance(values?: Partial<Appearance>, systemTheme?: string): Appearance;
export function applyAppearance(root: HTMLElement, values: Appearance): Appearance;
export function parseThemeMessage(data: unknown): Appearance | null;
export function connectAppearanceMessages(win?: Window, doc?: Document, devStudioOrigin?: string): () => void;
