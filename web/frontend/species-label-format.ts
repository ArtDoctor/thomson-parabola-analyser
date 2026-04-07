/**
 * Pretty-print species labels from the pipeline (e.g. C^1+, Si^4+).
 * Canvas: Unicode superscripts. DOM: HTML <sup> with escaping.
 */

const SPECIES_LABEL_RE = /^([A-Za-z]+)\^(\d+)\+$/;

const SUP_MAP: Record<string, string> = {
  '0': '⁰',
  '1': '¹',
  '2': '²',
  '3': '³',
  '4': '⁴',
  '5': '⁵',
  '6': '⁶',
  '7': '⁷',
  '8': '⁸',
  '9': '⁹',
  '+': '⁺',
  '-': '⁻',
};

function toSuperscriptUnicode(chunk: string): string {
  let out = '';
  for (const ch of chunk) {
    const mapped = SUP_MAP[ch];
    out += mapped !== undefined ? mapped : ch;
  }
  return out;
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** Label drawn on the detector canvas (2D context fillText). */
export function formatSpeciesLabelCanvas(label: string): string {
  const m = label.match(SPECIES_LABEL_RE);
  if (!m) return label;
  return `${m[1]}${toSuperscriptUnicode(`${m[2]}+`)}`;
}

/** Safe HTML fragment for tooltip / panel titles (no wrapper element). */
export function formatSpeciesLabelHtml(label: string): string {
  const m = label.match(SPECIES_LABEL_RE);
  if (!m) return escapeHtml(label);
  const symbol = escapeHtml(m[1]);
  const q = escapeHtml(m[2]);
  return `${symbol}<sup>${q}+</sup>`;
}
