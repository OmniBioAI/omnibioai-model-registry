type Props = {
  data: { value: number; step: number }[];
  width?: number;
  height?: number;
  color?: string;
  label?: string;
};

export default function RunMetricChart({
  data,
  width = 200,
  height = 60,
  color = '#4ADE80',
  label,
}: Props) {
  if (data.length === 0) return null;

  const pad = { top: 4, right: 44, bottom: 4, left: 4 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const values = data.map((d) => d.value);
  const steps = data.map((d) => d.step);
  const minV = Math.min(...values);
  const maxV = Math.max(...values);
  const minS = Math.min(...steps);
  const maxS = Math.max(...steps);
  const vRange = maxV - minV || 1;
  const sRange = maxS - minS || 1;

  function sx(step: number): number {
    return pad.left + ((step - minS) / sRange) * innerW;
  }
  function sy(value: number): number {
    return pad.top + innerH - ((value - minV) / vRange) * innerH;
  }

  const last = data[data.length - 1];
  const lx = sx(last.step);
  const ly = sy(last.value);
  const points = data.map((d) => `${sx(d.step)},${sy(d.value)}`).join(' ');

  const raw = last.value;
  const displayVal =
    raw === 0
      ? '0'
      : Math.abs(raw) < 0.0001 || Math.abs(raw) >= 10000
      ? raw.toExponential(2)
      : Math.abs(raw) < 1
      ? raw.toPrecision(3)
      : raw.toPrecision(4);

  return (
    <div>
      {label && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 2, fontFamily: 'ui-monospace, Consolas, monospace' }}>
          {label}
        </div>
      )}
      <svg width={width} height={height} style={{ display: 'block', overflow: 'visible' }}>
        {data.length > 1 && (
          <polyline
            points={points}
            fill="none"
            stroke={color}
            strokeWidth={1.5}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        )}
        <circle cx={lx} cy={ly} r={3} fill={color} />
        <text x={lx + 6} y={ly + 4} fontSize={10} fill={color} fontFamily="ui-monospace, Consolas, monospace">
          {displayVal}
        </text>
      </svg>
    </div>
  );
}
