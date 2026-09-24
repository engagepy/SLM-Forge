import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { MetricPoint } from "../api";

interface Series {
  key: string;
  label: string;
  split: "train" | "val";
  color: string;
  dashed?: boolean;
}

/** One chart per concern; train and validation share an x-axis of iterations. */
export function MetricChart({
  metrics,
  series,
  height = 220,
  format = (v: number) => v.toFixed(3),
}: {
  metrics: MetricPoint[];
  series: Series[];
  height?: number;
  format?: (v: number) => string;
}) {
  const byIter = new Map<number, Record<string, number>>();
  for (const m of metrics) {
    for (const s of series) {
      if (m.split !== s.split || m.values[s.key] == null) continue;
      const row = byIter.get(m.iteration) ?? { iteration: m.iteration };
      row[`${s.split}_${s.key}`] = m.values[s.key];
      byIter.set(m.iteration, row);
    }
  }
  const data = [...byIter.values()].sort((a, b) => a.iteration - b.iteration);
  if (!data.length) {
    return (
      <div className="grid place-items-center text-xs text-faint" style={{ height }}>
        Waiting for the first report…
      </div>
    );
  }
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -8 }}>
        <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="iteration" tick={{ fill: "var(--faint)", fontSize: 11 }} stroke="var(--border)" />
        <YAxis
          domain={[0, "auto"]}
          allowDataOverflow={false}
          tick={{ fill: "var(--faint)", fontSize: 11 }}
          stroke="var(--border)"
          tickFormatter={format}
          width={56}
        />
        <Tooltip
          contentStyle={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
          labelStyle={{ color: "var(--muted)" }}
          labelFormatter={(l) => `Iteration ${l}`}
          formatter={(v) => (typeof v === "number" ? format(v) : String(v))}
        />
        <Legend wrapperStyle={{ fontSize: 11, color: "var(--muted)" }} iconType="plainline" />
        {series.map((s) => (
          <Line
            key={`${s.split}_${s.key}`}
            dataKey={`${s.split}_${s.key}`}
            name={s.label}
            stroke={s.color}
            strokeWidth={2}
            strokeDasharray={s.dashed ? "5 4" : undefined}
            dot={s.split === "val" ? { r: 3, fill: s.color } : false}
            connectNulls
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
