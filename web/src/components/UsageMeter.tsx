import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "../api";
import { cx } from "../ui";

interface Totals {
  requests: number;
  input_tokens: number;
  cached_input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  by_purpose?: Record<string, { cost_usd: number; requests: number }>;
}

export interface Usage {
  key: string | null;
  model: string;
  prices: { input: number; cached: number; output: number; configured: boolean };
  all_time: Totals;
  project?: Totals;
}

const PURPOSE: Record<string, string> = {
  tuner: "Tuner turns",
  synthesize: "writing examples",
  review: "AI review",
  evaluate: "scoring",
  agents: "helper agents",
  other: "other",
};

export const usd = (n: number) => (n < 0.01 && n > 0 ? "<$0.01" : `$${n.toFixed(2)}`);
const tokens = (n: number) => (n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1000 ? `${Math.round(n / 1000)}k` : String(n));

/** The OpenAI meter; refreshed by the jobs feed's `usage` events. */
export function useUsage(projectId?: number) {
  return useQuery({
    queryKey: ["usage", projectId ?? "all"],
    queryFn: () => api.get<Usage>(`/api/usage${projectId != null ? `?project_id=${projectId}` : ""}`),
  });
}

/** A pill with what this project (or everything) has cost so far; click for the breakdown. */
export default function UsageMeter({ projectId, className }: { projectId?: number; className?: string }) {
  const { data } = useUsage(projectId);
  const [open, setOpen] = useState(false);
  if (!data) return null;
  const t = data.project ?? data.all_time;
  return (
    <div className={cx("relative", className)}>
      <button
        onClick={() => setOpen(!open)}
        title="OpenAI usage so far (estimated from token counts)"
        className="flex items-center gap-1.5 rounded-full border border-line px-2.5 py-1 text-xs text-muted hover:text-fg"
      >
        <span className="text-faint">API</span>
        <span className="num font-medium text-fg">{usd(t.cost_usd)}</span>
        <span className="num text-faint">· {tokens(t.input_tokens + t.output_tokens)} tokens</span>
      </button>
      {open && (
        <div className="absolute right-0 z-20 mt-1.5 w-72 rounded-xl border border-line bg-panel p-3 text-xs shadow-lg">
          <div className="flex items-center justify-between">
            <span className="font-semibold">{data.project ? "This project" : "All projects"}</span>
            <span className="num text-[15px] font-semibold">{usd(t.cost_usd)}</span>
          </div>
          <div className="num mt-1 text-faint">
            {t.requests} requests · {tokens(t.input_tokens)} in ({tokens(t.cached_input_tokens)} cached) · {tokens(t.output_tokens)} out
          </div>
          {t.by_purpose && Object.keys(t.by_purpose).length > 0 && (
            <ul className="mt-2 space-y-0.5 border-t border-line pt-2">
              {Object.entries(t.by_purpose)
                .sort((a, b) => b[1].cost_usd - a[1].cost_usd)
                .map(([p, v]) => (
                  <li key={p} className="flex justify-between">
                    <span className="text-muted">{PURPOSE[p] ?? p}</span>
                    <span className="num">{usd(v.cost_usd)}</span>
                  </li>
                ))}
            </ul>
          )}
          {data.project && (
            <div className="mt-2 flex justify-between border-t border-line pt-2">
              <span className="text-muted">All projects</span>
              <span className="num">{usd(data.all_time.cost_usd)}</span>
            </div>
          )}
          <div className="mt-2 border-t border-line pt-2 text-[11px] leading-relaxed text-faint">
            Key {data.key ?? "not set"} · {data.model}
            <br />
            Estimated at ${data.prices.input}/{data.prices.cached}/${data.prices.output} per 1M tokens (in / cached / out)
            {!data.prices.configured && ". Set SLM_OPENAI_PRICE_INPUT, _CACHED and _OUTPUT in .env to match your plan."}
          </div>
        </div>
      )}
    </div>
  );
}
