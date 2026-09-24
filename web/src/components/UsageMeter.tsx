import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "../api";
import { cx } from "../ui";

export interface Spend {
  provider: "openai";
  configured: boolean;
  key: string | null;
  project_id: string | null;
  model: string;
  setup?: string;
  error?: string | null;
  as_of?: string;
  today_usd?: number;
  month_to_date_usd?: number;
  last_7_days_usd?: number;
  last_30_days_usd?: number;
  daily?: { date: string; usd: number }[];
}

export const usd = (n: number | undefined) => (n == null ? "—" : `$${n.toFixed(2)}`);

/** The account's spend, from OpenAI's Costs API; the server caches it for ten minutes. */
export function useSpend() {
  return useQuery({ queryKey: ["usage"], queryFn: () => api.get<Spend>("/api/usage"), refetchInterval: 600_000 });
}

/** A pill with what the OpenAI account has spent; click for the breakdown or to refresh. */
export default function UsageMeter({ className }: { className?: string }) {
  const { data } = useSpend();
  const qc = useQueryClient();
  const [open, setOpen] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  if (!data) return null;
  const ok = data.configured && !data.error;
  const refresh = async () => {
    setRefreshing(true);
    try {
      qc.setQueryData(["usage"], await api.get<Spend>("/api/usage?refresh=true"));
    } finally {
      setRefreshing(false);
    }
  };
  return (
    <div className={cx("relative", className)}>
      <button
        onClick={() => setOpen(!open)}
        title="OpenAI spend, read from your account"
        className="flex items-center gap-1.5 rounded-full border border-line px-2.5 py-1 text-xs text-muted hover:text-fg"
      >
        <span className="text-faint">OpenAI</span>
        {ok ? (
          <>
            <span className="num font-medium text-fg">{usd(data.month_to_date_usd)}</span>
            <span className="num text-faint">this month · {usd(data.today_usd)} today</span>
          </>
        ) : (
          <span className={data.error ? "text-warn" : "text-faint"}>{data.error ? "can't read spend" : "spend not set up"}</span>
        )}
      </button>
      {open && (
        <div className="absolute right-0 z-20 mt-1.5 w-72 rounded-xl border border-line bg-panel p-3 text-xs shadow-lg">
          {ok ? (
            <>
              <ul className="space-y-1">
                {(
                  [
                    ["Today", data.today_usd],
                    ["This month", data.month_to_date_usd],
                    ["Last 7 days", data.last_7_days_usd],
                    ["Last 30 days", data.last_30_days_usd],
                  ] as [string, number | undefined][]
                ).map(([label, n]) => (
                  <li key={label} className="flex justify-between">
                    <span className="text-muted">{label}</span>
                    <span className="num font-medium">{usd(n)}</span>
                  </li>
                ))}
              </ul>
              <div className="mt-2 flex items-center justify-between border-t border-line pt-2 text-[11px] text-faint">
                <span>{data.project_id ? `project ${data.project_id}` : "whole organisation"} · as of {data.as_of?.slice(11, 16)} UTC</span>
                <button onClick={refresh} className="hover:text-fg" disabled={refreshing}>
                  {refreshing ? "…" : "refresh"}
                </button>
              </div>
              <p className="mt-1.5 text-[11px] leading-relaxed text-faint">
                From OpenAI's Costs API; OpenAI posts costs with a lag of a few hours. Key in use {data.key} · {data.model}
              </p>
            </>
          ) : data.error ? (
            <p className="leading-relaxed text-warn">{data.error}</p>
          ) : (
            <p className="leading-relaxed text-muted">{data.setup}</p>
          )}
        </div>
      )}
    </div>
  );
}
