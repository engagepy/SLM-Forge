import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate } from "react-router";

import { api, type Project } from "../api";
import ProfilePanel from "../components/ProfilePanel";
import { sessionDetail, useSessions } from "../components/SessionsSidebar";
import { useSystem } from "../hooks";
import { Badge, Button, ErrorNote, TextArea } from "../ui";

// Small, characterful goals: what a tiny model does well and people enjoy trying.
const EXAMPLES = [
  "A pirate who explains everyday science in two sentences",
  "A haiku poet that turns any topic into a haiku",
  "A cooking helper that answers with exact times and temperatures, in one line",
];

/** The central place: what's running, start something new, and what the Tuner knows about you. */
export default function Home() {
  const navigate = useNavigate();
  const { data: sys } = useSystem();
  const { data: sessions } = useSessions();
  const [goal, setGoal] = useState("");
  const create = useMutation({
    // Creating a project is the one place the Tuner starts by itself; opening one never does.
    mutationFn: async () => {
      const p = await api.post<Project>("/api/projects", { name: "New model", goal });
      await api.post(`/api/projects/${p.id}/tuner/start`);
      return p;
    },
    onSuccess: (p) => navigate(`/p/${p.id}`),
  });
  const active = sessions?.sessions.filter((s) => ["running", "queued", "thinking", "waiting"].includes(s.state)) ?? [];
  // Any project with an exported model, finished by the Tuner or not.
  const finished = sessions?.sessions.filter((s) => s.models > 0) ?? [];

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-3xl space-y-8 px-6 py-10">
        {active.length > 0 && (
          <section>
            <h2 className="mb-2 text-[11px] font-medium uppercase tracking-wide text-faint">Running now</h2>
            <ul className="space-y-2">
              {active.map((s) => (
                <li key={s.project_id} className="rounded-xl border border-info/30 bg-panel p-3">
                  <div className="flex items-center gap-2">
                    <span className="size-2 animate-pulse rounded-full bg-info" />
                    <span className="text-[13px] font-medium">{s.name}</span>
                    <Badge tone={s.state === "queued" || s.state === "waiting" ? "warn" : "info"}>{s.state === "waiting" ? "needs you" : s.state}</Badge>
                    <span className="ml-auto flex gap-2">
                      <Link to={`/p/${s.project_id}/overview`}>
                        <Button size="sm" variant="ghost">
                          Advanced
                        </Button>
                      </Link>
                      <Link to={`/p/${s.project_id}`}>
                        <Button size="sm" variant="primary">
                          Open →
                        </Button>
                      </Link>
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-muted">{sessionDetail(s)}</p>
                  {s.job?.progress?.percent != null && (
                    <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-panel-2">
                      <div className="h-full bg-info transition-all" style={{ width: `${s.job.progress.percent}%` }} />
                    </div>
                  )}
                </li>
              ))}
            </ul>
          </section>
        )}

        <section>
          <h1 className="text-2xl font-semibold tracking-tight">What should your model do?</h1>
          <p className="mt-2 text-[14px] leading-relaxed text-muted">
            Describe it in a sentence. The Tuner, an AI guide, picks the smallest model that can do it, writes or finds a little
            data and trains it on this Mac{sys ? ` (${sys.hardware.chip}, ${sys.hardware.total_memory_gb.toFixed(0)} GB)` : ""},
            explaining every step. It asks before each run, and at the end you chat with your finished model. Your projects are
            all in the sidebar.
          </p>
          <form
            className="mt-5 space-y-3"
            onSubmit={(e) => {
              e.preventDefault();
              if (goal.trim()) create.mutate();
            }}
          >
            <TextArea
              rows={3}
              autoFocus={!active.length}
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && goal.trim()) {
                  e.preventDefault();
                  create.mutate();
                }
              }}
              placeholder="A small model that…"
              className="text-[15px]"
            />
            <div className="flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  onClick={() => setGoal(ex)}
                  className="rounded-full border border-line px-3 py-1 text-xs text-muted hover:border-accent hover:text-fg"
                >
                  {ex}
                </button>
              ))}
            </div>
            <ErrorNote error={create.error} />
            <Button type="submit" variant="primary" disabled={!goal.trim()} loading={create.isPending}>
              Start with the Tuner →
            </Button>
          </form>
        </section>

        {finished.length > 0 && (
          <section>
            <h2 className="mb-2 text-[11px] font-medium uppercase tracking-wide text-faint">Your models</h2>
            <ul className="space-y-2">
              {finished.map((s) => (
                <li key={s.project_id} className="flex items-center gap-3 rounded-xl border border-line bg-panel p-3">
                  <span className="grid size-7 shrink-0 place-items-center rounded-full bg-good-soft text-[12px] text-good">✓</span>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-[13px] font-medium">{s.name}</div>
                    <div className="truncate text-xs text-muted">{s.goal}</div>
                  </div>
                  <Link to={`/p/${s.project_id}`}>
                    <Button size="sm" variant="ghost">
                      Studio
                    </Button>
                  </Link>
                  <Link to={`/p/${s.project_id}/try`}>
                    <Button size="sm" variant="primary">
                      ▶ Try it
                    </Button>
                  </Link>
                </li>
              ))}
            </ul>
          </section>
        )}

        <ProfilePanel />
      </div>
    </div>
  );
}
