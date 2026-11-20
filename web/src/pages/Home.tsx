import { useMutation } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { useNavigate } from "react-router";

import MetricsBar from "../components/MetricsBar";
import { api, type Project } from "../api";
import ProfilePanel from "../components/ProfilePanel";
import { sessionDetail, STATE, useSessions } from "../components/SessionsSidebar";
import { useSystem } from "../hooks";
import { Badge, Button, ErrorNote, LinkButton, TextArea } from "../ui";

// Small, characterful goals: what a tiny model does well and people enjoy trying.
interface SpecImport {
  project: { name: string; goal: string };
  warnings: string[];
}

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
  // A build spec (YAML, kind: slm_build): preview first (creates nothing), then import and start.
  const specInput = useRef<HTMLInputElement>(null);
  const [spec, setSpec] = useState<{ yaml: string; preview: SpecImport } | null>(null);
  const previewSpec = useMutation({
    mutationFn: async (file: File) => {
      const yaml = await file.text();
      return { yaml, preview: await api.post<SpecImport>("/api/projects/from-spec", { yaml, preview: true }) };
    },
    onSuccess: setSpec,
  });
  const importSpec = useMutation({
    mutationFn: async (yaml: string) => {
      const { project } = await api.post<{ project: Project }>("/api/projects/from-spec", { yaml });
      await api.post(`/api/projects/${project.id}/tuner/start`);
      return project;
    },
    onSuccess: (p) => navigate(`/p/${p.id}`),
  });
  const active = sessions?.sessions.filter((s) => ["running", "queued", "thinking", "waiting"].includes(s.state)) ?? [];
  // Any project with an exported model, finished by the Tuner or not.
  const finished = sessions?.sessions.filter((s) => s.models > 0) ?? [];

  return (
    <div className="flex h-full flex-col">
      <MetricsBar />
      <div className="min-h-0 flex-1 overflow-y-auto">
      {/* Centred in the viewport while it fits; scrolls from the top once it doesn't. */}
      <div className="mx-auto flex min-h-full max-w-3xl flex-col justify-center gap-8 px-6 py-10">
        {active.length > 0 && (
          <section>
            <h2 className="mb-2 text-[11px] font-medium uppercase tracking-wide text-faint">Running now</h2>
            <ul className="space-y-2">
              {active.map((s) => (
                <li key={s.project_id} className="rounded-xl border border-info/30 bg-panel p-3">
                  <div className="flex items-center gap-2">
                    <span className="size-2 animate-pulse rounded-full bg-info" />
                    <span className="text-[13px] font-medium">{s.name}</span>
                    <Badge tone={STATE[s.state].tone}>{STATE[s.state].label}</Badge>
                    <span className="ml-auto flex gap-2">
                      <LinkButton to={`/p/${s.project_id}`} size="sm" variant="primary">Studio</LinkButton>
                      <LinkButton to={`/p/${s.project_id}/overview`} size="sm" variant="ghost">Advanced</LinkButton>
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
          <h1 className="text-xl font-semibold tracking-tight">What should your model do?</h1>
          <p className="mt-1 text-[13px] leading-relaxed text-muted">
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
                  className="cursor-pointer rounded-full border border-line px-3 py-1 text-xs text-muted hover:border-accent hover:text-fg"
                >
                  {ex}
                </button>
              ))}
            </div>
            <ErrorNote error={create.error ?? previewSpec.error ?? importSpec.error} />
            <div className="flex flex-wrap items-center gap-2">
              <Button type="submit" variant="primary" disabled={!goal.trim()} loading={create.isPending}>
                Start with the Tuner →
              </Button>
              <span className="text-xs text-faint">or</span>
              <Button
                type="button"
                loading={previewSpec.isPending}
                onClick={() => specInput.current?.click()}
                title="A YAML build recipe (kind: slm_build). You'll see what it becomes before anything is created."
              >
                Import a build spec (YAML)
              </Button>
              <input
                ref={specInput}
                type="file"
                accept=".yaml,.yml"
                hidden
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  setSpec(null);
                  if (file) previewSpec.mutate(file);
                  e.target.value = "";
                }}
              />
            </div>
          </form>

          {spec && (
            <div className="mt-4 space-y-3 rounded-xl border border-accent/40 bg-panel p-4">
              <div>
                <div className="text-[11px] font-medium uppercase tracking-wide text-faint">From your build spec</div>
                <div className="mt-0.5 text-[14px] font-medium">{spec.preview.project.name}</div>
                <p className="mt-0.5 text-xs leading-relaxed text-muted">{spec.preview.project.goal}</p>
              </div>
              {spec.preview.warnings.length > 0 && (
                <details className="rounded-lg bg-panel-2 px-3 py-2">
                  <summary className="cursor-pointer text-xs text-warn">
                    {spec.preview.warnings.length} parts of the spec are done differently on this Mac
                  </summary>
                  <ul className="mt-2 list-disc space-y-0.5 pl-4 text-xs text-muted">
                    {spec.preview.warnings.map((w) => (
                      <li key={w}>{w}</li>
                    ))}
                  </ul>
                </details>
              )}
              <div className="flex gap-2">
                <Button variant="primary" loading={importSpec.isPending} onClick={() => importSpec.mutate(spec.yaml)}>
                  Import and start the Tuner →
                </Button>
                <Button variant="ghost" onClick={() => setSpec(null)}>
                  Cancel
                </Button>
              </div>
            </div>
          )}
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
                  <LinkButton to={`/p/${s.project_id}`} size="sm" variant="ghost">Studio</LinkButton>
                  <LinkButton to={`/p/${s.project_id}/try`} size="sm" variant="good">▶ Try it</LinkButton>
                </li>
              ))}
            </ul>
          </section>
        )}

        <ProfilePanel />
      </div>
      </div>
    </div>
  );
}
