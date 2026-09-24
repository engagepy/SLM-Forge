import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate } from "react-router";

import { api, fmt, type Project } from "../api";
import { useSystem } from "../hooks";
import ProfilePanel from "../components/ProfilePanel";
import { Button, ErrorNote, TextArea } from "../ui";

const EXAMPLES = [
  "A cooking assistant that answers home-cooking questions with exact times and temperatures",
  "A support bot that answers questions about our product in a friendly, brief tone",
  "A tutor that explains high-school physics step by step",
];

/** One question to start. Everything after this happens in the Studio with the Tuner. */
export default function Home() {
  const navigate = useNavigate();
  const { data: sys } = useSystem();
  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api.get<Project[]>("/api/projects") });
  const [goal, setGoal] = useState("");
  const create = useMutation({
    mutationFn: () => api.post<Project>("/api/projects", { name: "New model", goal }),
    onSuccess: (p) => navigate(`/p/${p.id}`),
  });

  return (
    <div className="mx-auto flex min-h-full max-w-2xl flex-col justify-center px-6 py-12">
      <div className="mb-8 flex items-center gap-2">
        <span className="grid size-8 place-items-center rounded-lg bg-accent font-bold text-white">▲</span>
        <span className="text-lg font-semibold tracking-tight">SLM Forge</span>
      </div>
      <h1 className="text-2xl font-semibold tracking-tight">What should your model do?</h1>
      <p className="mt-2 text-[14px] leading-relaxed text-muted">
        Describe it in a sentence. The Tuner, an AI guide, picks the model, finds and cleans the data, trains it on this Mac
        {sys ? ` (${sys.hardware.chip}, ${sys.hardware.total_memory_gb.toFixed(0)} GB)` : ""}, and explains every step as it goes.
      </p>
      <form
        className="mt-6 space-y-3"
        onSubmit={(e) => {
          e.preventDefault();
          if (goal.trim()) create.mutate();
        }}
      >
        <TextArea
          rows={3}
          autoFocus
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
            <button key={ex} type="button" onClick={() => setGoal(ex)} className="rounded-full border border-line px-3 py-1 text-xs text-muted hover:border-accent hover:text-fg">
              {ex}
            </button>
          ))}
        </div>
        <ErrorNote error={create.error} />
        <Button type="submit" variant="primary" disabled={!goal.trim()} loading={create.isPending}>
          Start with the Tuner →
        </Button>
      </form>

      <div className="mt-10">
        <ProfilePanel />
      </div>

      {!!projects.data?.length && (
        <div className="mt-8">
          <div className="mb-2 text-[11px] font-medium uppercase tracking-wide text-faint">Your models</div>
          <ul className="divide-y divide-line rounded-xl border border-line bg-panel">
            {projects.data.map((p) => (
              <li key={p.id}>
                <Link to={`/p/${p.id}`} className="flex items-center gap-3 px-4 py-3 hover:bg-panel-2">
                  <div className="min-w-0 flex-1">
                    <div className="text-[13px] font-medium">{p.name}</div>
                    <div className="truncate text-xs text-muted">{p.goal || "No goal yet"}</div>
                  </div>
                  <span className="text-[11px] text-faint">{fmt.ago(p.created_at)}</span>
                </Link>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
