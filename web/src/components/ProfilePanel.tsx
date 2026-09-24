import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "../api";
import { Badge, cx } from "../ui";

interface Profile {
  level: "unknown" | "beginner" | "intermediate" | "expert";
  level_evidence: string;
  notes: { id: string; kind: string; text: string; project_id: number | null }[];
}

const LEVELS: { value: Profile["level"]; label: string; hint: string }[] = [
  { value: "beginner", label: "New to ML", hint: "plain language, full autopilot" },
  { value: "intermediate", label: "Some ML", hint: "names the techniques, shows key numbers" },
  { value: "expert", label: "ML expert", hint: "technical detail; your overrides welcome" },
];

/** What the Tuner has learned about the user, and full control to correct or forget it. */
export default function ProfilePanel() {
  const qc = useQueryClient();
  const { data: p } = useQuery({ queryKey: ["profile"], queryFn: () => api.get<Profile>("/api/profile") });
  const refresh = (next: Profile) => qc.setQueryData(["profile"], next);
  const setLevel = useMutation({ mutationFn: (level: string) => api.post<Profile>("/api/profile/level", { level }), onSuccess: refresh });
  const forget = useMutation({
    mutationFn: (id: string) => api.delete<Profile>(`/api/profile/notes/${id}`),
    onSuccess: refresh,
  });
  const reset = useMutation({ mutationFn: () => api.post<Profile>("/api/profile/reset"), onSuccess: refresh });
  if (!p) return null;

  return (
    <div className="rounded-xl border border-line bg-panel p-4">
      <div className="flex items-center gap-2">
        <span className="text-[13px] font-semibold">What the Tuner knows about you</span>
        {p.notes.length > 0 || p.level !== "unknown" ? (
          <button className="ml-auto text-[11px] text-faint hover:text-bad" onClick={() => reset.mutate()}>
            Forget everything
          </button>
        ) : null}
      </div>
      <p className="mt-1 text-xs text-muted">
        It adapts to you across projects: how much it explains and how much it involves you. It notices this from how you
        write; you can correct it. Stored on this Mac, and included in the Tuner's prompts to OpenAI.
      </p>

      <div className="mt-3 grid grid-cols-3 gap-2">
        {LEVELS.map((l) => (
          <button
            key={l.value}
            onClick={() => setLevel.mutate(l.value)}
            className={cx(
              "rounded-lg border px-2.5 py-2 text-left transition",
              p.level === l.value ? "border-accent bg-accent-soft" : "border-line hover:border-line-strong",
            )}
          >
            <div className="text-xs font-medium">{l.label}</div>
            <div className="text-[11px] text-faint">{l.hint}</div>
          </button>
        ))}
      </div>
      <p className="mt-1.5 text-[11px] text-faint">
        {p.level === "unknown" ? "Not sure yet; it'll work it out as you talk." : p.level_evidence ? `Why: ${p.level_evidence}` : null}
      </p>

      {p.notes.length > 0 && (
        <ul className="mt-3 space-y-1">
          {p.notes
            .slice()
            .reverse()
            .map((n) => (
              <li key={n.id} className="group flex items-center gap-2 text-xs">
                <Badge tone={n.kind === "preference" ? "accent" : n.kind === "goal" ? "info" : "neutral"}>{n.kind}</Badge>
                <span className="flex-1">{n.text}</span>
                <button className="text-faint opacity-0 transition group-hover:opacity-100 hover:text-bad" onClick={() => forget.mutate(n.id)}>
                  forget
                </button>
              </li>
            ))}
        </ul>
      )}
    </div>
  );
}
