// Advanced · Evaluate: the Studio's evaluate stage at full width, with the fixed test set beside it.
import { useProjectId, useStudio } from "../hooks";
import { Card, cx, LinkButton, Spinner } from "../ui";
import { EvaluateView } from "./studio/Evaluate";

export default function EvaluatePage() {
  const projectId = useProjectId();
  const { data: s } = useStudio(projectId);
  if (!s) return <Spinner className="m-6" />;
  const tests = s.project.test_questions ?? [];
  const best = s.evals.length ? Math.max(...s.evals.map((e) => e.mean)) : null;
  return (
    <div className="mx-auto max-w-6xl space-y-5 p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Evaluate</h1>
          <p className="mt-1 max-w-3xl text-[13px] text-muted">
            Every checkpoint is scored on the same fixed test set (0–10 per case by the judge, or exact match where a case has an
            expected output). The best-scoring checkpoint is the one served, built on and exported.
          </p>
        </div>
        <LinkButton to={`/p/${projectId}/playground`} size="sm" className="shrink-0 whitespace-nowrap">
          Open the Playground →
        </LinkButton>
      </div>
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px]">
        <Card
          title="Scores and comparisons"
          subtitle={best != null ? `${s.evals.length} evaluations · best ${best.toFixed(1)} / 10` : "Nothing scored yet"}
        >
          <EvaluateView s={s} />
        </Card>
        <Card title="Test set" subtitle={`${tests.length} cases · ${tests.filter((q) => q.expected).length} with an expected output · fixed for the project`}>
          {tests.length ? (
            <ol className="list-decimal space-y-2 pl-5 text-xs">
              {tests.map((q) => (
                <li key={q.input} className={cx(q.kind === "should-not" && "italic")}>
                  <span className="text-fg">{q.input}</span>
                  {q.expected && <span className="mt-0.5 block font-mono text-[11px] break-words text-faint">→ {q.expected}</span>}
                  {q.kind === "should-not" && <span className="ml-1 text-faint">(should return nothing)</span>}
                </li>
              ))}
            </ol>
          ) : (
            <p className="text-xs text-muted">The Tuner writes the test set before any training data.</p>
          )}
        </Card>
      </div>
    </div>
  );
}
