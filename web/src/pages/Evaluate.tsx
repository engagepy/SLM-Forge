// Advanced · Evaluate: the Studio's evaluate stage at full width, with the fixed test set beside it.
import { NextStage, PAGE, PageHeader } from "../components/Page";
import { useProjectId, useStudio } from "../hooks";
import { Card, EmptyNote, LinkButton, Spinner } from "../ui";
import { TestSetList, testSetSummary } from "./studio/bits";
import { EvaluateView } from "./studio/Evaluate";

export default function EvaluatePage() {
  const projectId = useProjectId();
  const { data: s } = useStudio(projectId);
  if (!s) return <Spinner className="m-6" />;
  const tests = s.project.test_questions ?? [];
  const best = s.evals.length ? Math.max(...s.evals.map((e) => e.mean)) : null;
  return (
    <div className={PAGE}>
      <PageHeader
        title="Evaluate"
        actions={
          <>
            <LinkButton to={`/p/${projectId}/playground`}>Open the Playground →</LinkButton>
            <NextStage stage="evaluate" />
          </>
        }
      >
        Every checkpoint is scored on the same fixed test set (0–10 per case by the judge, or exact match where a case has an
        expected output). The best-scoring checkpoint is the one served, built on and exported.
      </PageHeader>
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px]">
        <Card
          title="Scores and comparisons"
          subtitle={best != null ? `${s.evals.length} evaluations · best ${best.toFixed(1)} / 10` : "Nothing scored yet"}
        >
          <EvaluateView s={s} />
        </Card>
        <Card title="Test set" subtitle={`${testSetSummary(tests)} · fixed for the project`}>
          {tests.length ? (
            <TestSetList tests={tests} />
          ) : (
            <EmptyNote>The Tuner writes the test set before any training data.</EmptyNote>
          )}
        </Card>
      </div>
    </div>
  );
}
