import { lazy, Suspense } from "react";
import { NavLink, Route, Routes, useMatch, useParams } from "react-router";

import { STAGES } from "./api";
import MetricsBar from "./components/MetricsBar";
import ProjectHeader from "./components/ProjectHeader";
import SessionsSidebar from "./components/SessionsSidebar";
import { ADVANCED_PAGE, StageStepper } from "./components/StageStepper";
import { useJobsFeed, useOverview, useProjectFeed, useStudio } from "./hooks";
import { Badge, cx, Spinner } from "./ui";

// Route-level code splitting: charting code only loads with the training view.
const AgentsPage = lazy(() => import("./pages/Agents"));
const DataPage = lazy(() => import("./pages/Data"));
const EvaluatePage = lazy(() => import("./pages/Evaluate"));
const ExportPage = lazy(() => import("./pages/Export"));
const FeedbackPage = lazy(() => import("./pages/Feedback"));
const Home = lazy(() => import("./pages/Home"));
const ModelPage = lazy(() => import("./pages/Model"));
const OverviewPage = lazy(() => import("./pages/Overview"));
const Playground = lazy(() => import("./pages/Playground"));
const Studio = lazy(() => import("./pages/studio/index"));
const TrainPage = lazy(() => import("./pages/Train"));
const TryModel = lazy(() => import("./pages/TryModel"));
const StoragePage = lazy(() => import("./pages/Storage"));

export default function App() {
  useJobsFeed();
  return (
    <Suspense fallback={<Spinner className="m-6" />}>
      <Routes>
        {/* Every screen has the sessions sidebar and the top bar; project screens add the Studio | Advanced | Try it switch. */}
        <Route path="/" element={<WithSessions><Home /></WithSessions>} />
        <Route path="/p/:projectId" element={<WithSessions><Studio /></WithSessions>} />
        <Route path="/storage" element={<WithSessions><StoragePage /></WithSessions>} />
        {/* The finished, exported model: where a project ends up. */}
        <Route path="/p/:projectId/try" element={<WithSessions><TryModel /></WithSessions>} />
        <Route path="/p/:projectId/*" element={<AdvancedLayout />} />
      </Routes>
    </Suspense>
  );
}

function WithSessions({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full">
      <SessionsSidebar />
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}

/** The expert screens: the same frame as the Studio (sidebar, metrics bar, project header, the
 * stage stepper with its ticks), with every setting and number exposed below it. */
function AdvancedLayout() {
  return (
    <WithSessions>
      <AdvancedShell />
    </WithSessions>
  );
}

function AdvancedShell() {
  const projectId = Number(useParams().projectId);
  useProjectFeed(projectId);
  const { data: s } = useStudio(projectId);
  const { data: ov } = useOverview(projectId);
  const page = useMatch("/p/:projectId/:page/*")?.params.page;
  const current = STAGES.find((st) => ADVANCED_PAGE[st] === page) ?? null;
  const tool = (to: string) =>
    cx("rounded-md px-2 py-1 text-[12px] font-medium transition", page === to ? "bg-accent-soft text-accent" : "text-muted hover:bg-panel-2 hover:text-fg");
  return (
    <div className="flex h-full flex-col">
      <MetricsBar snapshot={s} view="advanced" />
      <ProjectHeader snapshot={s}>
        <nav className="flex shrink-0 items-center gap-1" aria-label="Tools">
          <NavLink to={`/p/${projectId}/playground`} className={tool("playground")} title="Chat with any checkpoint or the base model">
            Playground
          </NavLink>
          <NavLink to={`/p/${projectId}/agents`} className={tool("agents")} title="Proposals from the data agents">
            Agents
            {!!ov?.counts.pending_proposals && (
              <Badge tone="accent" className="ml-1.5">
                {ov.counts.pending_proposals}
              </Badge>
            )}
          </NavLink>
        </nav>
      </ProjectHeader>
      <div className="shrink-0 border-b border-line px-5 py-2.5">
        {s ? <StageStepper s={s} current={current} linkTo={(st) => `/p/${projectId}/${ADVANCED_PAGE[st]}`} /> : <div className="h-6" />}
      </div>
      <main className="min-h-0 flex-1 overflow-y-auto">
        <Suspense fallback={<Spinner className="m-6" />}>
          <Routes>
            <Route path="overview" element={<OverviewPage />} />
            <Route path="model" element={<ModelPage />} />
            <Route path="data" element={<DataPage />} />
            <Route path="train" element={<TrainPage />} />
            <Route path="train/:jobId" element={<TrainPage />} />
            <Route path="evaluate" element={<EvaluatePage />} />
            <Route path="playground" element={<Playground />} />
            <Route path="feedback" element={<FeedbackPage />} />
            <Route path="agents" element={<AgentsPage />} />
            <Route path="export" element={<ExportPage />} />
          </Routes>
        </Suspense>
      </main>
    </div>
  );
}
