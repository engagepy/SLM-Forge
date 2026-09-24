import { useQuery } from "@tanstack/react-query";
import { lazy, Suspense } from "react";
import { NavLink, Route, Routes, useMatch, useParams } from "react-router";

import { api, type Project } from "./api";
import { useJobsFeed, useOverview, useProjectFeed, useSystem } from "./hooks";
import SessionsSidebar from "./components/SessionsSidebar";
import { Badge, cx, Spinner } from "./ui";

// Route-level code splitting: charting code only loads with the training view.
const AgentsPage = lazy(() => import("./pages/Agents"));
const DataPage = lazy(() => import("./pages/Data"));
const ExportPage = lazy(() => import("./pages/Export"));
const FeedbackPage = lazy(() => import("./pages/Feedback"));
const Home = lazy(() => import("./pages/Home"));
const ModelPage = lazy(() => import("./pages/Model"));
const OverviewPage = lazy(() => import("./pages/Overview"));
const Playground = lazy(() => import("./pages/Playground"));
const Studio = lazy(() => import("./pages/Studio"));
const TrainPage = lazy(() => import("./pages/Train"));
const TryModel = lazy(() => import("./pages/TryModel"));
const StoragePage = lazy(() => import("./pages/Storage"));

export default function App() {
  useJobsFeed();
  return (
    <Suspense fallback={<Spinner className="m-6" />}>
      <Routes>
        {/* Home and the Studio share the sessions sidebar, so running work is always visible. */}
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

/** The detailed manual screens, reachable from the Studio's "Advanced" link. */
function AdvancedLayout() {
  return (
    <div className="flex h-full">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <StatusBar />
        <main className="min-h-0 flex-1 overflow-y-auto">
          <Suspense fallback={<Spinner className="m-6" />}>
            <ProjectShell />
          </Suspense>
        </main>
      </div>
    </div>
  );
}

function ProjectShell() {
  const projectId = Number(useParams().projectId);
  useProjectFeed(projectId);
  return (
    <Routes>
      <Route path="overview" element={<OverviewPage />} />
      <Route path="model" element={<ModelPage />} />
      <Route path="data" element={<DataPage />} />
      <Route path="train" element={<TrainPage />} />
      <Route path="train/:jobId" element={<TrainPage />} />
      <Route path="playground" element={<Playground />} />
      <Route path="feedback" element={<FeedbackPage />} />
      <Route path="agents" element={<AgentsPage />} />
      <Route path="export" element={<ExportPage />} />
    </Routes>
  );
}

const STEPS = [
  { to: "overview", label: "Overview", end: true },
  { to: "model", label: "Base model", n: 1 },
  { to: "data", label: "Data", n: 2 },
  { to: "train", label: "Train", n: 3 },
  { to: "playground", label: "Playground", n: 4 },
  { to: "feedback", label: "Feedback", n: 5 },
  { to: "agents", label: "Agents" },
  { to: "export", label: "Export", n: 6 },
];

function Sidebar() {
  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api.get<Project[]>("/api/projects") });
  // The sidebar sits outside the project route, so match the URL itself.
  const match = useMatch("/p/:projectId/*");
  const projectId = Number(match?.params.projectId);
  const overview = useOverview(projectId);
  const counts = overview.data?.counts;

  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-line bg-panel">
      <NavLink to="/" className="flex items-center gap-2 px-4 py-4">
        <span className="grid size-7 place-items-center rounded-lg bg-accent text-sm font-bold text-white">▲</span>
        <span className="font-semibold tracking-tight">SLM Forge</span>
      </NavLink>

      {Number.isFinite(projectId) && overview.data && (
        <nav className="px-2">
          <NavLink to={`/p/${projectId}`} className="mb-3 flex items-center gap-2 rounded-lg bg-accent-soft px-2 py-1.5 text-[13px] font-medium text-accent">
            ✦ Back to the Studio
          </NavLink>
          <div className="truncate px-2 pb-2 text-[11px] font-medium uppercase tracking-wide text-faint">{overview.data.project.name} · advanced</div>
          {STEPS.map((s) => (
            <NavLink
              key={s.to}
              to={`/p/${projectId}/${s.to}`}
              end={s.end}
              className={({ isActive }) =>
                cx(
                  "flex items-center gap-2 rounded-lg px-2 py-1.5 text-[13px] transition",
                  isActive ? "bg-accent-soft text-accent" : "text-muted hover:bg-panel-2 hover:text-fg",
                )
              }
            >
              <span className="num w-4 text-center text-[11px] text-faint">{s.n ?? "·"}</span>
              <span className="flex-1">{s.label}</span>
              {s.to === "agents" && !!counts?.pending_proposals && <Badge tone="accent">{counts.pending_proposals}</Badge>}
              {s.to === "feedback" && !!counts?.awaiting_review && <Badge tone="warn">{counts.awaiting_review}</Badge>}
            </NavLink>
          ))}
        </nav>
      )}

      <div className="mt-4 min-h-0 flex-1 overflow-y-auto border-t border-line px-2 pt-3">
        <div className="px-2 pb-1.5 text-[11px] font-medium uppercase tracking-wide text-faint">Projects</div>
        {projects.data?.map((p) => (
          <NavLink
            key={p.id}
            to={`/p/${p.id}`}
            className={cx(
              "block truncate rounded-lg px-2 py-1.5 text-[13px]",
              p.id === projectId ? "text-fg" : "text-muted hover:bg-panel-2 hover:text-fg",
            )}
          >
            {p.name}
          </NavLink>
        ))}
        <NavLink to="/" className="block rounded-lg px-2 py-1.5 text-[13px] text-accent hover:bg-panel-2">
          + New project
        </NavLink>
      </div>
    </aside>
  );
}

function StatusBar() {
  const { data } = useSystem();
  if (!data) return <div className="h-10 border-b border-line" />;
  const gpu = data.worker.running.gpu;
  const loaded = data.inference.loaded;
  return (
    <div className="flex h-10 shrink-0 items-center gap-4 border-b border-line px-5 text-xs text-muted">
      <span>
        <span className="text-fg">{data.hardware.chip}</span> · {data.hardware.total_memory_gb.toFixed(0)} GB ·{" "}
        <span className="num">{data.hardware.budget_gb.toFixed(1)} GB</span> usable for ML
      </span>
      <span className="h-4 w-px bg-line" />
      <span className="flex items-center gap-1.5">
        GPU:
        {gpu ? (
          <Badge tone="info">
            <span className="size-1.5 animate-pulse rounded-full bg-current" /> job {gpu}
          </Badge>
        ) : loaded ? (
          <Badge tone="good">serving {shortPath(loaded.model_path)}{loaded.adapter_path ? " + adapter" : ""}</Badge>
        ) : (
          <Badge>idle</Badge>
        )}
        {data.worker.queued.gpu > 0 && <span>+{data.worker.queued.gpu} queued</span>}
      </span>
      <span className="ml-auto flex items-center gap-1.5">
        Agents: {data.agents.provider} · <span className="font-mono">{data.agents.model}</span>
        {!data.agents.key_configured && (
          <Badge tone="warn" className="ml-1">
            no API key: set {data.agents.key_env} in .env
          </Badge>
        )}
      </span>
    </div>
  );
}

function shortPath(p: string) {
  const m = p.match(/models--([^/]+)--([^/]+)/);
  if (m) return `${m[1]}/${m[2]}`;
  const parts = p.split("/").filter(Boolean);
  return parts.slice(-2).join("/");
}
