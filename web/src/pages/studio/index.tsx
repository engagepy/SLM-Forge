// The Studio: the Tuner chat on the left, the live canvas of pipeline stages on the right.
import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { useParams } from "react-router";
import { api, type Snapshot, type TunerMessage } from "../../api";
import MetricsBar from "../../components/MetricsBar";
import ProjectHeader from "../../components/ProjectHeader";
import { useStudio, useTunerStream } from "../../hooks";
import { Spinner } from "../../ui";
import { Canvas } from "./Canvas";
import { Chat } from "./Chat";

// ── page ─────────────────────────────────────────────────────────────────────

export default function Studio() {
  const projectId = Number(useParams().projectId);
  const snapshot = useStudio(projectId);
  const messages = useQuery({
    queryKey: ["tuner-messages", projectId],
    queryFn: () => api.get<TunerMessage[]>(`/api/projects/${projectId}/tuner/messages`),
  });
  const { streaming, busy, setBusy } = useTunerStream(projectId);

  // Opening a project never starts anything: the Tuner only begins from an explicit action
  // (creating the project on Home, or the Start button below).
  useEffect(() => {
    if (snapshot.data) setBusy((b) => b || snapshot.data!.tuner_busy);
  }, [snapshot.data, setBusy]);

  return (
    <div className="flex h-full flex-col">
      <TopBar snapshot={snapshot.data} />
      {/* Side by side from lg; stacked on a phone, each half scrolling on its own. */}
      <div className="grid min-h-0 flex-1 grid-cols-1 grid-rows-[minmax(0,1fr)_minmax(0,1fr)] lg:grid-cols-2 lg:grid-rows-1">
        <Chat
          projectId={projectId}
          messages={messages.data ?? []}
          loaded={messages.isSuccess}
          streaming={streaming}
          busy={busy}
          pending={snapshot.data?.pending_action ?? null}
        />
        {snapshot.data ? <Canvas snapshot={snapshot.data} messages={messages.data ?? []} /> : <div className="grid place-items-center"><Spinner /></div>}
      </div>
    </div>
  );
}

function TopBar({ snapshot }: { snapshot?: Snapshot }) {
  return (
    <>
      <MetricsBar snapshot={snapshot} view="studio" />
      <ProjectHeader snapshot={snapshot} />
    </>
  );
}
