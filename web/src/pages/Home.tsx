import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate } from "react-router";

import { api, fmt, type Project } from "../api";
import { useSystem } from "../hooks";
import { Button, Card, ErrorNote, Field, Input, Stat, TextArea } from "../ui";

export default function Home() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const system = useSystem();
  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api.get<Project[]>("/api/projects") });
  const [form, setForm] = useState({ name: "", goal: "", system_prompt: "" });
  const create = useMutation({
    mutationFn: () => api.post<Project>("/api/projects", form),
    onSuccess: (p) => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      navigate(`/p/${p.id}/model`);
    },
  });

  const hw = system.data?.hardware;
  return (
    <div className="mx-auto max-w-5xl space-y-6 p-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Build a small language model on this Mac</h1>
        <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-muted">
          Pick a base model that fits your hardware, let agents find and clean training data, fine-tune with MLX, then
          sharpen it with your own feedback. Everything runs locally; only the agents call out to an LLM.
        </p>
      </div>

      {hw && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat label="Chip" value={hw.chip.replace("Apple ", "")} sub={`${hw.cpu_cores} CPU cores · macOS ${hw.macos}`} />
          <Stat label="Unified memory" value={`${hw.total_memory_gb.toFixed(0)} GB`} sub={`GPU may use ${hw.gpu_working_set_gb.toFixed(1)} GB`} />
          <Stat label="Training budget" value={fmt.gb(hw.budget_gb)} sub="after headroom for macOS" tone="accent" />
          <Stat
            label="Largest trainable"
            value={`~${fmt.params(system.data!.max_params_4bit)}`}
            sub="params with 4-bit LoRA; smaller is faster"
          />
        </div>
      )}

      <div className="grid gap-6 md:grid-cols-5">
        <Card title="New project" subtitle="One project = one model you're building." className="md:col-span-3">
          <form
            className="space-y-3"
            onSubmit={(e) => {
              e.preventDefault();
              create.mutate();
            }}
          >
            <Field label="Name">
              <Input required value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="kitchen-helper" />
            </Field>
            <Field label="Goal" hint="Agents read this to find data and judge progress">
              <TextArea
                rows={3}
                required
                value={form.goal}
                onChange={(e) => setForm({ ...form, goal: e.target.value })}
                placeholder="Answer home-cooking questions concisely, with exact times and temperatures, in a friendly tone."
              />
            </Field>
            <Field label="System prompt" hint="optional, used for training examples and chat">
              <TextArea
                rows={2}
                value={form.system_prompt}
                onChange={(e) => setForm({ ...form, system_prompt: e.target.value })}
                placeholder="You are a concise, friendly cooking assistant."
              />
            </Field>
            <ErrorNote error={create.error} />
            <Button type="submit" variant="primary" loading={create.isPending}>
              Create and choose a base model →
            </Button>
          </form>
        </Card>

        <Card title="Projects" className="md:col-span-2" pad={false}>
          {projects.data?.length ? (
            <ul className="divide-y divide-line">
              {projects.data.map((p) => (
                <li key={p.id}>
                  <Link to={`/p/${p.id}`} className="block px-4 py-3 hover:bg-panel-2">
                    <div className="text-[13px] font-medium">{p.name}</div>
                    <div className="mt-0.5 line-clamp-2 text-xs text-muted">{p.goal || "No goal set"}</div>
                    <div className="mt-1 font-mono text-[11px] text-faint">{p.base_model ?? "no base model yet"}</div>
                  </Link>
                </li>
              ))}
            </ul>
          ) : (
            <p className="p-4 text-xs text-muted">No projects yet.</p>
          )}
        </Card>
      </div>
    </div>
  );
}
