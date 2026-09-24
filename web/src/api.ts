// Typed client for the SLM Forge API.

type Json = Record<string, unknown>;

interface Hardware {
  chip: string;
  total_memory_gb: number;
  gpu_working_set_gb: number;
  budget_gb: number;
  cpu_cores: number;
  macos: string;
}

export interface SystemStatus {
  hardware: Hardware;
  max_params_4bit: number;
  worker: {
    running: Record<"gpu" | "io" | "agent", number | null>;
    queued: Record<"gpu" | "io" | "agent", number>;
    inference_blocked: string | null;
  };
  inference: { loaded: { model_path: string; adapter_path: string | null } | null; blocked: string | null };
  agents: { provider: string; model: string; key_configured: boolean; key_env: string | null };
  workspace: string;
}

export interface Project {
  id: number;
  name: string;
  goal: string;
  base_model: string | null;
  system_prompt: string;
  current_model_path: string | null;
  current_adapter_path: string | null;
  created_at: string;
}

export interface Checkpoint {
  id: number;
  project_id: number;
  job_id: number | null;
  parent_id: number | null;
  kind: "sft" | "dpo";
  base_model_path: string;
  adapter_path: string;
  fused_path: string | null;
  metrics: Record<string, number>;
  created_at: string;
}

export interface Overview {
  project: Project;
  base_model_downloaded: boolean;
  counts: {
    datasets: number;
    dataset_versions: number;
    feedback: number;
    pairs_ready: number;
    sft_ready: number;
    awaiting_review: number;
    pending_proposals: number;
  };
  checkpoints: Checkpoint[];
}

export interface ModelCandidate {
  id: string;
  downloads: number;
  likes: number;
  params: number;
  bits: number;
  weights_gb: number;
  train_estimate_gb: number;
  fit: "fits" | "tight" | "too_big" | "unknown";
  gated: boolean;
  is_mlx: boolean;
  architecture: string;
  last_modified: string;
}

export interface MemoryEstimate {
  weights_gb: number;
  trainable_gb: number;
  activations_gb: number;
  logits_gb: number;
  overhead_gb: number;
  total_gb: number;
  budget_gb: number;
  fits: boolean;
  headroom_gb: number;
}

export interface TrainConfig {
  mode: "sft" | "dpo";
  fine_tune_type: "lora" | "dora" | "full";
  optimizer: "adam" | "adamw";
  weight_decay: number;
  learning_rate: number;
  lr_schedule: "constant" | "cosine" | "linear";
  warmup_steps: number;
  min_lr_ratio: number;
  lora_rank: number;
  lora_scale: number;
  lora_dropout: number;
  lora_keys: string[] | null;
  num_layers: number;
  batch_size: number;
  grad_accumulation_steps: number;
  epochs: number | null;
  iters: number | null;
  max_seq_length: number;
  grad_checkpoint: boolean;
  mask_prompt: boolean;
  seed: number;
  steps_per_report: number;
  steps_per_eval: number;
  save_every: number;
  val_batches: number;
  beta: number;
  dpo_loss_type: "sigmoid" | "hinge" | "ipo" | "dpop";
  dpop_delta: number;
}

export interface Preset {
  config: TrainConfig;
  estimate: MemoryEstimate;
}

export interface SamplingParams {
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number;
  repetition_penalty: number | null;
  repetition_context_size: number;
  presence_penalty: number | null;
  frequency_penalty: number | null;
  xtc_probability: number;
  xtc_threshold: number;
  max_tokens: number;
  seed: number | null;
  chat_template?: string | null;
  enable_thinking?: boolean | null;
}

export const DEFAULT_SAMPLING: SamplingParams = {
  temperature: 0.7,
  top_p: 0.95,
  top_k: 0,
  min_p: 0,
  repetition_penalty: 1.05,
  repetition_context_size: 64,
  presence_penalty: null,
  frequency_penalty: null,
  xtc_probability: 0,
  xtc_threshold: 0,
  max_tokens: 512,
  seed: null,
};

export interface Dataset {
  id: number;
  project_id: number;
  name: string;
  source: string;
  source_ref: string;
  license: string;
  raw_path: string;
  n_rows: number;
  columns: string[];
  created_at: string;
  suggested_mapping: Mapping;
}

export type Mapping = { format: "chat" | "instruction" | "text" | "preference" | "unknown" } & Record<string, string>;

export interface DatasetVersion {
  id: number;
  dataset_id: number | null;
  kind: "sft" | "dpo";
  path: string;
  n_train: number;
  n_valid: number;
  n_test: number;
  mapping: Json;
  cleaning_report: { input_rows?: number; kept?: number; dropped?: Record<string, number> };
  token_stats: { mean?: number; p50?: number; p95?: number; max?: number; total_tokens?: number; over_max_seq_length?: number };
  created_at: string;
}

type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface Job {
  id: number;
  project_id: number | null;
  kind: string;
  status: JobStatus;
  config: Json;
  result: Json & { progress?: { current: number; total: number }; metrics?: Record<string, number> };
  error: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface MetricPoint {
  iteration: number;
  split: "train" | "val";
  values: Record<string, number>;
}

export interface Proposal {
  id: number;
  project_id: number;
  agent: string;
  action: string;
  title: string;
  rationale: string;
  payload: Json;
  status: "pending" | "approved" | "rejected" | "executed" | "failed";
  result: Json;
  created_at: string;
}

export interface AgentEvent {
  id: number;
  project_id: number;
  agent: string;
  kind: "message" | "tool_call" | "tool_result" | "proposal" | "error";
  content: Json;
  created_at: string;
}

export interface PreferencePair {
  id: number;
  prompt: string;
  system: string;
  chosen: string;
  rejected: string;
  source: string;
  approved: boolean;
  used_in_job_id: number | null;
}

export interface SftExample {
  id: number;
  messages: { role: string; content: string }[];
  source: string;
  approved: boolean;
  used_in_job_id: number | null;
}

// ── the Studio ──────────────────────────────────────────────────────────────

export interface TunerMessage {
  id: number;
  role: "user" | "assistant" | "event" | "tool";
  content: string;
  meta: {
    name?: string;
    args?: Record<string, unknown>;
    status?: string;
    output?: string;
    error?: boolean;
    kickoff?: boolean;
    autopilot?: boolean;
    autopilot_paused?: boolean;
    confirmed?: string;
    declined?: string;
  } & Record<string, unknown>;
  created_at: string;
}

export interface Comparison {
  id: string;
  prompt: string;
  a: string;
  b: string;
  status: "pending" | "judged";
  choice?: string;
  judge?: "ai";
  critique?: string;
  ideal?: string;
}

export interface Sample {
  prompt: string;
  target: string;
  text: string;
  tokens_per_sec?: number;
}

/** A run the Tuner proposed; nothing starts until the user confirms it. */
export interface PendingAction {
  id: string;
  kind: "model" | "sft" | "dpo" | "export" | "synthesize" | "review" | "import";
  title: string;
  reason: string;
  details: Record<string, unknown>;
  created_at: string;
}

export interface Snapshot {
  project: Project;
  pending_action: PendingAction | null;
  stage: Stage;
  note: string;
  tuner_busy: boolean;
  autopilot: boolean;
  completed: boolean;
  model: { repo_id: string; downloaded: boolean; params?: number; bits?: number; size_gb?: number; layers?: number; inference?: MemoryEstimate } | null;
  datasets: Dataset[];
  versions: DatasetVersion[];
  jobs: Job[];
  checkpoints: Checkpoint[];
  samples: Sample[];
  comparisons: Comparison[];
  feedback: { judgements: number; pairs_ready: number };
  exports: { job_id: number; path: string; size_gb: number; min_ram_gb: number }[];
}

export const STAGES = ["goal", "data", "model", "train", "evaluate", "refine", "export"] as const;
export type Stage = (typeof STAGES)[number];

export const isActive = (j: Job) => j.status === "running" || j.status === "queued";

/** Is the project serving this checkpoint (its adapter, or its fused model)? */
export const isServing = (p: Project, c: Checkpoint) =>
  p.current_adapter_path === c.adapter_path || (!!c.fused_path && p.current_model_path === c.fused_path);

/** A finished model, from GET /projects/{id}/exports. */
export interface ExportRow {
  job_id: number;
  name: string;
  path: string;
  size_gb: number;
  min_ram_gb: number;
  on_disk: boolean;
  created_at: string | null;
}

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, headers: {} };
  if (body instanceof FormData) {
    init.body = body;
  } else if (body !== undefined) {
    init.body = JSON.stringify(body);
    (init.headers as Record<string, string>)["content-type"] = "application/json";
  }
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? data);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  get: <T>(url: string) => request<T>("GET", url),
  post: <T>(url: string, body?: unknown) => request<T>("POST", url, body ?? {}),
  patch: <T>(url: string, body: unknown) => request<T>("PATCH", url, body),
  delete: <T>(url: string) => request<T>("DELETE", url),
  upload: <T>(url: string, form: FormData) => request<T>("POST", url, form),
};

interface SseEvent {
  type: string;
  [k: string]: unknown;
}

/** POST a request and read its text/event-stream response (EventSource only supports GET). */
export async function postStream(
  url: string,
  body: unknown,
  onEvent: (ev: SseEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, String(detail));
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const data = frame
        .split(/\r?\n/)
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trimStart())
        .join("\n");
      if (data) onEvent(JSON.parse(data) as SseEvent);
    }
  }
}

export const fmt = {
  params: (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B` : `${Math.round(n / 1e6)}M`),
  gb: (n: number | undefined) => (n == null ? "–" : `${n.toFixed(n < 10 ? 2 : 1)} GB`),
  num: (n: number | undefined, digits = 3) => (n == null || Number.isNaN(n) ? "–" : Number(n.toFixed(digits)).toString()),
  compact: (n: number) => Intl.NumberFormat("en", { notation: "compact" }).format(n),
  ago: (iso: string | null | undefined) => {
    if (!iso) return "";
    const s = (Date.now() - new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z").getTime()) / 1000;
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
    return `${Math.floor(s / 86400)}d ago`;
  },
  duration: (a: string | null, b: string | null) => {
    if (!a) return "";
    const end = b ? new Date(b + (b.endsWith("Z") ? "" : "Z")) : new Date();
    const s = Math.max(0, (end.getTime() - new Date(a + (a.endsWith("Z") ? "" : "Z")).getTime()) / 1000);
    return s < 60 ? `${Math.round(s)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  },
};
