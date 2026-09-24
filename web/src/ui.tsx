// Small, consistent building blocks. Everything reads colours from the tokens in index.css.
import { type ButtonHTMLAttributes, createContext, type ReactNode, useCallback, useContext, useEffect, useId, useRef, useState } from "react";
import { Link } from "react-router";

export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}

export function Card({
  title,
  subtitle,
  actions,
  children,
  className,
  pad = true,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children?: ReactNode;
  className?: string;
  pad?: boolean;
}) {
  return (
    <section className={cx("rounded-xl border border-line bg-panel", className)}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 border-b border-line px-4 py-3">
          <div className="min-w-0">
            {title && <h2 className="text-[13px] font-semibold tracking-tight">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={pad ? "p-4" : ""}>{children}</div>
    </section>
  );
}

type Variant = "primary" | "secondary" | "ghost" | "danger" | "good";
type Size = "sm" | "md";

const buttonStyles: Record<Variant, string> = {
  primary: "bg-accent text-white hover:brightness-110 border-transparent",
  secondary: "bg-panel-2 text-fg border-line hover:border-line-strong",
  ghost: "bg-transparent text-muted border-transparent hover:text-fg hover:bg-panel-2",
  danger: "bg-bad-soft text-bad border-transparent hover:brightness-110",
  good: "bg-good-soft text-good border-transparent hover:brightness-110",
};

/** The classes a button (or a link that looks like one) wears. */
export function buttonClass(variant: Variant = "secondary", size: Size = "md", className?: string) {
  return cx(
    "inline-flex items-center justify-center gap-1.5 rounded-lg border font-medium transition",
    "disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-accent",
    size === "sm" ? "h-7 px-2.5 text-xs" : "h-8.5 px-3.5 text-[13px]",
    buttonStyles[variant],
    className,
  );
}

export function Button({
  variant = "secondary",
  size = "md",
  loading,
  className,
  children,
  disabled,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: Size; loading?: boolean }) {
  return (
    <button className={buttonClass(variant, size, className)} disabled={disabled || loading} {...rest}>
      {loading && <Spinner />}
      {children}
    </button>
  );
}

/** A route link styled as a button (a button inside a link is invalid HTML and reads badly to screen readers). */
export function LinkButton({
  to,
  variant = "secondary",
  size = "md",
  className,
  children,
}: {
  to: string;
  variant?: Variant;
  size?: Size;
  className?: string;
  children: ReactNode;
}) {
  return (
    <Link to={to} className={buttonClass(variant, size, className)}>
      {children}
    </Link>
  );
}

/** One chat message: the person's on the right, the model's or the Tuner's on the left. */
export function Bubble({ role, children, className }: { role: "user" | "assistant"; children: ReactNode; className?: string }) {
  return (
    <div className={cx("flex", role === "user" ? "justify-end" : "justify-start")}>
      <div
        className={cx(
          "max-w-[85%] min-w-0 rounded-2xl px-4 py-2.5 text-[14px] leading-relaxed break-words",
          role === "user" ? "rounded-tr-sm bg-accent text-white whitespace-pre-wrap" : "rounded-tl-sm border border-line bg-panel",
          className,
        )}
      >
        {children}
      </div>
    </div>
  );
}

/** The small uppercase heading above a group of facts. */
export function SectionLabel({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cx("text-[11px] font-medium uppercase tracking-wide text-faint", className)}>{children}</div>;
}

export function ProgressBar({ pct, failed = false }: { pct: number; failed?: boolean }) {
  return (
    <div className="h-1.5 overflow-hidden rounded-full bg-panel-2">
      <div className={cx("h-full transition-all", failed ? "bg-bad" : "bg-accent")} style={{ width: `${pct}%` }} />
    </div>
  );
}

/** A small panel anchored to a trigger. Closes on an outside click or Escape. `align` is which
 * edge lines up with the trigger; `side` is where it opens. Trigger buttons should set aria-expanded. */
export function Popover({
  open,
  onClose,
  align = "right",
  side = "down",
  className,
  children,
}: {
  open: boolean;
  onClose: () => void;
  align?: "left" | "right";
  side?: "down" | "up";
  className?: string;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      const el = ref.current;
      if (el && !el.contains(e.target as Node) && !el.parentElement?.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div
      ref={ref}
      role="dialog"
      className={cx(
        "absolute z-20 w-72 max-w-[calc(100vw-2rem)] rounded-xl border border-line bg-panel p-3 text-xs shadow-lg",
        align === "right" ? "right-0" : "left-0",
        side === "down" ? "mt-1.5 top-full" : "mb-1.5 bottom-full",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      className={cx("inline-block size-3.5 animate-spin rounded-full border-2 border-current border-t-transparent", className)}
      aria-hidden
    />
  );
}

type Tone = "neutral" | "accent" | "good" | "warn" | "bad" | "info";

export function Badge({ tone = "neutral", children, className }: { tone?: Tone; children: ReactNode; className?: string }) {
  const tones: Record<Tone, string> = {
    neutral: "bg-panel-2 text-muted border-line",
    accent: "bg-accent-soft text-accent border-transparent",
    good: "bg-good-soft text-good border-transparent",
    warn: "bg-warn-soft text-warn border-transparent",
    bad: "bg-bad-soft text-bad border-transparent",
    info: "bg-info-soft text-info border-transparent",
  };
  return (
    <span className={cx("inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-medium", tones[tone], className)}>
      {children}
    </span>
  );
}

const statusTone: Record<string, Tone> = {
  queued: "neutral",
  running: "info",
  succeeded: "good",
  executed: "good",
  failed: "bad",
  cancelled: "warn",
  rejected: "neutral",
  pending: "accent",
  approved: "info",
};

export function StatusBadge({ status }: { status: string }) {
  return (
    <Badge tone={statusTone[status] ?? "neutral"}>
      {status === "running" && <span className="size-1.5 animate-pulse rounded-full bg-current" />}
      {status}
    </Badge>
  );
}

export function Field({ label, hint, children }: { label: ReactNode; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 flex items-baseline justify-between gap-2 text-xs font-medium text-muted">
        {label}
        {hint && <span className="font-normal text-faint">{hint}</span>}
      </span>
      {children}
    </label>
  );
}

const inputCls =
  "w-full rounded-lg border border-line bg-bg px-2.5 py-1.5 text-[13px] text-fg placeholder:text-faint " +
  "focus:border-accent focus:outline-none disabled:opacity-60";

export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={cx(inputCls, props.className)} />;
}

export function TextArea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...props} className={cx(inputCls, "resize-y leading-relaxed", props.className)} />;
}

export function Select<T extends string>({
  value,
  onChange,
  options,
  className,
}: {
  value: T;
  onChange: (v: T) => void;
  options: readonly (T | { value: T; label: string })[];
  className?: string;
}) {
  return (
    <select className={cx(inputCls, "pr-7", className)} value={value} onChange={(e) => onChange(e.target.value as T)}>
      {options.map((o) => {
        const v = typeof o === "string" ? o : o.value;
        return (
          <option key={v} value={v}>
            {typeof o === "string" ? o : o.label}
          </option>
        );
      })}
    </select>
  );
}

/** Slider with a synced numeric input: fine control plus exact entry. */
export function Slider({
  label,
  hint,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: ReactNode;
  hint?: ReactNode;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
}) {
  const id = useId();
  return (
    <div>
      <div className="mb-1 flex items-center justify-between gap-2">
        <label htmlFor={id} className="text-xs font-medium text-muted">
          {label}
        </label>
        <input
          type="number"
          className="num w-20 rounded-md border border-line bg-bg px-1.5 py-0.5 text-right text-xs focus:border-accent focus:outline-none"
          value={value}
          step={step}
          min={min}
          max={max}
          onChange={(e) => e.target.value !== "" && onChange(Number(e.target.value))}
        />
      </div>
      <input id={id} type="range" className="w-full" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      {hint && <p className="mt-0.5 text-[11px] leading-snug text-faint">{hint}</p>}
    </div>
  );
}

export function NumberField({
  label,
  hint,
  value,
  onChange,
  step = 1,
  min,
  max,
  allowEmpty,
}: {
  label: ReactNode;
  hint?: ReactNode;
  value: number | null;
  onChange: (v: number | null) => void;
  step?: number;
  min?: number;
  max?: number;
  allowEmpty?: boolean;
}) {
  return (
    <Field label={label} hint={hint}>
      <Input
        type="number"
        className="num"
        value={value ?? ""}
        step={step}
        min={min}
        max={max}
        placeholder={allowEmpty ? "off" : undefined}
        onChange={(e) => {
          if (e.target.value === "") return allowEmpty ? onChange(null) : undefined;
          onChange(Number(e.target.value));
        }}
      />
    </Field>
  );
}

export function Toggle({ label, hint, checked, onChange }: { label: ReactNode; hint?: ReactNode; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-start gap-2.5">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cx("relative mt-0.5 h-4.5 w-8 shrink-0 rounded-full transition", checked ? "bg-accent" : "bg-line-strong")}
      >
        <span className={cx("absolute top-0.5 size-3.5 rounded-full bg-white transition-all", checked ? "left-4" : "left-0.5")} />
      </button>
      <span>
        <span className="text-[13px]">{label}</span>
        {hint && <span className="block text-[11px] text-faint">{hint}</span>}
      </span>
    </label>
  );
}

// Static class names: Tailwind only generates classes it can find literally in source.
const toneText: Record<Tone, string> = {
  neutral: "",
  accent: "text-accent",
  good: "text-good",
  warn: "text-warn",
  bad: "text-bad",
  info: "text-info",
};

export function Stat({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: Tone }) {
  const color = tone ? toneText[tone] : "";
  return (
    <div className="rounded-lg border border-line bg-panel-2/50 px-3 py-2.5">
      <div className="text-[11px] font-medium uppercase tracking-wide text-faint">{label}</div>
      <div className={cx("num mt-0.5 text-lg font-semibold tracking-tight", color)}>{value}</div>
      {sub && <div className="text-[11px] text-muted">{sub}</div>}
    </div>
  );
}

export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-line px-6 py-10 text-center">
      <p className="text-sm font-medium">{title}</p>
      {children && <div className="mt-1 max-w-md text-xs text-muted">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  if (!error) return null;
  return <div className="rounded-lg border border-bad/30 bg-bad-soft px-3 py-2 text-xs text-bad">{String((error as Error).message ?? error)}</div>;
}

export function Collapsible({ title, children }: { title: ReactNode; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-lg border border-line">
      <button type="button" onClick={() => setOpen(!open)} className="flex w-full items-center justify-between px-3 py-2 text-left text-xs font-medium text-muted hover:text-fg">
        {title}
        <span className={cx("transition", open && "rotate-90")}>›</span>
      </button>
      {open && <div className="border-t border-line p-3">{children}</div>}
    </div>
  );
}

/** A command or path the user will want to paste somewhere: monospace, with a copy button. */
export function CodeBlock({ text, label, className }: { text: string; label?: ReactNode; className?: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <div className={className}>
      {label && <div className="mb-0.5 text-[11px] text-faint">{label}</div>}
      <div className="group flex items-start gap-1 rounded-md bg-bg pr-1">
        <pre className="min-w-0 flex-1 overflow-x-auto py-2 pl-3 font-mono text-[11.5px] text-muted">{text}</pre>
        <button
          type="button"
          onClick={copy}
          title="Copy"
          className={cx(
            "mt-1 shrink-0 rounded px-1.5 py-0.5 text-[11px] transition",
            copied ? "bg-good-soft text-good" : "bg-panel-2 text-faint opacity-60 group-hover:opacity-100 hover:text-fg",
          )}
        >
          {copied ? "copied" : "copy"}
        </button>
      </div>
    </div>
  );
}

export function Mono({ children, className }: { children: ReactNode; className?: string }) {
  return <code className={cx("rounded bg-panel-2 px-1 py-0.5 font-mono text-[12px]", className)}>{children}</code>;
}

/** Horizontal memory bar: stacked components against the budget. */
export function MemoryBar({ est }: { est: { weights_gb: number; trainable_gb: number; activations_gb: number; logits_gb: number; overhead_gb: number; total_gb: number; budget_gb: number; fits: boolean } }) {
  const scale = Math.max(est.budget_gb, est.total_gb) * 1.05;
  const parts = [
    { k: "Weights", v: est.weights_gb, c: "var(--chart-2)" },
    { k: "Adapter+optimizer", v: est.trainable_gb, c: "var(--chart-4)" },
    { k: "Activations", v: est.activations_gb, c: "var(--chart-1)" },
    { k: "Logits", v: est.logits_gb, c: "var(--chart-3)" },
    { k: "Runtime", v: est.overhead_gb, c: "var(--faint)" },
  ];
  return (
    <div>
      <div className="relative h-3 overflow-hidden rounded-full bg-panel-2">
        <div className="flex h-full">
          {parts.map((p) => (
            <div key={p.k} style={{ width: `${(p.v / scale) * 100}%`, background: p.c }} title={`${p.k}: ${p.v.toFixed(2)} GB`} />
          ))}
        </div>
        <div className="absolute inset-y-0 w-0.5 bg-fg" style={{ left: `${(est.budget_gb / scale) * 100}%` }} title="Budget" />
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted">
        {parts.map((p) => (
          <span key={p.k} className="inline-flex items-center gap-1">
            <span className="size-2 rounded-sm" style={{ background: p.c }} />
            {p.k} <span className="num">{p.v.toFixed(2)}</span>
          </span>
        ))}
        <span className={cx("num ml-auto font-medium", est.fits ? "text-good" : "text-bad")}>
          {est.total_gb.toFixed(2)} / {est.budget_gb.toFixed(1)} GB {est.fits ? "fits" : "won't fit"}
        </span>
      </div>
    </div>
  );
}

// ── Toasts ───────────────────────────────────────────────────────────────────

export interface ToastInput {
  tone?: "good" | "bad" | "info";
  title: string;
  body?: string;
  action?: { label: string; to: string };
}

interface ToastItem extends ToastInput {
  id: number;
}

const ToastContext = createContext<(t: ToastInput) => void>(() => {});

export function useToast() {
  return useContext(ToastContext);
}

/** Brief, stacked notifications (bottom-right). Failures stay until dismissed. */
export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const nextId = useRef(1);
  const dismiss = useCallback((id: number) => setItems((xs) => xs.filter((x) => x.id !== id)), []);
  const push = useCallback(
    (t: ToastInput) => {
      const id = nextId.current++;
      setItems((xs) => [...xs.slice(-3), { ...t, id }]);
      if (t.tone !== "bad") setTimeout(() => dismiss(id), 7000);
    },
    [dismiss],
  );
  const bar = { good: "bg-good", bad: "bg-bad", info: "bg-info" };
  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="pointer-events-none fixed right-4 bottom-4 z-50 flex w-80 flex-col gap-2" aria-live="polite">
        {items.map((t) => (
          <div key={t.id} className="pointer-events-auto flex overflow-hidden rounded-xl border border-line bg-panel shadow-lg shadow-black/30">
            <span className={cx("w-1 shrink-0", bar[t.tone ?? "info"])} />
            <div className="min-w-0 flex-1 px-3 py-2.5">
              <div className="flex items-start gap-2">
                <p className="flex-1 text-[13px] font-medium">{t.title}</p>
                <button className="text-faint hover:text-fg" onClick={() => dismiss(t.id)} aria-label="Dismiss">
                  ×
                </button>
              </div>
              {t.body && <p className="mt-0.5 text-xs text-muted">{t.body}</p>}
              {t.action && (
                <Link to={t.action.to} onClick={() => dismiss(t.id)} className="mt-1.5 inline-block text-xs font-medium text-accent hover:underline">
                  {t.action.label} →
                </Link>
              )}
            </div>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

/** Briefly highlight an element (by DOM id) after scrolling it into view. */
export function useSpotlight(): [string | null, (domId: string) => void] {
  const [active, setActive] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const spotlight = useCallback((domId: string) => {
    setActive(domId);
    // Wait a frame so newly rendered rows exist before scrolling.
    requestAnimationFrame(() =>
      requestAnimationFrame(() => document.getElementById(domId)?.scrollIntoView({ behavior: "smooth", block: "center" })),
    );
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setActive(null), 2500);
  }, []);
  return [active, spotlight];
}
