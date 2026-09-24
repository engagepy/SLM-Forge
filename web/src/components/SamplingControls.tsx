import type { SamplingParams } from "../api";
import { Collapsible, NumberField, Slider, Toggle } from "../ui";

/** Every generation knob MLX exposes, with the common ones up front. */
export default function SamplingControls({
  value,
  onChange,
  compact,
}: {
  value: SamplingParams;
  onChange: (v: SamplingParams) => void;
  compact?: boolean;
}) {
  const set = <K extends keyof SamplingParams>(k: K, v: SamplingParams[K]) => onChange({ ...value, [k]: v });
  return (
    <div className="space-y-3.5">
      <Slider
        label="Temperature"
        hint={compact ? undefined : "0 = always the most likely token. Higher = more varied."}
        value={value.temperature}
        onChange={(v) => set("temperature", v)}
        min={0}
        max={2}
        step={0.05}
      />
      <Slider
        label="Top-p"
        hint={compact ? undefined : "Sample only from the smallest token set whose probability sums to p."}
        value={value.top_p}
        onChange={(v) => set("top_p", v)}
        min={0}
        max={1}
        step={0.01}
      />
      <Slider label="Max tokens" value={value.max_tokens} onChange={(v) => set("max_tokens", v)} min={16} max={4096} step={16} />
      <Collapsible title="More sampling controls">
        <div className="space-y-3.5">
          <Slider label="Top-k" hint="0 = off" value={value.top_k} onChange={(v) => set("top_k", v)} min={0} max={200} step={1} />
          <Slider
            label="Min-p"
            hint="Drop tokens below min-p × the top token's probability. 0.05–0.1 tames high temperatures."
            value={value.min_p}
            onChange={(v) => set("min_p", v)}
            min={0}
            max={0.5}
            step={0.01}
          />
          <div className="grid grid-cols-2 gap-3">
            <NumberField
              label="Repetition penalty"
              hint="1 = off"
              value={value.repetition_penalty}
              onChange={(v) => set("repetition_penalty", v)}
              step={0.01}
              min={0.5}
              max={2}
              allowEmpty
            />
            <NumberField
              label="…over last N tokens"
              value={value.repetition_context_size}
              onChange={(v) => set("repetition_context_size", v ?? 64)}
              min={1}
            />
            <NumberField
              label="Presence penalty"
              value={value.presence_penalty}
              onChange={(v) => set("presence_penalty", v)}
              step={0.05}
              min={-2}
              max={2}
              allowEmpty
            />
            <NumberField
              label="Frequency penalty"
              value={value.frequency_penalty}
              onChange={(v) => set("frequency_penalty", v)}
              step={0.05}
              min={-2}
              max={2}
              allowEmpty
            />
            <NumberField
              label="XTC probability"
              hint="exclude top choices"
              value={value.xtc_probability}
              onChange={(v) => set("xtc_probability", v ?? 0)}
              step={0.05}
              min={0}
              max={1}
            />
            <NumberField
              label="XTC threshold"
              value={value.xtc_threshold}
              onChange={(v) => set("xtc_threshold", v ?? 0)}
              step={0.01}
              min={0}
              max={0.5}
            />
            <NumberField label="Seed" hint="blank = random" value={value.seed} onChange={(v) => set("seed", v)} allowEmpty />
          </div>
          <Toggle
            label="Thinking mode"
            hint="For chat templates that support it (e.g. Qwen3). Off by default."
            checked={!!value.enable_thinking}
            onChange={(v) => set("enable_thinking", v || null)}
          />
        </div>
      </Collapsible>
    </div>
  );
}
