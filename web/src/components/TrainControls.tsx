import type { TrainConfig } from "../api";
import { Collapsible, Field, NumberField, Select, Toggle } from "../ui";

/** The full hyperparameter surface. Essentials first; the rest behind disclosures. */
export default function TrainControls({ value, onChange }: { value: TrainConfig; onChange: (v: TrainConfig) => void }) {
  const set = <K extends keyof TrainConfig>(k: K, v: TrainConfig[K]) => onChange({ ...value, [k]: v });
  const dpo = value.mode === "dpo";
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <NumberField
          label="Learning rate"
          value={value.learning_rate}
          onChange={(v) => v && set("learning_rate", v)}
          step={dpo ? 1e-7 : 1e-5}
          min={0}
        />
        <Field label="Length">
          <div className="flex gap-1">
            <Select
              className="w-24"
              value={value.iters != null ? "iters" : "epochs"}
              onChange={(k) => onChange(k === "iters" ? { ...value, iters: 200 } : { ...value, iters: null, epochs: value.epochs ?? 2 })}
              options={["epochs", "iters"] as const}
            />
            <input
              type="number"
              className="num w-full rounded-lg border border-line bg-bg px-2 py-1.5 text-[13px] focus:border-accent focus:outline-none"
              value={value.iters ?? value.epochs ?? ""}
              step={value.iters != null ? 10 : 0.5}
              min={value.iters != null ? 1 : 0.1}
              onChange={(e) =>
                e.target.value && (value.iters != null ? set("iters", Math.round(Number(e.target.value))) : set("epochs", Number(e.target.value)))
              }
            />
          </div>
        </Field>
        <NumberField label="Batch size" value={value.batch_size} onChange={(v) => v && set("batch_size", v)} min={1} />
        <NumberField
          label="Max sequence length"
          hint="tokens"
          value={value.max_seq_length}
          onChange={(v) => v && set("max_seq_length", v)}
          step={64}
          min={64}
        />
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Field label="Method">
          <Select
            value={value.fine_tune_type}
            onChange={(v) => set("fine_tune_type", v)}
            options={[
              { value: "lora", label: "LoRA" },
              { value: "dora", label: "DoRA" },
              { value: "full", label: "Full fine-tune" },
            ]}
          />
        </Field>
        <NumberField label="LoRA rank" value={value.lora_rank} onChange={(v) => v && set("lora_rank", v)} min={1} max={256} />
        <NumberField label="LoRA scale" hint="≈ alpha / rank" value={value.lora_scale} onChange={(v) => v && set("lora_scale", v)} step={1} />
        <NumberField label="Layers tuned" hint="-1 = all" value={value.num_layers} onChange={(v) => v != null && set("num_layers", v)} min={-1} />
      </div>

      {dpo && (
        <div className="grid grid-cols-2 gap-3 rounded-lg border border-accent/30 bg-accent-soft/40 p-3 md:grid-cols-3">
          <NumberField
            label="β (beta)"
            hint="how far from the reference"
            value={value.beta}
            onChange={(v) => v && set("beta", v)}
            step={0.01}
            min={0.01}
          />
          <Field label="DPO loss">
            <Select value={value.dpo_loss_type} onChange={(v) => set("dpo_loss_type", v)} options={["sigmoid", "hinge", "ipo", "dpop"] as const} />
          </Field>
          {value.dpo_loss_type === "dpop" && (
            <NumberField label="DPOP δ" value={value.dpop_delta} onChange={(v) => v != null && set("dpop_delta", v)} />
          )}
        </div>
      )}

      <div className="flex flex-wrap gap-x-8 gap-y-3">
        <Toggle
          label="Gradient checkpointing"
          hint="Much less memory, ~30% slower"
          checked={value.grad_checkpoint}
          onChange={(v) => set("grad_checkpoint", v)}
        />
        {!dpo && (
          <Toggle
            label="Train on answers only"
            hint="Mask prompt tokens out of the loss"
            checked={value.mask_prompt}
            onChange={(v) => set("mask_prompt", v)}
          />
        )}
      </div>

      <Collapsible title="Optimizer, schedule, regularisation and logging">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Field label="Optimizer">
            <Select value={value.optimizer} onChange={(v) => set("optimizer", v)} options={["adamw", "adam"] as const} />
          </Field>
          <NumberField label="Weight decay" value={value.weight_decay} onChange={(v) => v != null && set("weight_decay", v)} step={0.01} min={0} />
          <Field label="LR schedule">
            <Select value={value.lr_schedule} onChange={(v) => set("lr_schedule", v)} options={["cosine", "linear", "constant"] as const} />
          </Field>
          <NumberField
            label="Warmup steps"
            hint="capped at 25% of run"
            value={value.warmup_steps}
            onChange={(v) => v != null && set("warmup_steps", v)}
            min={0}
          />
          <NumberField
            label="Final LR ratio"
            hint="of peak"
            value={value.min_lr_ratio}
            onChange={(v) => v != null && set("min_lr_ratio", v)}
            step={0.05}
            min={0}
            max={1}
          />
          <NumberField
            label="Grad accumulation"
            value={value.grad_accumulation_steps}
            onChange={(v) => v && set("grad_accumulation_steps", v)}
            min={1}
          />
          <NumberField label="LoRA dropout" value={value.lora_dropout} onChange={(v) => v != null && set("lora_dropout", v)} step={0.05} min={0} max={0.9} />
          <NumberField label="Seed" value={value.seed} onChange={(v) => v != null && set("seed", v)} />
          <NumberField label="Report every" hint="iters" value={value.steps_per_report} onChange={(v) => v && set("steps_per_report", v)} min={1} />
          <NumberField label="Validate every" hint="iters" value={value.steps_per_eval} onChange={(v) => v && set("steps_per_eval", v)} min={1} />
          <NumberField label="Checkpoint every" hint="iters" value={value.save_every} onChange={(v) => v && set("save_every", v)} min={1} />
          <NumberField label="Val batches" hint="-1 = all" value={value.val_batches} onChange={(v) => v != null && set("val_batches", v)} min={-1} />
        </div>
        <div className="mt-3">
          <Field label="LoRA target modules" hint="comma-separated; blank = all linear layers">
            <input
              className="w-full rounded-lg border border-line bg-bg px-2.5 py-1.5 font-mono text-xs focus:border-accent focus:outline-none"
              placeholder="self_attn.q_proj, self_attn.v_proj"
              value={value.lora_keys?.join(", ") ?? ""}
              onChange={(e) => {
                const keys = e.target.value.split(",").map((s) => s.trim()).filter(Boolean);
                set("lora_keys", keys.length ? keys : null);
              }}
            />
          </Field>
        </div>
      </Collapsible>
    </div>
  );
}
