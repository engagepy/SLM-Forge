import { useEffect, useRef } from "react";

/** Terminal-style log that follows the tail unless the user has scrolled up. */
export default function JobLog({ lines, height = 260 }: { lines: string[]; height?: number }) {
  const ref = useRef<HTMLPreElement>(null);
  const stick = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [lines]);
  return (
    <pre
      ref={ref}
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
      }}
      style={{ height }}
      className="overflow-auto rounded-lg border border-line bg-bg p-3 font-mono text-[11.5px] leading-relaxed text-muted"
    >
      {lines.length ? lines.join("\n") : "No output yet."}
    </pre>
  );
}
