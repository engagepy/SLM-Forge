import { useState } from "react";

import { Button, TextArea } from "../ui";

/** The message box under a chat: Enter sends, Shift+Enter adds a line. With `onStop` there is a
 * Stop button while it streams; with `onClear` a clear button. If `onSend` returns a promise that
 * rejects, the text comes back so it isn't lost. */
export default function ChatComposer({
  busy,
  sending = false,
  onSend,
  onStop,
  onClear,
  clearLabel = "Clear",
  canClear = true,
  disabled = false,
  placeholder,
  autoFocus,
}: {
  busy: boolean;
  sending?: boolean;
  onSend: (text: string) => void | Promise<unknown>;
  onStop?: () => void;
  onClear?: () => void;
  clearLabel?: string;
  canClear?: boolean;
  disabled?: boolean;
  placeholder: string;
  autoFocus?: boolean;
}) {
  const [input, setInput] = useState("");
  const send = () => {
    const text = input.trim();
    if (!text || sending || disabled || (busy && onStop)) return;
    setInput("");
    Promise.resolve(onSend(text)).catch(() => setInput(text));
  };
  return (
    <form
      className="flex gap-2 border-t border-line p-4"
      onSubmit={(e) => {
        e.preventDefault();
        send();
      }}
    >
      <TextArea
        rows={2}
        className="flex-1"
        value={input}
        autoFocus={autoFocus}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            send();
          }
        }}
        placeholder={placeholder}
      />
      <div className="flex flex-col gap-1.5">
        {busy && onStop ? (
          <Button type="button" variant="danger" onClick={onStop}>
            Stop
          </Button>
        ) : (
          <Button type="submit" variant="primary" disabled={!input.trim() || disabled} loading={sending}>
            Send
          </Button>
        )}
        {onClear && (
          <Button type="button" variant="ghost" size="sm" onClick={onClear} disabled={busy || !canClear}>
            {clearLabel}
          </Button>
        )}
      </div>
    </form>
  );
}
