import { useState } from "react";

import { Button, TextArea } from "../ui";

/** The message box under a chat: Enter sends, Shift+Enter adds a line, Stop while it streams. */
export default function ChatComposer({
  busy,
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
  onSend: (text: string) => void;
  onStop: () => void;
  onClear: () => void;
  clearLabel?: string;
  canClear?: boolean;
  disabled?: boolean;
  placeholder: string;
  autoFocus?: boolean;
}) {
  const [input, setInput] = useState("");
  const send = () => {
    if (!input.trim() || busy || disabled) return;
    onSend(input);
    setInput("");
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
        {busy ? (
          <Button type="button" variant="danger" onClick={onStop}>
            Stop
          </Button>
        ) : (
          <Button type="submit" variant="primary" disabled={!input.trim() || disabled}>
            Send
          </Button>
        )}
        <Button type="button" variant="ghost" size="sm" onClick={onClear} disabled={busy || !canClear}>
          {clearLabel}
        </Button>
      </div>
    </form>
  );
}
