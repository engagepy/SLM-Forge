// Day / night switch. Until the user picks one, the theme follows macOS (index.css); a pick is
// saved in localStorage and applied before first paint by the script in index.html.
import { useEffect, useState } from "react";

import { cx } from "../ui";

type Theme = "light" | "dark";
const KEY = "slm.theme";
const media = () => window.matchMedia("(prefers-color-scheme: light)");

function saved(): Theme | null {
  try {
    const t = localStorage.getItem(KEY);
    return t === "light" || t === "dark" ? t : null;
  } catch {
    return null;
  }
}

function useTheme(): [Theme, () => void] {
  const [choice, setChoice] = useState<Theme | null>(saved);
  const [system, setSystem] = useState<Theme>(() => (media().matches ? "light" : "dark"));

  // Follow the OS while nothing is chosen.
  useEffect(() => {
    const m = media();
    const on = () => setSystem(m.matches ? "light" : "dark");
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);

  // Keep every mounted toggle (rail, sidebar, Advanced) and other tabs in step.
  useEffect(() => {
    const on = () => setChoice(saved());
    window.addEventListener("slm-theme", on);
    window.addEventListener("storage", on);
    return () => {
      window.removeEventListener("slm-theme", on);
      window.removeEventListener("storage", on);
    };
  }, []);

  useEffect(() => {
    if (choice) document.documentElement.dataset.theme = choice;
    else delete document.documentElement.dataset.theme;
  }, [choice]);

  const theme = choice ?? system;
  const toggle = () => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    // Picking the OS's own theme goes back to following the OS.
    const store = next === system ? null : next;
    try {
      if (store) localStorage.setItem(KEY, store);
      else localStorage.removeItem(KEY);
    } catch {
      /* private mode: applies to this page only */
    }
    setChoice(store);
    window.dispatchEvent(new Event("slm-theme"));
  };
  return [theme, toggle];
}

export default function ThemeToggle({ className }: { className?: string }) {
  const [theme, toggle] = useTheme();
  const label = theme === "dark" ? "Switch to day mode" : "Switch to dark mode";
  return (
    <button
      onClick={toggle}
      title={label}
      aria-label={label}
      className={cx("grid size-7 place-items-center rounded-md text-faint hover:bg-panel-2 hover:text-fg", className)}
    >
      {theme === "dark" ? (
        <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
          <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
        </svg>
      )}
    </button>
  );
}
