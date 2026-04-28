import { FC, MutableRefObject, useCallback, useEffect, useMemo, useState } from "react";
import { colors } from "../../../theme/colors";

const QUESTION_BLOCKS: {
  id: "q1" | "q2" | "q3" | "q4";
  prompt: string;
  options: { value: string; label: string }[];
}[] = [
  {
    id: "q1",
    prompt: "Have you tried the original Moshi demo before?",
    options: [
      { value: "yes", label: "Yes" },
      { value: "no", label: "No" },
    ],
  },
  {
    id: "q2",
    prompt:
      "Overall experience\nBased on your overall experience, which version did you prefer?",
    options: [
      { value: "moshi", label: "Moshi" },
      { value: "moshirag", label: "MoshiRAG" },
      { value: "no_preference", label: "No preference" },
      { value: "not_sure", label: "Not sure" },
    ],
  },
  {
    id: "q3",
    prompt:
      "Usefulness of information\nWhich one gave you more useful information?",
    options: [
      { value: "moshi", label: "Moshi" },
      { value: "moshirag", label: "MoshiRAG" },
      { value: "no_preference", label: "No preference" },
      { value: "not_sure", label: "Not sure" },
    ],
  },
  {
    id: "q4",
    prompt:
      "Naturalness of conversation\nWhich version felt more natural to interact with?",
    options: [
      { value: "moshi", label: "Moshi" },
      { value: "moshirag", label: "MoshiRAG" },
      { value: "no_preference", label: "No preference" },
      { value: "not_sure", label: "Not sure" },
    ],
  },
];

const OPEN_PROMPT =
  "Any other feedback?";

function sessionFeedbackUrl(workerAddr: string): string {
  let addr = workerAddr;
  if (addr === "same" || addr === "") {
    addr = `${window.location.hostname}:${window.location.port}`;
  }
  const protocol = window.location.protocol === "https:" ? "https" : "http";
  return `${protocol}://${addr}/api/session_feedback`;
}

export type SessionFeedbackModalProps = {
  open: boolean;
  workerAddr: string;
  /** If set, call `onDismiss` with no POST when the user does not act in time (welcome page). */
  autoDismissMs?: number;
  /** Wall-clock duration in ms; read `.current` at submit (set on WS disconnect). */
  conversationDurationMs: MutableRefObject<number | null>;
  /** Close the modal only (return to the conversation view) without sending feedback. */
  onClose: () => void;
  onDismiss: () => void;
};

export const SessionFeedbackModal: FC<SessionFeedbackModalProps> = ({
  open,
  workerAddr,
  autoDismissMs,
  conversationDurationMs,
  onClose,
  onDismiss,
}) => {
  const [q1, setQ1] = useState("");
  const [q2, setQ2] = useState("");
  const [q3, setQ3] = useState("");
  const [q4, setQ4] = useState("");
  const [openText, setOpenText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || autoDismissMs == null || autoDismissMs <= 0) {
      return;
    }
    const id = window.setTimeout(() => {
      onDismiss();
    }, autoDismissMs);
    return () => window.clearTimeout(id);
  }, [open, autoDismissMs, onDismiss]);

  const url = useMemo(() => sessionFeedbackUrl(workerAddr), [workerAddr]);

  const postPayload = useCallback(
    async (payload: Record<string, unknown>) => {
      setBusy(true);
      setError(null);
      try {
        const res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...payload,
            submittedAt: new Date().toISOString(),
          }),
        });
        if (!res.ok && res.status !== 202) {
          const t = await res.text().catch(() => "");
          throw new Error(t || `HTTP ${res.status}`);
        }
        onDismiss();
      } catch (e) {
        console.error("session_feedback", e);
        setError("Could not send feedback. Try again or close without sending.");
      } finally {
        setBusy(false);
      }
    },
    [url, onDismiss],
  );

  const onSubmit = useCallback(() => {
    void postPayload({
      skipped: false,
      // String plays nicely with Google Sheets / Apps Script appendRow.
      durationMs:
        conversationDurationMs.current == null
          ? null
          : String(conversationDurationMs.current),
      q1: q1 || null,
      q2: q2 || null,
      q3: q3 || null,
      q4: q4 || null,
      openFeedback: openText.trim() || null,
    });
  }, [postPayload, q1, q2, q3, q4, openText, conversationDurationMs]);

  if (!open) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-[200] flex items-center justify-center p-4 backdrop-blur-sm"
      style={{ backgroundColor: colors.overlayMask }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="session-feedback-title"
    >
      <div className="w-[90vw] min-w-[50vw] max-w-[800px] max-h-[90vh] overflow-y-auto rounded-2xl border shadow-xl" style={{ borderColor: colors.border, backgroundColor: colors.bgCanvas, color: colors.textPrimary }}>
        <div className="p-6 space-y-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 id="session-feedback-title" className="text-lg font-semibold" style={{ color: colors.accentGreenDark }}>
                Quick feedback
              </h2>
              <p className="text-sm mt-1" style={{ color: colors.textLight }}>
                Feel free to skip, but your answers help us improve the demo.
              </p>
            </div>
            <button
              type="button"
              disabled={busy}
              aria-label="Close feedback form"
              onClick={() => onClose()}
              className="shrink-0 rounded-full p-2 transition-colors disabled:opacity-50 hover:opacity-80"
              style={{ color: colors.textLight }}
            >
              <span aria-hidden="true" className="text-lg leading-none">
                ×
              </span>
            </button>
          </div>

          {QUESTION_BLOCKS.map((block) => {
            const value =
              block.id === "q1"
                ? q1
                : block.id === "q2"
                  ? q2
                  : block.id === "q3"
                    ? q3
                    : q4;
            const setValue =
              block.id === "q1"
                ? setQ1
                : block.id === "q2"
                  ? setQ2
                  : block.id === "q3"
                    ? setQ3
                    : setQ4;
            return (
              <fieldset key={block.id} className="space-y-2 border-0 p-0 m-0">
                <legend className="text-sm font-medium mb-2" style={{ color: colors.textPrimary }}>
                  {block.prompt.split("\n").map((line, idx, arr) => (
                    <span key={`${block.id}-l${idx}`}>
                      {idx === 0 && arr.length > 1 ? (
                        <strong className="font-black text-[0.875em] leading-tight" style={{ color: colors.accentBlue }}>
                          {block.id === "q1" ? (
                            (() => {
                              const token = "Moshi demo";
                              if (!line.includes(token)) return line;
                              const [before, after] = line.split(token);
                              return (
                                <>
                                  {before}
                                  <a
                                    href="https://moshi.chat/"
                                    target="_blank"
                                    rel="noreferrer"
                                    className="hover:underline"
                                    style={{ color: colors.accentGreenDark }}
                                  >
                                    {token}
                                  </a>
                                  {after}
                                </>
                              );
                            })()
                          ) : (
                            line
                          )}
                        </strong>
                      ) : (
                        block.id === "q1" ? (
                          (() => {
                            const token = "Moshi demo";
                            if (!line.includes(token)) return line;
                            const [before, after] = line.split(token);
                            return (
                              <>
                                {before}
                                <a
                                  href="https://moshi.chat/"
                                  target="_blank"
                                  rel="noreferrer"
                                  className="hover:underline"
                                  style={{ color: colors.accentGreenDark }}
                                >
                                  {token}
                                </a>
                                {after}
                              </>
                            );
                          })()
                        ) : (
                          line
                        )
                      )}
                      {idx < arr.length - 1 && <br />}
                    </span>
                  ))}
                </legend>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
                  {block.options.map((opt) => (
                    <label
                      key={opt.value}
                      className="flex items-center justify-start gap-2 text-sm cursor-pointer rounded-lg px-2 py-1.5 w-full"
                      style={{ backgroundColor: colors.black80 }}
                    >
                      <input
                        type="radio"
                        className=""
                        style={{ accentColor: colors.accentGreenDark }}
                        name={block.id}
                        value={opt.value}
                        checked={value === opt.value}
                        onChange={() => setValue(opt.value)}
                      />
                      <span>{opt.label}</span>
                    </label>
                  ))}
                </div>
              </fieldset>
            );
          })}

          <div>
            <label htmlFor="session-feedback-open" className="text-sm font-medium block mb-2">
              {OPEN_PROMPT}
            </label>
            <textarea
              id="session-feedback-open"
              rows={4}
              className="w-full rounded-xl border px-3 py-2 text-sm focus:outline-none focus:ring-2"
              style={{ borderColor: colors.border, backgroundColor: colors.black90, color: colors.textPrimary }}
              placeholder="Type here…"
              value={openText}
              onChange={(e) => setOpenText(e.target.value)}
            />
          </div>

          {error && <p className="text-xs text-amber-700">{error}</p>}

          <div className="flex flex-wrap gap-3 justify-end pt-2">
            <button
              type="button"
              disabled={busy}
              onClick={onSubmit}
              className="px-5 py-2 rounded-full text-sm font-medium text-white shadow-sm disabled:opacity-50 transition-transform enabled:hover:brightness-110 enabled:hover:scale-[1.02] enabled:active:scale-[0.99]"
              style={{ backgroundColor: colors.accentGreenDark }}
            >
              {busy ? "Sending…" : "Submit"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
