import { forwardRef, useState } from 'react';
import { SearchResult } from '../../hooks/useServerText';
import { colors } from '../../../../theme/colors';

type RetrievalBackendOption = { id: string };

/** Clickable tabs for retrieval LLM (2+ backends). */
function RetrievalBackendTabs({
  backends,
  selectedId,
  onChange,
}: {
  backends: RetrievalBackendOption[];
  selectedId: string;
  onChange: (id: string) => void;
}) {
  return (
    <div
      role="tablist"
      aria-label="Retrieval LLM"
      className="flex flex-wrap items-end gap-x-0 min-w-0 shrink overflow-x-auto border-b"
      style={{ borderColor: colors.border }}
    >
      {backends.map((b) => {
        const active = b.id === selectedId;
        const tabText = b.id;
        return (
          <button
            key={b.id}
            type="button"
            role="tab"
            aria-selected={active}
            title={b.id}
            className="-mb-px border-b-2 px-2.5 py-1.5 text-[12px] font-mono lowercase tracking-wide whitespace-nowrap transition-colors 
            duration-150 shrink-0 focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-0 rounded-t-sm hover:opacity-100"
            style={{
              color: active ? colors.textPrimary : colors.textLight,
              opacity: active ? 1 : 0.72,
              borderBottomColor: active ? colors.accentBlue : 'transparent',
            }}
            onClick={() => onChange(b.id)}
          >
            {tabText}
          </button>
        );
      })}
    </div>
  );
}

interface SearchPanelProps {
  result: SearchResult | null;
  isRetrieving: boolean;
  hasActiveText: boolean; // Controls the Blue Text (10s duration)
  isFreshResult?: boolean; // True when fresh result is available
  isRecovering?: boolean; // True during gradual recovery phase
  retrievalBackends?: { id: string }[];
  selectedRetrievalId?: string | null;
  onRetrievalChange?: (id: string) => void;
}

const SearchPanel = forwardRef<HTMLDivElement, SearchPanelProps>(({
  result,
  isRetrieving,
  hasActiveText,
  isFreshResult = false,
  isRecovering = false,
  retrievalBackends,
  selectedRetrievalId,
  onRetrievalChange,
}, ref) => {
  const [showRetrievalNotice, setShowRetrievalNotice] = useState(false);
  const showRetrievalDisclaimer =
    Boolean(retrievalBackends && retrievalBackends.length > 1);

  // Determine styles based on state
  // Retrieval (Thinking) = Green Glow (Shadow)
  // Fresh Result (Done) = Blue Glow (Shadow)

  let shadowClass = 'shadow-sm';
  let shadowStyle: { boxShadow: string } | undefined;
  let titleColor = colors.textLight;

  // Shadow Logic (Glow)
  // Priority: isRetrieving (green) > isFreshResult (blue) > isRecovering (fade to no color)
  if (isRetrieving && !isFreshResult) {
    // Green glow when retrieving and no fresh result yet
    // Increased blur radius for a larger aura
    shadowStyle = { boxShadow: colors.searchGlowGreen };
  } else if (isFreshResult) {
    // Blue glow when fresh result is available
    // Increased blur radius for a larger aura
    shadowStyle = { boxShadow: colors.searchGlowBlue };
  } else if (isRecovering && !isFreshResult) {
    // Gradual fade to no glow (transparent) during recovery - transition handled by CSS
    shadowClass = 'shadow-sm';
  }

  // Title Logic
  if (isRetrieving && !isFreshResult) {
    titleColor = colors.accentGreen;
  } else if (isFreshResult) {
    titleColor = colors.accentBlue;
  } else if (isRecovering && !isFreshResult) {
    // Gradual transition from blue to original color during recovery
    titleColor = colors.textLight;
  }

  // Border Logic
  let borderClass = colors.border;
  if (isRetrieving && !isFreshResult) {
    borderClass = colors.accentGreenTransparent;
  } else if (isFreshResult) {
    borderClass = colors.accentBlueTransparent;
  } else if (isRecovering && !isFreshResult) {
    // Gradually fade border during recovery
    borderClass = colors.border;
  }

  // Transition: 7s glow fade during recovery (blue to none); text fades out over 5s
  const transitionClass = isRecovering
    ? 'transition-all duration-[9000ms] ease-in-out'
    : 'transition-all duration-200 ease-in-out';

  return (
    <div ref={ref} className={`relative flex flex-col h-full backdrop-blur-md rounded-2xl border ${transitionClass} overflow-hidden ${shadowClass}`} style={{ backgroundColor: colors.black90, borderColor: borderClass, ...shadowStyle }}>
      <div className="p-4 border-b flex justify-between items-center gap-2 transition-colors duration-200" style={{ borderColor: colors.border }}>
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <h2 className="text-xs font-mono uppercase tracking-widest transition-colors duration-200 shrink-0" style={{ color: titleColor }}>Retrieval Back End</h2>
          {retrievalBackends && retrievalBackends.length > 1 && onRetrievalChange && selectedRetrievalId != null ? (
            <RetrievalBackendTabs
              backends={retrievalBackends}
              selectedId={selectedRetrievalId}
              onChange={onRetrievalChange}
            />
          ) : null}
        </div>
        <div className="flex items-center gap-3 shrink-0">
          {isRetrieving && (
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style={{ backgroundColor: colors.accentGreen }}></span>
              <span className="relative inline-flex rounded-full h-2 w-2" style={{ backgroundColor: colors.accentGreen }}></span>
            </span>
          )}
          {showRetrievalDisclaimer ? (
            <button
              type="button"
              title="Retrieval selection disclaimer"
              aria-label="Retrieval selection disclaimer"
              onClick={() => setShowRetrievalNotice((prev) => !prev)}
              className="text-[10px] font-mono uppercase tracking-wide transition-colors hover:opacity-80"
              style={{ color: colors.textLight }}
            >
              Notice
            </button>
          ) : null}
        </div>
      </div>

      {showRetrievalNotice && showRetrievalDisclaimer && (
        <div
          className="absolute top-14 right-4 z-30 w-80 max-w-[calc(100%-2rem)] backdrop-blur-md border rounded-lg p-3 shadow-lg"
          style={{ backgroundColor: colors.black95, borderColor: colors.border }}
        >
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold uppercase tracking-wide" style={{ color: colors.textPrimary }}>Notice</h3>
            <button
              type="button"
              onClick={() => setShowRetrievalNotice(false)}
              className="hover:opacity-80"
              style={{ color: colors.textLight }}
              aria-label="Close notice"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div className="mt-2 border-t pt-3" style={{ borderColor: colors.border }}>
            <p className="text-xs leading-relaxed" style={{ color: colors.textTranscript }}>
              Your selected retrieval back end may not always be used. If an online API is under heavy traffic
              and responds too slowly, it can time out before returning a result. When that happens, the system
              falls back to another configured retrieval method so the session can continue smoothly.
            </p>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-6 scroll-smooth">
        {isRetrieving && !result ? (
          <div className="flex flex-col items-center justify-center h-full space-y-4 animate-pulse" style={{ color: colors.accentGreen }}>
            <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <p className="text-xs font-mono">Retrieving...</p>
          </div>
        ) : !result ? (
          <div className="flex flex-col items-center justify-center h-full space-y-4" style={{ color: colors.textDisabled }}>
            <svg className="w-10 h-10 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
            </svg>
            <p className="text-xs font-mono opacity-60">System Idle</p>
          </div>
        ) : (
          <div className="space-y-6">
            <div key={result.id} className="relative">
              {/* Meta: section title + LM id from payload */}
              <div className="mb-2 flex items-baseline justify-between gap-2 min-w-0 w-full">
                <span className="text-xs font-mono uppercase tracking-wider shrink-0" style={{ color: hasActiveText ? colors.accentBlue : colors.accentGreen }}>
                  Reference text
                </span>
                {result.lmLabel ? (
                  <span
                    className="text-[13px] font-mono leading-tight text-right truncate min-w-0 flex-1 pl-2 opacity-90"
                    style={{ color: colors.textLight }}
                    title={result.lmLabel}
                  >
                    {result.lmLabel}
                  </span>
                ) : null}
              </div>

              {/* Result */}
              <p className="text-xs leading-relaxed opacity-90 font-mono" style={{ color: colors.textTranscript }}>
                {result.result}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

SearchPanel.displayName = 'SearchPanel';

export default SearchPanel;
