import { useCallback, useLayoutEffect, useState } from 'react';
import { colors } from '../../../../theme/colors';

interface PipeAnimationProps {
  visualizerWrapperRef: React.RefObject<HTMLDivElement>;
  searchPanelRef: React.RefObject<HTMLDivElement>;
  isRetrieving: boolean;
  isFreshResult: boolean;
  isRecovering: boolean;
}

const PipeAnimation: React.FC<PipeAnimationProps> = ({
  visualizerWrapperRef,
  searchPanelRef,
  isRetrieving,
  isFreshResult,
  isRecovering,
}) => {
  const [dynamicPaths, setDynamicPaths] = useState({ request: '', response: '' });

  // --- Dynamic Path Calculation ---
  const updatePaths = useCallback(() => {
    if (!searchPanelRef.current || !visualizerWrapperRef.current) return;

    // 1. Get container context (Main Element)
    const mainEl = document.querySelector('main');
    if (!mainEl) return;
    const mainRect = mainEl.getBoundingClientRect();

    // 2. Get Element Rects
    const vizRect = visualizerWrapperRef.current.getBoundingClientRect();
    const panelRect = searchPanelRef.current.getBoundingClientRect();

    // 3. Calculate Coordinates Relative to Main/SVG
    // Visualizer Center
    const vizCenterX = vizRect.left - mainRect.left + vizRect.width / 2;
    // Use center (height / 2) because we resized the box to fit the orb tightly
    const vizCenterY = vizRect.top - mainRect.top + vizRect.height / 2;

    // Orb Boundary (Radius ~30px for the smaller pearl size)
    const orbRadius = 30;
    const orbBottomY = vizCenterY + orbRadius;

    // Panel Top Center
    const panelTopY = panelRect.top - mainRect.top;

    // 4. Generate Straight Paths
    // We separate the pipes slightly by X-axis to avoid overlap
    const spacing = 12;

    // Request Pipe (Left): Orb Bottom -> Panel Top
    const reqX = vizCenterX - spacing;
    const requestPath = `M ${reqX} ${orbBottomY} L ${reqX} ${panelTopY}`;

    // Response Pipe (Right): Panel Top -> Orb Bottom
    const resX = vizCenterX + spacing;
    const responsePath = `M ${resX} ${panelTopY} L ${resX} ${orbBottomY}`;

    setDynamicPaths({ request: requestPath, response: responsePath });
  }, [searchPanelRef, visualizerWrapperRef]);

  // Setup Observers for Resizing
  useLayoutEffect(() => {
    updatePaths();
    window.addEventListener('resize', updatePaths);

    const resizeObserver = new ResizeObserver(() => {
      updatePaths();
    });

    if (searchPanelRef.current) {
      resizeObserver.observe(searchPanelRef.current);
    }
    if (visualizerWrapperRef.current) {
      resizeObserver.observe(visualizerWrapperRef.current);
    }

    return () => {
      window.removeEventListener('resize', updatePaths);
      resizeObserver.disconnect();
    };
  }, [updatePaths, searchPanelRef, visualizerWrapperRef, isRetrieving, isFreshResult]);

  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none z-10 overflow-visible">
      <defs>
        <linearGradient id="pipeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor={colors.border} stopOpacity="0" />
          <stop offset="50%" stopColor={colors.accentGreen} />
          <stop offset="100%" stopColor={colors.border} stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* Static Rails (Ghost Paths) - hidden when corresponding active pipe is visible */}
      {!(isRetrieving && !isFreshResult) && (
        <path
          d={dynamicPaths.request}
          stroke={colors.rail}
          strokeWidth="2"
          fill="none"
          opacity="0.3"
          strokeDasharray="4 4"
          strokeLinecap="round"
        />
      )}
      {!(isFreshResult && !isRecovering) && (
        <path
          d={dynamicPaths.response}
          stroke={colors.rail}
          strokeWidth="2"
          fill="none"
          opacity="0.3"
          strokeDasharray="4 4"
          strokeLinecap="round"
        />
      )}

      {/* Request Pipe (Active) - Orb → Panel: flows INTO retrieval box */}
      {isRetrieving && !isFreshResult && (
        <path
          d={dynamicPaths.request}
          stroke={colors.accentGreenTransparent}
          strokeWidth="2"
          strokeDasharray="4 4"
          fill="none"
          className="pipe-flow-in"
        />
      )}

      {/* Response Pipe (Active) - Panel → Orb: flows OUT OF retrieval box */}
      <path
        d={dynamicPaths.response}
        stroke={colors.accentBlueTransparent}
        strokeWidth="2"
        strokeDasharray="4 4"
        fill="none"
        className={`pipe-flow-out transition-all duration-[9000ms] ease-out ${isRecovering ? 'opacity-0' : isFreshResult ? 'opacity-100' : 'opacity-0'}`}
      />
    </svg>
  );
};

export default PipeAnimation;
