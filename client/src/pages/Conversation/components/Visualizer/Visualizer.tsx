import { useEffect, useRef } from 'react';
import { colors } from '../../../../theme/colors';

interface VisualizerProps {
  userAnalyser: AnalyserNode | null;
  serverAnalyser: AnalyserNode | null;
  isActive: boolean;
  isRetrieving: boolean; // True when retrieving (Green mode)
  isFreshResult: boolean; // True when fresh result is available (Blue mode)
  isRecovering?: boolean; // True during gradual recovery phase (5s after 5s blue glow)
  captureCanvasRef?: React.MutableRefObject<HTMLCanvasElement | null>;
}

const Visualizer: React.FC<VisualizerProps> = ({
  userAnalyser,
  serverAnalyser,
  isActive,
  isRetrieving,
  isFreshResult,
  isRecovering = false,
  captureCanvasRef,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);

  // Refs to hold latest values for the animation loop
  const isActiveRef = useRef(isActive);
  const isRetrievingRef = useRef(isRetrieving);
  const isFreshResultRef = useRef(isFreshResult);
  const isRecoveringRef = useRef(isRecovering);

  // Sync refs with props
  useEffect(() => {
    isActiveRef.current = isActive;
    isRetrievingRef.current = isRetrieving;
    isFreshResultRef.current = isFreshResult;
    isRecoveringRef.current = isRecovering;
  }, [isActive, isRetrieving, isFreshResult, isRecovering]);

  useEffect(() => {
    if (!captureCanvasRef) {
      return;
    }
    captureCanvasRef.current = canvasRef.current;
    return () => {
      captureCanvasRef.current = null;
    };
  }, [captureCanvasRef]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // High DPI Canvas Setup - Use container's actual size (responsive)
    const container = canvas.parentElement;
    const getSize = () => {
      if (container) {
        const rect = container.getBoundingClientRect();
        return Math.min(rect.width, rect.height, 360); // Max 360px, but responsive below that
      }
      return 360; // Fallback to 360px
    };

    const updateCanvasSize = () => {
      const size = getSize();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = size * dpr;
      canvas.height = size * dpr;
      canvas.style.width = `${size}px`;
      canvas.style.height = `${size}px`;
      ctx.scale(dpr, dpr);
    };

    updateCanvasSize();

    // Update on resize
    const resizeObserver = new ResizeObserver(() => {
      updateCanvasSize();
    });
    if (container) {
      resizeObserver.observe(container);
    }

    // Audio Data Buffers
    const userDataArray = new Uint8Array(userAnalyser ? userAnalyser.frequencyBinCount : 0);
    const serverDataArray = new Uint8Array(serverAnalyser ? serverAnalyser.frequencyBinCount : 0);

    // Animation Smoothing Variables
    let smoothedUserVol = 0;
    let smoothedServerVol = 0;

    // Color Interpolation Variables
    // Default Green -> 57, 242, 174
    // Logic Blue -> 61, 163, 230
    let currentR = 57;
    let currentG = 242;
    let currentB = 174;

    const draw = () => {
      // Get current canvas dimensions (accounting for DPR scaling)
      const currentWidth = canvas.width / (window.devicePixelRatio || 1);
      const currentHeight = canvas.height / (window.devicePixelRatio || 1);

      // Scale radii proportionally to canvas size (base size is 360px)
      const scale = Math.min(currentWidth, currentHeight) / 360;

      // Clear
      ctx.clearRect(0, 0, currentWidth, currentHeight);
      const cx = currentWidth / 2;
      const cy = currentHeight / 2;

      // 1. Color Transitions
      // Green when retrieving, Blue only when fresh result (and not retrieving)
      let targetR = 57; // Default green
      let targetG = 242;
      let targetB = 174;

      if (isRetrievingRef.current) {
        // Green mode when retrieving
        targetR = 57;
        targetG = 242;
        targetB = 174;
      } else if (isFreshResultRef.current && !isRecoveringRef.current) {
        // Blue mode only when fresh result is available (and not retrieving)
        targetR = 61;
        targetG = 163;
        targetB = 230;
      } else if (isRecoveringRef.current) {
        // Gradual recovery: interpolate between blue and green over 5 seconds
        targetR = 57;
        targetG = 242;
        targetB = 174;
      }

      // Lerp color for smooth transition
      // Slower lerp (0.05) during recovery for 7-second transition, faster (0.02) otherwise
      const lerpRate = isRecoveringRef.current ? 0.006 : 0.02;
      currentR += (targetR - currentR) * lerpRate;
      currentG += (targetG - currentG) * lerpRate;
      currentB += (targetB - currentB) * lerpRate;

      // 2. Determine Target Volumes from real analysers
      let targetUserVol = 0;
      let targetServerVol = 0;

      if (userAnalyser) {
        userAnalyser.getByteFrequencyData(userDataArray);
        // Use max instead of average, with reduced amplification for less sensitivity
        const max = Math.max(...Array.from(userDataArray));
        // Reduced amplification to 2.0x for less sensitive user voice response
        if (isActiveRef.current) targetUserVol = max * 2.0;
      }

      if (serverAnalyser) {
        serverAnalyser.getByteFrequencyData(serverDataArray);
        // Use max instead of average for more sensitivity, with amplification
        const max = Math.max(...Array.from(serverDataArray));
        if (isActiveRef.current) targetServerVol = max * 1.5; // Amplify for more sensitivity
      }

      // If retrieving or fresh result, enforce a minimum "pulse" even if silent
      if (isRetrievingRef.current || isFreshResultRef.current) {
        targetServerVol = Math.max(targetServerVol, 40 + Math.sin(Date.now() / 200) * 10);
      }

      // 3. Smooth Values - More sensitive (faster response)
      // User voice: Higher smoothing factor (0.6) for faster, more responsive reaction
      smoothedUserVol += (targetUserVol - smoothedUserVol) * 0.6;
      // Server voice: Keep at 0.35 for smoother server audio visualization
      smoothedServerVol += (targetServerVol - smoothedServerVol) * 0.35;

      // 4. Common Idle Animation
      const time = Date.now() / 2000;
      const breathe = Math.sin(time) * 2.25; // Base breathe amount (will be scaled)

      // 5. Render Server (The Aura)
      const isAct = isActiveRef.current;

      if (isAct) {
        const sVolNorm = smoothedServerVol / 255;

        // SCALED: Max expansion proportional to canvas size
        // Base radius: 60px * scale
        // Max expansion: 105px * scale (at full volume)
        // Total max radius: 165px * scale
        const expansion = Math.pow(sVolNorm, 0.5) * 105 * scale; // Reduced power from 0.6 to 0.5 for more sensitivity 
        const sRadius = 60 * scale + expansion;

        // Even small volume triggers drawing when active
        const isGlowing = isRetrievingRef.current || isFreshResultRef.current;
        if (sVolNorm > 0.005 || isGlowing) { // Lowered threshold from 0.01 to 0.005 for more sensitivity
          const gradient = ctx.createRadialGradient(cx, cy, 50 * scale, cx, cy, sRadius); // Scaled inner radius proportionally

          const opacity = Math.min(0.8, 0.2 + sVolNorm * 1.2); // Increased multiplier for more visible response

          gradient.addColorStop(0, `rgba(${Math.round(currentR)}, ${Math.round(currentG)}, ${Math.round(currentB)}, ${opacity})`);
          gradient.addColorStop(0.4, `rgba(${Math.round(currentR)}, ${Math.round(currentG)}, ${Math.round(currentB)}, ${opacity * 0.5})`);
          gradient.addColorStop(1, `rgba(${Math.round(currentR)}, ${Math.round(currentG)}, ${Math.round(currentB)}, 0)`);

          ctx.beginPath();
          ctx.arc(cx, cy, sRadius, 0, Math.PI * 2);
          ctx.fillStyle = gradient;
          ctx.globalCompositeOperation = 'source-over';
          ctx.fill();

          // Secondary Inner Bloom
          if (sVolNorm > 0.15 || isGlowing) { // Lowered threshold from 0.2 to 0.15
            const innerR = (60 + (sVolNorm * 45)) * scale; // Scaled proportionally
            const innerGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, innerR);
            // Make center whitish-blue when fresh result, whitish-green when retrieving
            const innerMix = isFreshResultRef.current && !isRetrievingRef.current ? '200, 220, 255' : '167, 243, 208';
            innerGrad.addColorStop(0, `rgba(${innerMix}, ${sVolNorm * 0.7})`); // Increased from 0.6 to 0.7
            innerGrad.addColorStop(1, `rgba(${innerMix}, 0)`);

            ctx.beginPath();
            ctx.arc(cx, cy, innerR, 0, Math.PI * 2);
            ctx.fillStyle = innerGrad;
            ctx.fill();
          }
        }
      }

      // 6. Render User (Pearl)
      // SCALED: Base radius proportional to canvas size
      const uVolNorm = isAct ? (smoothedUserVol / 255) : 0;
      // Increased multiplier for much more visible size changes with user voice
      const uRadius = (45 + (uVolNorm * 30)) * scale + breathe * scale;

      const pearlGrad = ctx.createRadialGradient(cx - 10 * scale, cy - 10 * scale, 4 * scale, cx, cy, uRadius);
      pearlGrad.addColorStop(0, colors.white);
      pearlGrad.addColorStop(0.3, colors.pearlMid);
      pearlGrad.addColorStop(1, colors.pearlEdge);

      ctx.beginPath();
      ctx.arc(cx, cy, uRadius, 0, Math.PI * 2);
      ctx.fillStyle = pearlGrad;

      ctx.shadowColor = colors.shadowWarm;
      ctx.shadowBlur = 15;
      ctx.shadowOffsetY = 6;

      ctx.fill();

      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
      ctx.shadowOffsetY = 0;

      // 7. Idle State
      if (!isAct) {
        ctx.beginPath();
        ctx.arc(cx, cy, uRadius + 12 * scale, 0, Math.PI * 2); // Scaled proportionally
        ctx.strokeStyle = colors.rail;
        ctx.lineWidth = 1.5 * scale; // Scaled proportionally
        ctx.stroke();
      }

      animationRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      resizeObserver.disconnect();
    };
  }, [userAnalyser, serverAnalyser]);

  return (
    <div className="relative flex items-center justify-center w-full h-full">
      <canvas ref={canvasRef} className="w-full h-full" />
    </div>
  );
};

export default Visualizer;
