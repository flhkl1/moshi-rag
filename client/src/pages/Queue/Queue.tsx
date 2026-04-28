import moshiProcessorUrl from "../../audio-processor.ts?worker&url";
import { FC, useState, useCallback, useRef, MutableRefObject, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { Conversation } from "../Conversation/Conversation";
import { colors } from "../../theme/colors";

/** t ∈ [0,1]: breathing glow size for Start Session (accent green, matches theme) */
function queueStartGlowBoxShadow(t: number): string {
  const lerp = (a: number, b: number) => a + (b - a) * t;
  const r1 = lerp(4, 6);
  const r2 = lerp(6, 9);
  const r3 = lerp(8, 12);
  const o1 = lerp(0.5, 0.8);
  const o2 = lerp(0.2, 0.5);
  const o3 = lerp(0.1, 0.2);
  return `0 0 ${r1}px rgba(57,242,174,${o1}), 0 0 ${r2}px rgba(57,242,174,${o2}), 0 0 ${r3}px rgba(57,242,174,${o3})`;
}

export const Queue: FC = () => {
  const [searchParams] = useSearchParams();
  const overrideWorkerAddr = searchParams.get("worker_addr");
  const [hasMicrophoneAccess, setHasMicrophoneAccess] = useState<boolean>(false);
  const [showMicrophoneAccessMessage, setShowMicrophoneAccessMessage] = useState<boolean>(false);
  const [isInQueue, setIsInQueue] = useState<boolean>(false);
  const [queueError, setQueueError] = useState<string | null>(null);
  const [conversationKey, setConversationKey] = useState<number>(0);
  const [showConversation, setShowConversation] = useState<boolean>(false);
  /** True only after WS handshake; keeps queue layout on top until then (Conversation mounts underneath). */
  const [conversationHandshakeDone, setConversationHandshakeDone] = useState<boolean>(false);
  const retryIntervalRef = useRef<number | null>(null);

  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);

  /** 0–1 drives glow size; updated from rAF while the welcome Start Session button is shown */
  const [startSessionGlowT, setStartSessionGlowT] = useState(0);

  useEffect(() => {
    // Only animate the welcome "Start Session" button (not during active chat or queue wait)
    if (showConversation || isInQueue) {
      return;
    }
    let frameId: number;
    const periodMs = 1500;
    const start = performance.now();
    const tick = (now: number) => {
      const elapsed = (now - start) % periodMs;
      const t = (1 - Math.cos((elapsed / periodMs) * Math.PI * 2)) / 2;
      setStartSessionGlowT(t);
      frameId = requestAnimationFrame(tick);
    };
    frameId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameId);
  }, [showConversation, isInQueue]);

  const getMicrophoneAccess = useCallback(async () => {
    try {
      console.log("Requesting microphone access...");
      await window.navigator.mediaDevices.getUserMedia({ audio: true });
      console.log("Microphone access granted");
      setHasMicrophoneAccess(true);
      return true;
    } catch (e) {
      console.error("Microphone access denied:", e);
      setShowMicrophoneAccessMessage(true);
      setHasMicrophoneAccess(false);
    }
    return false;
  }, [setHasMicrophoneAccess, setShowMicrophoneAccessMessage]);

  const startProcessor = useCallback(async () => {
    console.log("Starting audio processor...");
    if (!audioContext.current) {
      console.log("Creating new AudioContext");
      audioContext.current = new AudioContext();
    }
    if (worklet.current) {
      console.log("Worklet already exists, skipping");
      return;
    }
    const ctx = audioContext.current;
    await ctx.resume();
    console.log("AudioContext resumed");
    try {
      console.log("Trying to create AudioWorkletNode...");
      worklet.current = new AudioWorkletNode(ctx, "moshi-processor");
      console.log("AudioWorkletNode created successfully");
    } catch (err) {
      console.log("AudioWorkletNode creation failed, loading module:", err);
      await ctx.audioWorklet.addModule(moshiProcessorUrl);
      worklet.current = new AudioWorkletNode(ctx, "moshi-processor");
      console.log("AudioWorkletNode created after loading module");
    }
    worklet.current.connect(ctx.destination);
    console.log("Worklet connected to destination");
  }, [audioContext, worklet]);

  const startRetryPolling = useCallback(() => {
    // Retry connection every 2 seconds when service is unavailable
    if (retryIntervalRef.current != null) {
      window.clearInterval(retryIntervalRef.current);
    }
    setIsInQueue(true);
    setQueueError(null);
    retryIntervalRef.current = window.setInterval(() => {
      // Increment key and show conversation to trigger connection attempt
      setConversationKey((prev) => {
        const newKey = prev + 1;
        setShowConversation(true);
        return newKey;
      });
    }, 2000);
  }, []);

  const stopRetryPolling = useCallback(() => {
    if (retryIntervalRef.current != null) {
      window.clearInterval(retryIntervalRef.current);
      retryIntervalRef.current = null;
    }
    setIsInQueue(false);
  }, [setIsInQueue]);

  /**
   * BFCache restore can bring back the in-memory React tree (capacity / connecting / chat).
   * Full normal reload already starts from welcome — this only handles persisted pageshow.
   */
  const resetSessionToWelcome = useCallback(() => {
    stopRetryPolling();
    setShowConversation(false);
    setConversationHandshakeDone(false);
    setIsInQueue(false);
    setQueueError(null);
    setHasMicrophoneAccess(false);
    setShowMicrophoneAccessMessage(false);
    setConversationKey((k) => k + 1);
    try {
      worklet.current?.disconnect();
    } catch {
      /* ignore */
    }
    worklet.current = null;
    void audioContext.current?.close().catch(() => undefined);
    audioContext.current = null;
  }, [stopRetryPolling]);

  useEffect(() => {
    const onPageShow = (e: PageTransitionEvent) => {
      if (e.persisted) {
        resetSessionToWelcome();
      }
    };
    window.addEventListener("pageshow", onPageShow);
    return () => window.removeEventListener("pageshow", onPageShow);
  }, [resetSessionToWelcome]);

  const onConnect = useCallback(async () => {
    console.log("onConnect called - starting processor and getting microphone access");
    await startProcessor();
    const micResult = await getMicrophoneAccess();
    console.log("Microphone access result:", micResult);
    if (!micResult) {
      return;
    }
    // Show conversation which will attempt to connect
    setShowConversation(true);
  }, [startProcessor]);

  const onConnectionFailed = useCallback(() => {
    console.log("Connection failed - showing queue message and starting retry");
    setShowConversation(false);
    setConversationHandshakeDone(false);
    startRetryPolling();
  }, [startRetryPolling]);

  const onConversationStart = useCallback(() => {
    console.log("Conversation started - stopping retry polling");
    setConversationHandshakeDone(true);
    stopRetryPolling();
  }, [stopRetryPolling]);

  useEffect(() => {
    return () => {
      stopRetryPolling();
    };
  }, [stopRetryPolling]);

  const effectiveWorkerAddr = overrideWorkerAddr ?? "";

  const canMountConversation =
    showConversation &&
    hasMicrophoneAccess &&
    audioContext.current != null &&
    worklet.current != null;

  const queueLayerVisible = !canMountConversation || !conversationHandshakeDone;

  return (
    <div className="relative h-screen w-screen overflow-hidden font-sans">
      {canMountConversation && (
        <div
          className={
            conversationHandshakeDone
              ? "fixed inset-0 z-0 h-full w-full"
              : "fixed inset-0 z-[5] h-full w-full opacity-0 pointer-events-none select-none"
          }
          aria-hidden={!conversationHandshakeDone}
        >
          <Conversation
            key={conversationKey}
            workerAddr={effectiveWorkerAddr}
            hideMainUIUntilConnected={!conversationHandshakeDone}
            audioContext={audioContext as MutableRefObject<AudioContext>}
            worklet={worklet as MutableRefObject<AudioWorkletNode>}
            onConversationEnd={() => {
              setShowConversation(false);
              setConversationHandshakeDone(false);
              stopRetryPolling();
            }}
            onConnectionFailed={onConnectionFailed}
            onConversationStart={onConversationStart}
          />
        </div>
      )}

      {queueLayerVisible && (
        <div
          className="absolute inset-0 z-20 flex h-full w-full flex-col items-center justify-center p-4"
          style={{ backgroundColor: colors.bgCanvas, color: colors.textPrimary }}
        >
          {/* Subtle Grid Background - matching Conversation page */}
          <div
            className="absolute inset-0 z-0 opacity-10"
            style={{
              backgroundImage:
                `linear-gradient(${colors.gridStroke} 1px, transparent 1px), linear-gradient(90deg, ${colors.gridStroke} 1px, transparent 1px)`,
              backgroundSize: "40px 40px",
            }}
          />

          <div className="max-w-md relative z-10">
        <div className="flex items-center justify-center mb-6">
          <a
            href="/"
            className="flex items-center gap-3 hover:opacity-90 transition-opacity"
            aria-label="Back to welcome page"
          >
            <img src="/assets/kyutai.jpeg" alt="Kyutai" className="h-12 w-12 object-contain" />
            <h1 className="text-5xl font-semibold tracking-tight text-center flex flex-row items-center gap-2" style={{ color: colors.accentGreenDark }}>
              Moshi
              <span className="text-[1.5rem] font-bold flex flex-col items-center leading-none" style={{ color: colors.accentBlue }}>
                <span>R</span>
                <span className="-mt-1">A</span>
                <span className="-mt-1">G</span>
              </span>
            </h1>
          </a>
        </div>
        {isInQueue ? (
          <div className="text-left">
            <p className="text-sm leading-relaxed mb-2" style={{ color: colors.textLight }}>
              We are currently at maximum capacity. All conversation slots are in use. Please stay on this page; we’ll connect you as soon as possible.
            </p>
            {queueError && (
              <p className="text-xs text-red-600 leading-relaxed mt-2">
                {queueError}
              </p>
            )}
          </div>
        ) : canMountConversation && !conversationHandshakeDone ? (
          <div className="text-left">
            <p className="text-sm leading-relaxed mb-2" style={{ color: colors.textLight }}>
              Connecting…
            </p>
          </div>
        ) : (
          <>
            <div className="text-left">
              <p className="text-sm leading-relaxed mb-2" style={{ color: colors.textLight }}>
                Meet <span className="font-bold" style={{ color: colors.accentGreenDark }}>Moshi</span>
                <span className="font-bold" style={{ color: colors.accentBlue }}>RAG</span>: A full-duplex conversational AI with Retrieval-Augmented Generation (RAG) capabilities.
              </p>
              <p className="text-sm leading-relaxed mb-2" style={{ color: colors.textLight }}>
                <span className="font-bold" style={{ color: colors.accentGreenDark }}>Moshi</span>
                <span className="font-bold" style={{ color: colors.accentBlue }}>RAG</span> listens and speaks simultaneously, creating a natural flow and a seamless communication experience.
              </p>
              <p className="text-sm leading-relaxed mb-2" style={{ color: colors.textLight }}>
                When things get complex, <span className="font-bold" style={{ color: colors.accentGreenDark }}>Moshi</span>
                <span className="font-bold" style={{ color: colors.accentBlue }}>RAG</span> triggers a back-end model to retrieve reference documents asynchronously, providing expert assistance without ever breaking the conversation.
              </p>
              <p className="text-sm leading-relaxed mb-2" style={{ color: colors.textLight }}>
                Ask it to plan your summer vacation, share a recipe for lasagna, or dive into a philosophical debate on the meaning of life.
              </p>
              <p className="text-sm leading-relaxed mb-2" style={{ color: colors.textLight }}>Conversations are limited to 5 minutes.</p>
              <p className="text-sm leading-relaxed mb-2" style={{ color: colors.textLight }}>
                While we strive for universal support, Google Chrome offers the best experience.
              </p>
              <p className="text-sm leading-relaxed mb-2" style={{ color: colors.textLight }}>
                Baked with &lt;3 @
                <a href="https://kyutai.org/" className="underline" style={{ color: colors.accentGreenDark }}>
                  Kyutai
                </a>
                .
              </p>
            </div>
            {showMicrophoneAccessMessage && (
              <p className="text-red-600 mb-4 text-sm text-left">
                Please enable your microphone before proceeding
              </p>
            )}
            <div className="text-center mt-6">
              <button
                type="button"
                onClick={async () => await onConnect()}
                className="px-8 py-3 text-white rounded-full text-base font-medium transition-transform hover:brightness-110 hover:scale-[1.02] active:scale-[0.99]"
                style={{
                  backgroundColor: colors.accentGreenDark,
                  boxShadow: queueStartGlowBoxShadow(startSessionGlowT),
                }}
              >
                Start Session
              </button>
            </div>
          </>
        )}
      </div>
      <div className="absolute bottom-8 text-center text-xs z-10" style={{ color: colors.textLight }}>
        <a
          target="_blank"
          rel="noreferrer"
          href="https://kyutai.org/moshi-terms.pdf"
          className="transition-colors hover:opacity-85"
        >
          Terms of Use
        </a>
        <span className="mx-2">•</span>
        <a
          target="_blank"
          rel="noreferrer"
          href="https://kyutai.org/moshi-privacy.pdf"
          className="transition-colors hover:opacity-85"
        >
          Privacy Policy
        </a>
      </div>
    </div>
      )}
    </div>
  );
};
