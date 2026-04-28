import { FC, MutableRefObject, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSocket } from "./hooks/useSocket";
import { SocketContext, useSocketContext } from "./SocketContext";
import { MediaContext } from "./MediaContext";
import { useServerAudio } from "./hooks/useServerAudio";
import { useUserAudio } from "./hooks/useUserAudio";
import { useServerText } from "./hooks/useServerText";
import { useRecording } from "./hooks/useRecording";
import { useRetrievalBackendChoice } from "./hooks/useRetrievalBackendChoice";
import Visualizer from "./components/Visualizer/Visualizer";
import TranscriptionPanel from "./components/TranscriptionPanel/TranscriptionPanel";
import SearchPanel from "./components/SearchPanel/SearchPanel";
import PipeAnimation from "./components/PipeAnimation/PipeAnimation";
import { SessionFeedbackModal } from "./components/SessionFeedbackModal";
import { ServerAudioStats } from "./components/ServerAudio/ServerAudioStats";
import { AudioStats } from "./hooks/useServerAudio";
import { colors } from "../../theme/colors";
import type { WSMessage } from "../../protocol/types";
import {
  parseRetrievalCapabilities,
  type ParsedRetrievalCapabilities,
} from "./utils/retrievalCapabilities";

type ConversationProps = {
  workerAddr: string;
  workerAuthId?: string;
  sessionAuthId?: string;
  sessionId?: number;
  email?: string;
  audioContext: MutableRefObject<AudioContext>;
  worklet: MutableRefObject<AudioWorkletNode>;
  /** When the session ends (WS closed), notify queue so the slot is free while feedback stays open. */
  onFeedbackFlowReleaseSlot?: () => void;
  onConversationEnd?: () => void;
  onConnectionFailed?: () => void;
  onConversationStart?: () => void;
  isBypass?: boolean;
  /** When true, full chat chrome stays hidden until the WebSocket handshake succeeds. */
  hideMainUIUntilConnected?: boolean;
};

const SESSION_MAX_DURATION_MS = 301 * 1000; // 5 minutes

const buildURL = ({
  workerAddr,
  workerAuthId,
  email,
}: {
  workerAddr: string;
  workerAuthId?: string;
  email?: string;
}) => {
  if (workerAddr == "same" || workerAddr == "") {
    workerAddr = window.location.hostname + ":" + window.location.port;
    console.log("Overriding workerAddr to", workerAddr);
  }
  const wsProtocol = (window.location.protocol === 'https:') ? 'wss' : 'ws';
  const url = new URL(`${wsProtocol}://${workerAddr}/api/chat`);
  if (workerAuthId) {
    url.searchParams.append("worker_auth_id", workerAuthId);
  }
  if (email) {
    url.searchParams.append("email", email);
  }
  console.log(url.toString());
  return url.toString();
};

export const Conversation: FC<ConversationProps> = ({
  workerAddr,
  workerAuthId,
  audioContext,
  worklet,
  onFeedbackFlowReleaseSlot,
  onConversationEnd,
  onConnectionFailed,
  onConversationStart,
  isBypass = false,
  email,
  hideMainUIUntilConnected = false,
}) => {
  const getAudioStats = useRef<() => AudioStats>(() => ({
    playedAudioDuration: 0,
    missedAudioDuration: 0,
    totalAudioMessages: 0,
    delay: 0,
    minPlaybackDelay: 0,
    maxPlaybackDelay: 0,
  }));
  const [isOver, setIsOver] = useState(false);
  const conversationStartAtRef = useRef<number | null>(null);
  /** Wall-clock session length in ms; set on WS disconnect (ref avoids stale submit closures). */
  const conversationDurationMs = useRef<number | null>(null);
  const [showAudioStats, setShowAudioStats] = useState(false);
  const micDuration = useRef<number>(0);
  const actualAudioPlayed = useRef<number>(0);
  const audioStreamDestination = useRef<MediaStreamAudioDestinationNode>(audioContext.current.createMediaStreamDestination());
  const visualizerCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const isRecordingBridgeConnectedRef = useRef(false);

  const WSURL = useMemo(() => buildURL({
    workerAddr,
    workerAuthId,
    email: email,
  }), [workerAddr, workerAuthId, email]);

  const [retrievalCapabilities, setRetrievalCapabilities] =
    useState<ParsedRetrievalCapabilities | null>(null);

  const onDisconnect = useCallback(() => {
    // Only record once per session; stop() may invoke this before socket.close() fires again.
    if (
      conversationDurationMs.current == null &&
      conversationStartAtRef.current != null
    ) {
      conversationDurationMs.current =
        Date.now() - conversationStartAtRef.current;
    }
    setRetrievalCapabilities(null);
    setIsOver(true);
  }, []);

  const onSocketMessage = useCallback((message: WSMessage) => {
    if (message.type !== "metadata") {
      return;
    }
    const parsed = parseRetrievalCapabilities(message.data);
    if (parsed) {
      setRetrievalCapabilities(parsed);
    }
  }, []);

  const [handshakeCompleted, setHandshakeCompleted] = useState(false);
  const { isConnected, sendMessage, socket, start, stop } = useSocket({
    uri: WSURL,
    onMessage: onSocketMessage,
    onDisconnect,
    onError: (error) => {
      console.warn("Received socket error message:", error);
    },
    onConnectionError: (reason) => {
      console.warn("Connection failed:", reason);
      if (onConnectionFailed) {
        onConnectionFailed();
      }
    },
    onConnectionSuccess: () => {
      console.log("Connection successful");
      if (onConversationStart) {
        onConversationStart();
      }
    },
  });

  useEffect(() => {
    if (isConnected) {
      setHandshakeCompleted(true);
    }
  }, [isConnected]);

  const showMainChrome =
    !hideMainUIUntilConnected ||
    isConnected ||
    (isOver && handshakeCompleted);

  // Auto-connect when component mounts (only once)
  const hasAutoConnectedRef = useRef(false);
  useEffect(() => {
    if (!hasAutoConnectedRef.current) {
      console.log("Conversation component mounted, auto-connecting...");
      hasAutoConnectedRef.current = true;
      conversationStartAtRef.current = Date.now();
      conversationDurationMs.current = null;
      start();
    }
    return () => {
      if (hasAutoConnectedRef.current) {
        console.log("Conversation component unmounting, disconnecting...");
        hasAutoConnectedRef.current = false;
        conversationStartAtRef.current = null;
        stop();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run once on mount

  /** If disconnect ran in an edge case without writing duration, fill it when the session ends. */
  useEffect(() => {
    if (!isOver) {
      return;
    }
    if (conversationDurationMs.current != null) {
      return;
    }
    if (conversationStartAtRef.current == null) {
      return;
    }
    conversationDurationMs.current =
      Date.now() - conversationStartAtRef.current;
  }, [isOver]);

  const startRecording = useCallback(() => {
    if (isRecordingBridgeConnectedRef.current) {
      return;
    }
    worklet.current?.connect(audioStreamDestination.current);
    isRecordingBridgeConnectedRef.current = true;
  }, [audioStreamDestination, worklet]);

  const stopRecordingCallback = useCallback(() => {
    if (!isRecordingBridgeConnectedRef.current) {
      return;
    }
    worklet.current?.disconnect(audioStreamDestination.current);
    isRecordingBridgeConnectedRef.current = false;
  }, [audioStreamDestination, worklet]);

  return (
    <SocketContext.Provider
      value={{
        isConnected,
        sendMessage,
        socket,
        retrievalCapabilities,
      }}
    >
      <MediaContext.Provider value={
        {
          startRecording,
          stopRecording: stopRecordingCallback,
          audioContext,
          worklet,
          audioStreamDestination,
          visualizerCanvasRef,
          micDuration,
          actualAudioPlayed,
        }
      }>
        <ConversationContent
          workerAddr={workerAddr}
          isOver={isOver}
          isBypass={isBypass}
          handshakeCompleted={handshakeCompleted}
          showMainChrome={showMainChrome}
          onFeedbackFlowReleaseSlot={onFeedbackFlowReleaseSlot}
          onConversationEnd={onConversationEnd}
          conversationDurationMs={conversationDurationMs}
          audioContext={audioContext}
          visualizerCanvasRef={visualizerCanvasRef}
          start={start}
          stop={stop}
          getAudioStats={getAudioStats}
          showAudioStats={showAudioStats}
          setShowAudioStats={setShowAudioStats}
        />
      </MediaContext.Provider>
    </SocketContext.Provider >
  );
};

// Inner component that uses MediaContext-dependent hooks
// Moved outside Conversation to prevent remounting on state changes
const ConversationContent: FC<{
  workerAddr: string;
  isOver: boolean;
  isBypass: boolean;
  handshakeCompleted: boolean;
  showMainChrome: boolean;
  onFeedbackFlowReleaseSlot?: () => void;
  onConversationEnd?: () => void;
  conversationDurationMs: MutableRefObject<number | null>;
  audioContext: MutableRefObject<AudioContext>;
  visualizerCanvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  start: () => void;
  stop: () => void;
  getAudioStats: React.MutableRefObject<() => AudioStats>;
  showAudioStats: boolean;
  setShowAudioStats: (show: boolean) => void;
}> = ({
  workerAddr,
  isOver,
  isBypass,
  handshakeCompleted,
  showMainChrome,
  onFeedbackFlowReleaseSlot,
  onConversationEnd,
  conversationDurationMs,
  audioContext: audioContextProp,
  visualizerCanvasRef,
  start,
  stop,
  getAudioStats: getAudioStatsProp,
  showAudioStats,
  setShowAudioStats,
}) => {
    const { isConnected, sendMessage } = useSocketContext();

    // Text/RAG hooks (need SocketContext - must be inside provider)
    const {
      messages,
      isRetrieving,
      searchResult,
      isFreshResult,
      isResultDisplayActive,
      isRecovering,
    } = useServerText();

    const {
      retrievalBackends,
      selectedRetrievalId,
      setSelectedRetrievalId,
    } = useRetrievalBackendChoice();

    // Audio hooks (need MediaContext - will call useMediaContext internally)
    const { analyser: serverAnalyser, hasCriticalDelay, setHasCriticalDelay } = useServerAudio({
      setGetAudioStats: (callback) => {
        getAudioStatsProp.current = callback;
      },
    });

    const [userAnalyser, setUserAnalyser] = useState<AnalyserNode | null>(null);
    const recordingStartedRef = useRef(false);

    // Refs for pipe animation
    const visualizerWrapperRef = useRef<HTMLDivElement>(null);
    const searchPanelRef = useRef<HTMLDivElement>(null);

    // Left panel (retrieval) collapsed by default; orb centered, pipes hidden
    const [showRetrievalPanel, setShowRetrievalPanel] = useState(false);
    const [showFeedbackModal, setShowFeedbackModal] = useState(false);
    const releasedSlotForFeedbackRef = useRef(false);

    useEffect(() => {
      if (isOver && !isBypass && handshakeCompleted) {
        setShowFeedbackModal(true);
      }
    }, [isOver, isBypass, handshakeCompleted]);

    useEffect(() => {
      if (!isOver || isBypass || !onFeedbackFlowReleaseSlot) {
        return;
      }
      if (releasedSlotForFeedbackRef.current) {
        return;
      }
      releasedSlotForFeedbackRef.current = true;
      onFeedbackFlowReleaseSlot();
    }, [isOver, isBypass, onFeedbackFlowReleaseSlot]);

    const dismissFeedbackAndLeave = useCallback(() => {
      setShowFeedbackModal(false);
      onConversationEnd?.();
    }, [onConversationEnd]);

    const closeFeedbackModalOnly = useCallback(() => {
      setShowFeedbackModal(false);
    }, []);

    // Only show pipe animation on desktop; on mobile the stacked layout makes pipe direction wrong
    const [isDesktop, setIsDesktop] = useState(() => typeof window !== "undefined" && window.matchMedia("(min-width: 768px)").matches);
    useEffect(() => {
      const mql = window.matchMedia("(min-width: 768px)");
      const handler = () => setIsDesktop(mql.matches);
      mql.addEventListener("change", handler);
      return () => mql.removeEventListener("change", handler);
    }, []);

    // Recording hooks
    const transcriptTextForRecording = useMemo(
      () =>
        messages
          .map((m) => `${m.role === "user" ? "User" : "Moshi"}: ${m.text}`)
          .join("\n"),
      [messages],
    );
    const retrievalTextForRecording = useMemo(
      () => searchResult?.result ?? "",
      [searchResult],
    );
    const saveTranscriptJson = useCallback(() => {
      const turns = messages.map((msg, index) => ({
        turn: index + 1,
        speaker: msg.role === "user" ? "You" : "Moshi",
        role: msg.role,
        text: msg.text.trim(),
        reference: msg.referenceText?.trim() || null,
        referenceLm: msg.referenceLmLabel?.trim() || null,
        timestamp: msg.timestamp instanceof Date ? msg.timestamp.toISOString() : String(msg.timestamp),
      }));
      const retrieval =
        retrievalBackends.length >= 1
          ? {
              methodId: selectedRetrievalId ?? null,
              methodsAvailable: retrievalBackends.map((b) => b.id),
            }
          : null;
      const payload = {
        exportedAt: new Date().toISOString(),
        totalTurns: turns.length,
        transcript: turns,
        retrieval,
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `moshirag-transcript-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, [messages, retrievalBackends, selectedRetrievalId]);

    const {
      startAudioRecording,
      stopAudioRecording,
      saveAudio,
      startVideoRecording,
      stopVideoRecording,
      saveVideo,
      hasAudioRecording,
      hasVideoRecording,
    } = useRecording({
      transcriptText: transcriptTextForRecording,
      retrievalText: retrievalTextForRecording,
    });
    const autoCaptureStartedRef = useRef(false);
    const { startRecordingUser, stopRecording } = useUserAudio({
      constraints: {
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
        video: false,
      },
      onDataChunk: (chunk: Uint8Array) => {
        if (!isConnected) {
          return;
        }
        // Copy: encoder may transfer ArrayBuffer ownership; avoid sending a reused view.
        sendMessage({
          type: "audio",
          data: new Uint8Array(chunk),
        });
      },
      onRecordingStart: () => {
        console.log("Recording started");
      },
      onRecordingStop: () => {
        console.log("Recording stopped");
      },
    });

    useEffect(() => {
      if (isConnected && !recordingStartedRef.current) {
        recordingStartedRef.current = true;
        startRecordingUser().then(result => {
          if (result) {
            setUserAnalyser(result.analyser);
          }
        });
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isConnected]);

    /** Stop mic when disconnected unless we're in post-session feedback (keep graph for saves). */
    useEffect(() => {
      if (!isConnected && recordingStartedRef.current) {
        if (isOver && !isBypass) {
          return;
        }
        recordingStartedRef.current = false;
        stopRecording();
      }
    }, [isConnected, isOver, isBypass, stopRecording]);

    useEffect(() => {
      return () => {
        if (recordingStartedRef.current) {
          recordingStartedRef.current = false;
          stopRecording();
        }
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
      if (!isConnected) {
        return;
      }
      const timeout = window.setTimeout(() => {
        console.log("Session reached 5 minute limit, disconnecting");
        stop();
      }, SESSION_MAX_DURATION_MS);
      return () => {
        window.clearTimeout(timeout);
      };
    }, [isConnected, stop]);

    useEffect(() => {
      if (isConnected && !autoCaptureStartedRef.current) {
        autoCaptureStartedRef.current = true;
        startAudioRecording();
        startVideoRecording();
        return;
      }
      if (!isConnected && autoCaptureStartedRef.current) {
        autoCaptureStartedRef.current = false;
        void stopAudioRecording();
        void stopVideoRecording();
      }
    }, [isConnected, startAudioRecording, startVideoRecording, stopAudioRecording, stopVideoRecording]);

    if (!showMainChrome) {
      return (
        <div
          className="relative flex flex-col h-screen w-screen items-center justify-center overflow-hidden font-sans px-6"
          style={{ backgroundColor: colors.bgCanvas, color: colors.textPrimary }}
        >
          <div
            className="absolute inset-0 z-0 opacity-10"
            style={{
              backgroundImage: `linear-gradient(${colors.gridStroke} 1px, transparent 1px), linear-gradient(90deg, ${colors.gridStroke} 1px, transparent 1px)`,
              backgroundSize: "40px 40px",
            }}
          />
          <p className="relative z-10 text-center text-sm leading-relaxed max-w-sm" style={{ color: colors.textLight }}>
            Connecting…
          </p>
        </div>
      );
    }

    return (
      <div className="flex flex-col h-screen overflow-hidden font-sans" style={{ backgroundColor: colors.bgCanvas, color: colors.textPrimary }}>
        <SessionFeedbackModal
          open={showFeedbackModal}
          workerAddr={workerAddr}
          onClose={closeFeedbackModalOnly}
          onDismiss={closeFeedbackModalOnly}
          conversationDurationMs={conversationDurationMs}
        />
        {/* Header */}
        <header className="flex-none h-16 border-b flex items-center justify-between px-6 backdrop-blur-md z-50 relative" style={{ borderColor: colors.border, backgroundColor: colors.bgHeader }}>
          <a
            href="/"
            className="flex items-center gap-3 hover:opacity-90 transition-opacity"
            aria-label="Back to welcome page"
          >
            <img src="/assets/kyutai.jpeg" alt="Kyutai" className="h-8 w-8 object-contain" />
            <h1 className="text-lg font-semibold tracking-tight flex flex-row items-center gap-1" style={{ color: colors.accentGreenDark }}>
              Moshi
              <span className="text-xs font-bold flex flex-col items-center leading-none" style={{ color: colors.accentBlue }}>
                <span>R</span>
                <span className="-mt-0.5">A</span>
                <span className="-mt-0.5">G</span>
              </span>
            </h1>
          </a>

          <div className="flex items-center gap-4">
            {isOver && !isBypass && (
              <button
                type="button"
                onClick={() => {
                  const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
                  if (!isIOS) {
                    stop();
                    dismissFeedbackAndLeave();
                    return;
                  }
                  document.location.reload();
                }}
                className="px-4 py-2 rounded-full text-sm font-medium text-white transition-transform hover:brightness-110 hover:scale-[1.02] active:scale-[0.99] disabled:opacity-50"
                style={{ backgroundColor: colors.accentGreenDark }}
              >
                Start Over
              </button>
            )}

            {(!isOver || isBypass) && (
              <button
                type="button"
                onClick={() => {
                  audioContextProp.current.resume();
                  if (isConnected) {
                    stop();
                    if (isBypass) {
                      onConversationEnd?.();
                    }
                    return;
                  }
                  start();
                }}
                className={`px-6 py-2 rounded-full text-sm font-medium transition-transform enabled:hover:brightness-110 enabled:hover:scale-[1.02] enabled:active:scale-[0.99] ${isConnected
                  ? 'bg-red-50 text-red-600 border border-red-200 hover:bg-red-100'
                  : 'text-white'
                  }`}
                style={!isConnected ? { backgroundColor: colors.accentGreenDark } : undefined}
              >
                {!isConnected ? "Start Session" : "End Session"}
              </button>
            )}

            <div className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-600' : 'bg-red-600'}`} />

            {/* Audio Stats Toggle - hidden on mobile */}
            <button
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setShowAudioStats(!showAudioStats);
              }}
              className="hidden md:block p-2 transition-colors hover:opacity-80"
              style={{ color: colors.textLight }}
              title="Toggle Audio Stats"
              type="button"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </button>

          </div>
        </header>

        {/* Main Content - column stack on mobile (orb → left panel → transcript), side-by-side on md+; mobile: min-height so content can extend and user can scroll to transcript */}
        <main className="flex-1 relative flex flex-col md:flex-row p-4 md:p-6 gap-4 md:gap-6 min-h-[calc(100vh-64px)] md:min-h-0 md:h-[calc(100vh-64px)] overflow-y-auto md:overflow-hidden">
          {/* Subtle Grid */}
          <div className="absolute inset-0 z-0 opacity-10"
            style={{
              backgroundImage: `linear-gradient(${colors.gridStroke} 1px, transparent 1px), linear-gradient(90deg, ${colors.gridStroke} 1px, transparent 1px)`,
              backgroundSize: '40px 40px'
            }}>
          </div>

          {/* Pipe Animation Overlay - only when retrieval panel is shown and on desktop (mobile stacked layout draws pipes wrong) */}
          {showRetrievalPanel && isDesktop && (
            <PipeAnimation
              visualizerWrapperRef={visualizerWrapperRef}
              searchPanelRef={searchPanelRef}
              isRetrieving={isRetrieving}
              isFreshResult={isFreshResult}
              isRecovering={isRecovering}
            />
          )}

          {/* Left Column - full width on mobile (orb then retrieval), 50% on md+; on mobile when collapsed use min-height so orb isn't cut off */}
          <div className={`relative flex flex-col w-full md:w-1/2 h-auto md:h-full min-h-0 shrink-0 md:shrink z-20 pointer-events-none overflow-visible ${showRetrievalPanel ? '' : 'items-center justify-center md:justify-center min-h-[360px] md:min-h-0'}`}>
            {isOver && (
              <div className="absolute top-4 left-1/2 -translate-x-1/2 z-40 pointer-events-auto flex items-center gap-2">
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    saveAudio();
                  }}
                  className="px-3 py-2 text-xs font-medium rounded-full transition-transform enabled:hover:brightness-110 enabled:hover:scale-[1.02] enabled:active:scale-[0.99]"
                  style={{
                    color: hasAudioRecording ? colors.textPrimary : colors.textDisabled,
                    backgroundColor: hasAudioRecording ? colors.black90 : colors.black70,
                  }}
                  title="Save audio recording"
                  type="button"
                  disabled={!hasAudioRecording}
                >
                  Save Audio
                </button>
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    saveVideo();
                  }}
                  className="px-3 py-2 text-xs font-medium rounded-full transition-transform enabled:hover:brightness-110 enabled:hover:scale-[1.02] enabled:active:scale-[0.99]"
                  style={{
                    color: hasVideoRecording ? colors.textPrimary : colors.textDisabled,
                    backgroundColor: hasVideoRecording ? colors.black90 : colors.black70,
                  }}
                  title="Save video recording"
                  type="button"
                  disabled={!hasVideoRecording}
                >
                  Save Video
                </button>
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    saveTranscriptJson();
                  }}
                  className="px-3 py-2 text-xs font-medium rounded-full transition-transform enabled:hover:brightness-110 enabled:hover:scale-[1.02] enabled:active:scale-[0.99]"
                  style={{
                    color: messages.length > 0 ? colors.textPrimary : colors.textDisabled,
                    backgroundColor: messages.length > 0 ? colors.black90 : colors.black70,
                  }}
                  title="Save transcript as JSON"
                  type="button"
                  disabled={messages.length === 0}
                >
                  Save JSON
                </button>
              </div>
            )}
            {/* Visualizer Orb - on PC when panel expanded allow shrinking with viewport height so orb + panel fit */}
            <div ref={visualizerWrapperRef} className={`pointer-events-auto relative aspect-square flex items-center justify-center max-w-full min-w-0 min-h-0 ${showRetrievalPanel ? 'flex-shrink mx-auto mb-4 w-[360px] md:w-[min(360px,40vh)]' : 'flex-shrink w-[360px] md:w-[min(360px,40vh)]'}`}>
              <Visualizer
                userAnalyser={userAnalyser}
                serverAnalyser={serverAnalyser?.current || null}
                isActive={isConnected}
                isRetrieving={isRetrieving}
                isFreshResult={isFreshResult}
                isRecovering={isRecovering}
                captureCanvasRef={visualizerCanvasRef}
              />
            </div>

            {/* Retrieval block (Hide button + panel) - negative margin so only this moves up, orb stays put */}
            {showRetrievalPanel && (
              <div className="pointer-events-auto flex flex-col flex-1 min-h-0 -mt-12 min-h-[30vh] md:min-h-0 shrink-0">
                {/* Hide Retrieval - above the retrieval panel, no border */}
                <button
                  type="button"
                  onClick={() => setShowRetrievalPanel(false)}
                  className="inline-flex flex-col items-center gap-0.5 self-end mb-1 mr-3 cursor-pointer border-0 bg-transparent p-0 text-xs font-medium transition-transform hover:brightness-110 hover:scale-[1.02] active:scale-[0.99] focus:outline-none focus-visible:underline"
                  style={{ color: colors.textLight }}
                  title="Hide retrieval panel"
                >
                  <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                    <path d="M6 8l6 6 6-6" />
                    <path d="M6 14l6 6 6-6" />
                  </svg>
                  <span>Hide Retrieval</span>
                </button>

                {/* Search Panel */}
                <div className="w-full flex-1 min-h-0 relative z-30">
                  <SearchPanel
                    ref={searchPanelRef}
                    result={searchResult}
                    isRetrieving={isRetrieving}
                    hasActiveText={isResultDisplayActive}
                    isFreshResult={isFreshResult}
                    isRecovering={isRecovering}
                    retrievalBackends={retrievalBackends}
                    selectedRetrievalId={selectedRetrievalId}
                    onRetrievalChange={setSelectedRetrievalId}
                  />
                </div>
              </div>
            )}
            {/* Show Retrieval - below orb in flow on mobile, absolute bottom-center of left column on md+ */}
            {!showRetrievalPanel && (
              <button
                type="button"
                onClick={() => setShowRetrievalPanel(true)}
                className="z-40 pointer-events-auto inline-flex flex-col items-center gap-0.5 cursor-pointer border-0 bg-transparent p-0 text-xs font-medium transition-transform md:absolute md:bottom-6 md:left-1/2 md:-translate-x-1/2 shrink-0 hover:brightness-110 hover:scale-[1.02] active:scale-[0.99] focus:outline-none focus-visible:underline"
                style={{ color: colors.textLight }}
                title="Show retrieval panel"
              >
                <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                  <path d="M6 10l6-6 6 6" />
                  <path d="M6 16l6-6 6 6" />
                </svg>
                <span>Show Retrieval</span>
              </button>
            )}
          </div>

          {/* Right Panel - Transcript; full width on mobile below left column (z-0 so not on top of orb/retrieval), 50% on md+ */}
          <div className="w-full md:w-1/2 flex-1 min-h-0 md:h-full z-0 md:z-30 pointer-events-none flex flex-col min-h-[50vh] md:min-h-0 shrink-0">
            <div className="pointer-events-auto h-full min-h-0 w-full flex flex-col">
              <TranscriptionPanel messages={messages} />
            </div>
          </div>

          {/* Error Display */}
          {hasCriticalDelay && (
            <div className="absolute bottom-10 left-1/2 transform -translate-x-1/2 bg-red-50 border border-red-200 text-red-600 px-4 py-2 rounded-lg text-sm shadow-lg pointer-events-auto z-40">
              A connection issue has been detected, you've been reconnected
              <button
                onClick={() => setHasCriticalDelay(false)}
                className="ml-4 px-2 py-1 bg-white text-red-600 rounded hover:bg-red-50"
              >
                Dismiss
              </button>
            </div>
          )}

          {/* Audio Stats Panel (Collapsible) */}
          {showAudioStats && (
            <div className="fixed top-[72x] right-6 backdrop-blur-md border rounded-lg p-4 shadow-lg z-40 pointer-events-auto max-w-xs" style={{ backgroundColor: colors.black95, borderColor: colors.border }}>
              <div className="flex justify-between items-center mb-2">
                <h3 className="text-sm font-semibold" style={{ color: colors.textPrimary }}>Audio Stats</h3>
                <button
                  onClick={() => setShowAudioStats(false)}
                  className="hover:opacity-80"
                  style={{ color: colors.textLight }}
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <ServerAudioStats getAudioStats={getAudioStatsProp} />
            </div>
          )}
        </main>
      </div>
    );
  };
