import { useCallback, useEffect, useRef, useState } from "react";
import { useSocketContext } from "../SocketContext";
import { decodeMessage } from "../../../protocol/encoder";

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  text: string;
  timestamp: Date;
  referenceText?: string;
  /** LM id from referencetext payload (segment before first tab). */
  referenceLmLabel?: string;
}

export interface SearchResult {
  id: string;
  result: string;
  /** LM id is the segment before the first tab in referencetext payloads. */
  lmLabel?: string;
}

function parseReferenceMessageData(raw: string): { text: string; lmLabel?: string } {
  const payload = raw.trim();
  const tabIdx = payload.indexOf("\t");
  if (tabIdx === -1) {
    return { text: payload };
  }
  const lm = payload.slice(0, tabIdx).trim();
  const text = payload.slice(tabIdx + 1);
  return {
    text,
    lmLabel: lm.length > 0 ? lm : undefined,
  };
}

export const useServerText = () => {
  // Legacy state (kept for backward compatibility if needed)
  const [text, setText] = useState<string[]>([]);
  const [textColor] = useState<number[]>([]);
  const [textType] = useState<string[]>([]);
  const [totalTextMessages] = useState(0);

  // New RAG state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isRetrieving, setIsRetrieving] = useState<boolean>(false);
  const [searchResult, setSearchResult] = useState<SearchResult | null>(null);
  const [isFreshResult, setIsFreshResult] = useState<boolean>(false);
  const [isResultDisplayActive, setIsResultDisplayActive] = useState<boolean>(false);
  const [isRecovering, setIsRecovering] = useState<boolean>(false);

  // Refs for timers
  const glowTimeoutRef = useRef<number | null>(null);
  const recoveryEndTimeoutRef = useRef<number | null>(null);
  const textDisappearTimeoutRef = useRef<number | null>(null);

  // Buffer for reference when last message is not a model message
  const bufferedReferenceTextRef = useRef<string | null>(null);
  const bufferedReferenceLmLabelRef = useRef<string | null>(null);

  const { socket } = useSocketContext();

  const stripRolePrefix = (text: string, prefix: string): string => {
    const trimmed = text;
    const prefixRegex = new RegExp(`^\\s*${prefix}:\\s+`, 'i');
    if (prefixRegex.test(trimmed)) {
      return trimmed.replace(prefixRegex, '');
    }
    return trimmed;
  };

  const onSocketMessage = useCallback((e: MessageEvent) => {
    try {
      const dataArray = new Uint8Array(e.data);
      const message = decodeMessage(dataArray);

      if (message.type === "coloredtext") {
        const role: 'user' | 'model' | null = message.color === 4 ? 'model' : message.color === 10 ? 'user' : null;

        if (!message.data || message.data.length === 0) {
          return;
        }

        if (role === 'model') {
          if (message.data === "[RET]") {
            if (glowTimeoutRef.current) window.clearTimeout(glowTimeoutRef.current);
            if (recoveryEndTimeoutRef.current) window.clearTimeout(recoveryEndTimeoutRef.current);
            if (textDisappearTimeoutRef.current) window.clearTimeout(textDisappearTimeoutRef.current);
            setIsRetrieving(true);
            setIsFreshResult(false);
            setIsRecovering(false);
            setIsResultDisplayActive(false);
            return;
          }
          if (message.data === "[RET_FAILED]") {
            if (glowTimeoutRef.current) window.clearTimeout(glowTimeoutRef.current);
            if (recoveryEndTimeoutRef.current) window.clearTimeout(recoveryEndTimeoutRef.current);
            if (textDisappearTimeoutRef.current) window.clearTimeout(textDisappearTimeoutRef.current);
            setIsRetrieving(false);
            setIsFreshResult(false);
            setIsRecovering(false);
            setIsResultDisplayActive(false);
            return;
          }

          const processedData = stripRolePrefix(message.data, 'moshi');
          const bufferedRefText = bufferedReferenceTextRef.current;
          const bufferedLm = bufferedReferenceLmLabelRef.current;

          setMessages(prev => {
            const last = prev[prev.length - 1];
            if (last && last.role === 'model') {
              const processedLastText = stripRolePrefix(last.text, 'moshi');
              const refText = bufferedRefText || last.referenceText;
              const refLm = bufferedRefText
                ? bufferedLm && bufferedLm.length > 0
                  ? bufferedLm
                  : undefined
                : last.referenceLmLabel;
              const updated: ChatMessage[] = [...prev.slice(0, -1), {
                id: last.id,
                role: last.role,
                text: (processedLastText + processedData),
                timestamp: last.timestamp,
                referenceText: refText,
                referenceLmLabel: refLm,
              }];
              if (bufferedRefText) {
                bufferedReferenceTextRef.current = null;
                bufferedReferenceLmLabelRef.current = null;
              }
              return updated;
            } else {
              const updated: ChatMessage[] = [...prev, {
                id: Date.now().toString() + '-m',
                role: 'model' as const,
                text: processedData,
                timestamp: new Date(),
                referenceText: bufferedRefText || undefined,
                referenceLmLabel:
                  bufferedRefText && bufferedLm && bufferedLm.length > 0 ? bufferedLm : undefined,
              }];
              if (bufferedRefText) {
                bufferedReferenceTextRef.current = null;
                bufferedReferenceLmLabelRef.current = null;
              }
              return updated;
            }
          });
        } else if (role === 'user') {
          const processedData = stripRolePrefix(message.data, 'user');

          setMessages(prev => {
            const last = prev[prev.length - 1];
            if (last && last.role === 'user') {
              const processedLastText = stripRolePrefix(last.text, 'user');
              return [...prev.slice(0, -1), {
                id: last.id,
                role: last.role,
                text: (processedLastText + processedData),
                timestamp: last.timestamp,
                referenceText: last.referenceText,
                referenceLmLabel: last.referenceLmLabel,
              }];
            } else {
              return [...prev, {
                id: Date.now().toString() + '-u',
                role: 'user' as const,
                text: processedData,
                timestamp: new Date()
              }];
            }
          });
        }
      } else if (message.type === "coloredreferencetext") {
        if (message.data === "\0") {
          return;
        }

        setTimeout(() => {
          const { text: resultText, lmLabel } = parseReferenceMessageData(message.data);

          const newResult: SearchResult = {
            id: Date.now().toString(),
            result: resultText,
            ...(lmLabel != null ? { lmLabel } : {}),
          };

          setSearchResult(newResult);
          setIsRetrieving(false);
          setIsFreshResult(true);
          setIsResultDisplayActive(true);
          setIsRecovering(false);

          if (glowTimeoutRef.current) window.clearTimeout(glowTimeoutRef.current);
          if (recoveryEndTimeoutRef.current) window.clearTimeout(recoveryEndTimeoutRef.current);
          if (textDisappearTimeoutRef.current) window.clearTimeout(textDisappearTimeoutRef.current);

          // After 5s: start recovery (glow fades over 7s); text fades over 5s then is cleared
          glowTimeoutRef.current = window.setTimeout(() => {
            setIsFreshResult(false);
            setIsRecovering(true);
            // Text disappears over 5s (CSS fade); clear from state after 5s
            textDisappearTimeoutRef.current = window.setTimeout(() => {
              setSearchResult(null);
              setIsResultDisplayActive(false);
            }, 5000);
            // Glow recovery ends after 7s
            recoveryEndTimeoutRef.current = window.setTimeout(() => {
              setIsRecovering(false);
            }, 9000);
          }, 5000);


          setMessages(prev => {
            const last = prev[prev.length - 1];
            if (last && last.role === 'model') {
              return [
                ...prev.slice(0, -1),
                {
                  ...last,
                  referenceText: resultText,
                  referenceLmLabel: lmLabel,
                },
              ];
            } else {
              bufferedReferenceTextRef.current = resultText;
              bufferedReferenceLmLabelRef.current =
                lmLabel && lmLabel.length > 0 ? lmLabel : null;
              return prev;
            }
          });
        }, 100);
      }
    } catch (error) {
      console.error("Error processing message in useServerText:", error);
    }
  }, []);

  useEffect(() => {
    const currentSocket = socket;
    if (!currentSocket) {
      return;
    }
    setText([]);
    setMessages([]);
    setIsRetrieving(false);
    setSearchResult(null);
    setIsFreshResult(false);
    setIsResultDisplayActive(false);
    bufferedReferenceTextRef.current = null;
    bufferedReferenceLmLabelRef.current = null;

    if (glowTimeoutRef.current) window.clearTimeout(glowTimeoutRef.current);
    if (recoveryEndTimeoutRef.current) window.clearTimeout(recoveryEndTimeoutRef.current);
    if (textDisappearTimeoutRef.current) window.clearTimeout(textDisappearTimeoutRef.current);

    currentSocket.addEventListener("message", onSocketMessage);
    return () => {
      currentSocket.removeEventListener("message", onSocketMessage);
    };
  }, [socket, onSocketMessage]);

  return {
    text,
    textColor,
    textType,
    totalTextMessages,
    messages,
    isRetrieving,
    searchResult,
    isFreshResult,
    isResultDisplayActive,
    isRecovering,
  };
};
