import { useState, useEffect, useCallback, useRef } from "react";
import { WSMessage } from "../../../protocol/types";
import { decodeMessage, encodeMessage } from "../../../protocol/encoder";

export const useSocket = ({
  onMessage,
  uri,
  onDisconnect: onDisconnectProp,
  onError,
  onConnectionError,
  onConnectionSuccess,
}: {
  onMessage?: (message: WSMessage) => void;
  uri: string;
  onDisconnect?: () => void;
  onError?: (error: string) => void;
  onConnectionError?: (reason: string) => void;
  onConnectionSuccess?: () => void;
}) => {
  const lastMessageTime = useRef<null|number>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [socket, setSocket] = useState<WebSocket | null>(null);

  const sendMessage = useCallback(
    (message: WSMessage) => {
      if (!socket || !isConnected) {
        return;
      }
      // Avoid "WebSocket is already in CLOSING or CLOSED state" races.
      if (socket.readyState !== WebSocket.OPEN) {
        return;
      }
      socket.send(encodeMessage(message));
      // Outbound audio/text counts as activity; server may send nothing for a long time
      // while STT connects or the pipeline waits for a full PCM frame.
      lastMessageTime.current = Date.now();
    },
    [socket, isConnected],
  );

  const onConnect = useCallback(() => {
    console.log("connected, now waiting for handshake.");
    // setIsConnected(true);
  }, [setIsConnected]);

  const onDisconnect = useCallback((event: CloseEvent) => {
    console.log("disconnected", event.code, event.reason);
    if (onDisconnectProp) {
      onDisconnectProp();
    }
    setIsConnected(false);
  }, [onDisconnectProp]);

  const onMessageEvent = useCallback(
    (eventData: MessageEvent) => {
      lastMessageTime.current = Date.now();
      const dataArray = new Uint8Array(eventData.data);
      const message = decodeMessage(dataArray);
      if (message.type === "error") {
        if (onError) {
          onError(message.data);
        }
        return;
      }
      if (message.type == "handshake") {
        console.log("Handshake received, let's rocknroll.");
        setIsConnected(true);
        // Call success callback when handshake is received
        if (onConnectionSuccess) {
          onConnectionSuccess();
        }
      }
      if (onMessage) {
        onMessage(message);
      }
    },
    [onMessage, onError, setIsConnected, onConnectionSuccess],
  );

  const start = useCallback(() => {
    console.log("Attempting to connect to:", uri);
    const ws = new WebSocket(uri);
    ws.binaryType = "arraybuffer";

    // Handle connection errors before the connection is established
    ws.addEventListener("error", (event) => {
      console.error("WebSocket connection error:", event);
      // Check if the error is due to service unavailable (503)
      // Note: WebSocket errors don't give us HTTP status codes directly,
      // but we can infer from the close event if available
    });

    ws.addEventListener("open", onConnect);
    ws.addEventListener("close", (event) => {
      // Check if closed before handshake (likely service unavailable)
      if (!isConnected && event.code >= 1000) {
        console.warn("Connection closed before handshake:", event.code, event.reason);
        if (event.code === 1006 && onConnectionError) {
          onConnectionError(event.reason || "Connection failed");
        }
      }
      onDisconnect(event);
    });
    ws.addEventListener("message", onMessageEvent);
    setSocket(ws);
    console.log("Socket created", ws);
    lastMessageTime.current = Date.now();
  }, [uri, onMessage, onDisconnect, onConnectionError, isConnected]);

  const stop = useCallback(() => {
      setIsConnected(false);
      if (onDisconnectProp) {
        onDisconnectProp();
      }
      socket?.close();
      setSocket(null);
  }, [socket, onDisconnectProp]);

  useEffect(() => {
    if(!isConnected){
      return;
    }
    let intervalId = setInterval(() => {
      if (lastMessageTime.current && Date.now() - lastMessageTime.current > 10000) {
        console.log("closing socket due to inactivity", socket);
        socket?.close();
        clearInterval(intervalId);
      }
    }, 500);

    return () => {
      lastMessageTime.current = null;
      clearInterval(intervalId);
    };
  }, [isConnected, socket, onDisconnect]);

  return {
    isConnected,
    socket,
    sendMessage,
    start,
    stop,
  };
};