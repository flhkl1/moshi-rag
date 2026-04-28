import { createContext, useContext } from "react";
import { WSMessage } from "../../protocol/types";
import type { ParsedRetrievalCapabilities } from "./utils/retrievalCapabilities";

type SocketContextType = {
  isConnected: boolean;
  socket: WebSocket | null;
  sendMessage: (message: WSMessage) => void;
  /** From server metadata when ≥2 retrieval LLMs configured; avoids missing early WS frames before child effects attach. */
  retrievalCapabilities: ParsedRetrievalCapabilities | null;
};

export const SocketContext = createContext<SocketContextType>({
  isConnected: false,
  socket: null,
  sendMessage: () => {},
  retrievalCapabilities: null,
});

export const useSocketContext = () => {
  return useContext(SocketContext);
};
