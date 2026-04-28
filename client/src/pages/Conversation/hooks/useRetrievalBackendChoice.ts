import { useCallback, useEffect, useState } from "react";
import { useSocketContext } from "../SocketContext";

export function useRetrievalBackendChoice() {
  const { sendMessage, retrievalCapabilities: caps } = useSocketContext();
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    if (!caps) {
      setSelectedId(null);
      return;
    }
    setSelectedId((prev) =>
      prev !== null && caps.backends.some((b) => b.id === prev) ? prev : caps.defaultId,
    );
  }, [caps]);

  const sendChoiceToServer = useCallback(
    (id: string) => {
      sendMessage({
        type: "metadata",
        data: { retrieval_backend_id: id },
      });
    },
    [sendMessage],
  );

  const setSelectedAndNotify = useCallback(
    (id: string) => {
      setSelectedId(id);
      sendChoiceToServer(id);
    },
    [sendChoiceToServer],
  );

  // Avoid a one-frame gap where caps exist but `useEffect` has not yet synced `selectedId`
  // (SearchPanel required `selectedRetrievalId != null` and hid tabs).
  const effectiveSelectedId = selectedId ?? caps?.defaultId ?? null;

  return {
    retrievalBackends: caps?.backends ?? [],
    selectedRetrievalId: effectiveSelectedId,
    setSelectedRetrievalId: setSelectedAndNotify,
  };
}
