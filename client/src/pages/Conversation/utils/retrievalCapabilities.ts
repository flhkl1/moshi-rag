import { z } from "zod";

/** Rust sends full `MetaData` JSON; Python sends only these fields — accept both. */
const schema = z
  .object({
    retrieval_backends: z.array(z.object({ id: z.string() })),
    retrieval_backend_default: z.string().optional().nullable(),
  })
  .passthrough();

export type RetrievalBackendOption = { id: string };

export type ParsedRetrievalCapabilities = {
  backends: RetrievalBackendOption[];
  defaultId: string;
};

export function parseRetrievalCapabilities(data: unknown): ParsedRetrievalCapabilities | null {
  const r = schema.safeParse(data);
  if (!r.success) {
    return null;
  }
  const backends = r.data.retrieval_backends.map((b) => ({ id: b.id }));
  if (backends.length < 2) {
    return null;
  }
  const rawDefault = r.data.retrieval_backend_default;
  const defaultId =
    typeof rawDefault === "string" && rawDefault.trim() !== ""
      ? rawDefault.trim()
      : backends[0]!.id;
  return { backends, defaultId };
}
