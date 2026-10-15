/** Public contract mirrors FastAPI schemas; requests are centralized here. */
export type ModelID = "e02" | "b01" | "f01";
export type Model = {
  id: ModelID;
  name: string;
  description: string;
  parameters: number;
  checkpoint_step: number;
  validation_bpb: number;
  architecture: {
    d_model: number;
    n_layers: number;
    n_heads: number;
    position: string;
    context_length: number;
  };
  availability: "ready" | "not_loaded" | "unavailable";
  limitations: string;
};
export type Settings = {
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  seed: number | null;
  greedy: boolean;
};
export type Generation = {
  model_id: ModelID;
  model_name: string;
  checkpoint_step: number;
  prompt: string;
  continuation: string;
  full_output: string;
  generated_tokens: number;
  output_characters: number;
  prompt_tokens: number;
  effective_context: number;
  context_cropped: boolean;
  settings: Record<string, number | boolean>;
  latency_seconds: number;
  replacement_characters: boolean;
  seed: number;
  request_id: string;
};
export type Comparison = {
  request_id: string;
  seed: number;
  total_latency_seconds: number;
  results: {
    model_id: ModelID;
    status: "completed" | "unavailable" | "failed";
    result?: Generation | null;
    error?: string | null;
  }[];
};
const origin = process.env.NEXT_PUBLIC_API_ORIGIN || "http://127.0.0.1:8000";
async function request<T>(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(origin + path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    signal,
    cache: "no-store",
  });
  const value = await response.json();
  if (!response.ok)
    throw new Error(
      value.error?.message || "The backend could not complete this request.",
    );
  return value;
}
export const api = {
  models: (signal?: AbortSignal) =>
    request<{ models: Model[] }>("/v1/models", undefined, signal),
  generate: (settings: Settings, model_id: ModelID, signal?: AbortSignal) =>
    request<Generation>("/v1/generate", { ...settings, model_id }, signal),
  compare: (settings: Settings, signal?: AbortSignal) =>
    request<Comparison>("/v1/compare", settings, signal),
};
