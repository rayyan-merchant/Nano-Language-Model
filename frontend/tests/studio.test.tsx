/** Explicit API fixtures: these component tests do not qualify genuine inference. */
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi, it, expect, beforeEach } from "vitest";
import Studio from "@/components/Studio";
import { api, type ModelID } from "@/lib/api";
import evidence from "@/lib/research.json";
vi.mock("@/lib/api", () => ({
  api: { models: vi.fn(), generate: vi.fn(), compare: vi.fn() },
}));
const fixture = {
  model_id: "b01",
  model_name: "NanoLM Balanced",
  checkpoint_step: 3000,
  prompt: "ROMEO:\n",
  continuation: "fixture continuation",
  full_output: "ROMEO:\nfixture continuation",
  generated_tokens: 128,
  output_characters: 20,
  prompt_tokens: 4,
  effective_context: 4,
  context_cropped: false,
  settings: { greedy: false, temperature: 0.7, max_new_tokens: 128 },
  latency_seconds: 0.25,
  replacement_characters: false,
  seed: 42,
  request_id: "explicit-test-fixture",
};
beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.models).mockResolvedValue({
    models: evidence.models.map((m) => ({
      ...m,
      id: m.id as ModelID,
      availability: "ready",
    })),
  });
  vi.mocked(api.generate).mockResolvedValue(fixture as never);
});
it("selects the exact model and sends user settings", async () => {
  render(<Studio />);
  await screen.findByText("Local inference connected");
  fireEvent.click(screen.getByRole("button", { name: /Compact/ }));
  fireEvent.change(screen.getByLabelText("YOUR OPENING LINE"), {
    target: { value: "JULIET:\n" },
  });
  fireEvent.click(
    screen.getByRole("button", { name: /Generate continuation/ }),
  );
  await screen.findByText("fixture continuation", { exact: true });
  expect(document.querySelector(".generated")?.textContent).toBe(
    fixture.full_output,
  );
  expect(api.generate).toHaveBeenCalledWith(
    expect.objectContaining({
      prompt: "JULIET:\n",
      max_new_tokens: 128,
      seed: 42,
      temperature: 0.7,
    }),
    "e02",
    expect.any(AbortSignal),
  );
});
it("renders real response metadata and copies literal output", async () => {
  const writeText = vi.fn().mockResolvedValue(undefined);
  Object.defineProperty(navigator, "clipboard", {
    value: { writeText },
    configurable: true,
  });
  render(<Studio />);
  await screen.findByText("Local inference connected");
  fireEvent.click(
    screen.getByRole("button", { name: /Generate continuation/ }),
  );
  await screen.findByText("128 new tokens");
  fireEvent.click(screen.getByRole("button", { name: /Copy output/ }));
  await screen.findByText("Copied ✓");
  expect(writeText).toHaveBeenCalledWith(fixture.full_output);
});
it("shows errors and clear/reset works", async () => {
  vi.mocked(api.generate).mockRejectedValue(new Error("The worker is busy."));
  render(<Studio />);
  await screen.findByText("Local inference connected");
  fireEvent.click(
    screen.getByRole("button", { name: /Generate continuation/ }),
  );
  await screen.findByText("The worker is busy.");
  fireEvent.click(screen.getByRole("button", { name: "Clear" }));
  expect(screen.getByLabelText("YOUR OPENING LINE")).toHaveValue("");
});
it("disables inference while offline and keeps metadata", async () => {
  vi.mocked(api.models).mockRejectedValue(new Error("offline"));
  render(<Studio />);
  await screen.findByText("Backend unavailable");
  expect(
    screen.getByRole("button", { name: /Generate continuation/ }),
  ).toBeDisabled();
  expect(screen.getByText("Balanced")).toBeVisible();
});
it("comparison renders all slots including an unavailable model", async () => {
  vi.mocked(api.compare).mockResolvedValue({
    request_id: "fixture",
    seed: 42,
    total_latency_seconds: 0.5,
    results: [
      {
        model_id: "e02",
        status: "unavailable",
        error: "Missing fixture artifact",
      },
      { model_id: "b01", status: "completed", result: fixture as never },
      {
        model_id: "f01",
        status: "completed",
        result: { ...fixture, model_id: "f01" } as never,
      },
    ],
  });
  render(<Studio compare />);
  await screen.findByText("Local inference connected");
  fireEvent.click(screen.getByRole("button", { name: /Compare all three/ }));
  await screen.findByText("Missing fixture artifact");
  expect(screen.getAllByText("128 new tokens")).toHaveLength(2);
});
it("cancel aborts and reports its honest server limitation", async () => {
  vi.mocked(api.generate).mockImplementation(() => new Promise(() => {}));
  render(<Studio />);
  await screen.findByText("Local inference connected");
  fireEvent.click(
    screen.getByRole("button", { name: /Generate continuation/ }),
  );
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  await screen.findByText(/bounded server worker may still be finishing/);
});
it("changes decoding, temperature, token limit and seed and shows loading", async () => {
  vi.mocked(api.generate).mockImplementation(() => new Promise(() => {}));
  render(<Studio />);
  await screen.findByText("Local inference connected");
  fireEvent.change(screen.getByLabelText("Decoding"), { target: { value: "greedy" } });
  expect(screen.getByLabelText("Temperature")).toBeDisabled();
  fireEvent.change(screen.getByLabelText("Decoding"), { target: { value: "sample" } });
  fireEvent.change(screen.getByLabelText("Temperature"), { target: { value: "1.2" } });
  fireEvent.change(screen.getByLabelText("New tokens"), { target: { value: "64" } });
  fireEvent.change(screen.getByLabelText("Seed"), { target: { value: "123" } });
  const button = screen.getByRole("button", { name: /Generate continuation/ });
  button.focus();
  expect(button).toHaveFocus();
  fireEvent.click(button);
  await screen.findByText("Finding the next word.");
  expect(api.generate).toHaveBeenCalledWith(expect.objectContaining({ greedy: false, temperature: 1.2, max_new_tokens: 64, seed: 123 }), "b01", expect.any(AbortSignal));
  expect(screen.getByRole("button", { name: "Generating…" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
});
