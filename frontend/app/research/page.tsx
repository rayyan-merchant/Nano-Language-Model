import evidence from "@/lib/research.json";
import Link from "next/link";
export default function Research() {
  return (
    <>
      <section className="page-heading">
        <span className="eyebrow">THE RESEARCH NOTEBOOK / OPEN BY DESIGN</span>
        <h1>
          A language model.
          <br />
          <em>Nothing hidden.</em>
        </h1>
        <p>
          From explicit attention to independently recovered checkpoints. A
          study of the trade-offs between capacity, computation and next-token
          prediction.
        </p>
      </section>
      <section className="research-section">
        <div className="section-top">
          <span className="eyebrow">01 / WHAT’S INSIDE</span>
          <h2>A Transformer, from first principles.</h2>
        </div>
        <div className="architecture">
          <span>Text</span>
          <b>→</b>
          <span>
            Byte-BPE
            <br />
            <small>1,024 tokens</small>
          </span>
          <b>→</b>
          <span>
            Causal attention
            <br />
            <small>Q · K · V</small>
          </span>
          <b>→</b>
          <span>
            GELU MLP
            <br />
            <small>Pre-LayerNorm blocks</small>
          </span>
          <b>→</b>
          <span>Next token</span>
        </div>
        <div className="explain-grid">
          <article>
            <span className="index">01</span>
            <h3>Look back. Never ahead.</h3>
            <p>
              Queries, keys and values are projected explicitly. Scaled dot
              products and softmax determine which past tokens matter. A causal
              mask prevents the model from seeing future text.
            </p>
          </article>
          <article>
            <span className="index">02</span>
            <h3>A shared alphabet of bytes.</h3>
            <p>
              The same train-only Byte-BPE tokenizer is used by all three
              models. A token can contain part of a word or UTF-8 character.
              Whole-sequence decoding preserves canonical output, including
              replacement characters.
            </p>
          </article>
          <article>
            <span className="index">03</span>
            <h3>Position changes perspective.</h3>
            <p>
              E02 adds fixed sinusoidal positions. B01 and F01 rotate queries
              and keys with RoPE. Each model sees at most its latest 128 tokens
              during generation.
            </p>
          </article>
        </div>
      </section>
      <section className="research-section results">
        <div className="section-top">
          <span className="eyebrow">02 / RECORDED EVIDENCE</span>
          <h2>Capacity has a cost.</h2>
        </div>
        <p className="lead">
          Fourteen single-factor seed-42 ablations informed two separately
          approved combined architectures. These three selected checkpoints
          share a tokenizer and score the same validation bytes.
        </p>
        <div className="charts">
          <div>
            <h3>Trainable parameters</h3>
            <p className="axis-label">
              Millions of parameters · longer bar = larger model
            </p>
            {evidence.models.map((m) => (
              <div className="bar-row" key={m.id}>
                <span>{m.name.replace("NanoLM ", "")}</span>
                <div className="bar-track">
                  <div
                    className={`bar ${m.id}`}
                    style={{ width: `${(m.parameters / 6500000) * 100}%` }}
                  />
                </div>
                <b>{(m.parameters / 1e6).toFixed(2)}M</b>
              </div>
            ))}
            <div className="axis">
              <span>0</span>
              <span>3.25M</span>
              <span>6.50M</span>
            </div>
          </div>
          <div>
            <h3>Selected validation BPB</h3>
            <p className="axis-label">
              Bits per original byte · lower = better predictive likelihood
            </p>
            {evidence.models.map((m) => (
              <div className="score-row" key={m.id}>
                <span>{m.name.replace("NanoLM ", "")}</span>
                <strong>{m.validation_bpb.toFixed(5)}</strong>
                <small>
                  Selected at update {m.checkpoint_step.toLocaleString()}
                </small>
              </div>
            ))}
            <p className="note">
              B01 is nearly F01’s selected likelihood with about half the
              parameters. This does not establish equal generation quality.
            </p>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <caption>
              Exact selected architectures and best-validation checkpoints
            </caption>
            <thead>
              <tr>
                <th>Model</th>
                <th>Width</th>
                <th>Blocks</th>
                <th>Position</th>
                <th>Selected update</th>
                <th>BPB</th>
              </tr>
            </thead>
            <tbody>
              {evidence.models.map((m) => (
                <tr key={m.id}>
                  <th>
                    {m.id.toUpperCase()} · {m.name.replace("NanoLM ", "")}
                  </th>
                  <td>{m.architecture.d_model}</td>
                  <td>{m.architecture.n_layers}</td>
                  <td>
                    {m.architecture.position === "rope" ? "RoPE" : "Sinusoidal"}
                  </td>
                  <td>{m.checkpoint_step.toLocaleString()}</td>
                  <td>{m.validation_bpb.toFixed(8)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="note">
          Source: accepted E02, B01 and F01 research manifests and reports.
          Conditional validation excludes 256 conditioning-only bytes and scores
          55,514 original bytes.
        </p>
      </section>
      <section className="research-section limitations" id="limits">
        <span className="eyebrow">03 / A NOTE ON EXPECTATIONS</span>
        <h2>Curiosity, with context.</h2>
        <div className="explain-grid">
          <article>
            <h3>A tiny literary world.</h3>
            <p>
              These models learned only from Tiny Shakespeare. They are not
              instruction-tuned, fact-checking assistants or general-purpose
              LLMs. They can form dialogue without understanding its meaning.
            </p>
          </article>
          <article>
            <h3>Lower loss isn’t better prose.</h3>
            <p>
              Greedy decoding can loop. Sampling can produce malformed words,
              broken grammar and invalid byte sequences. F01’s validation
              deteriorated with prolonged training; its default is update 1,750,
              not 10,000.
            </p>
          </article>
          <article>
            <h3>One seed is one observation.</h3>
            <p>
              The multi-seed expansion was discontinued. Width, depth,
              initialization and compute change together. No statistical
              significance or general superiority is claimed.
            </p>
          </article>
        </div>
        <a
          className="secondary"
          href="https://github.com/rayyan-merchant/Nano-Language-Model/tree/main/report"
          target="_blank"
          rel="noreferrer"
        >
          Read the complete research records ↗
        </a>
        <Link className="primary" href="/#studio">
          Try the models →
        </Link>
      </section>
    </>
  );
}
