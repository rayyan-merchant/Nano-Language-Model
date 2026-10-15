import Link from "next/link";
import Studio from "@/components/Studio";
export default function Home() {
  return (
    <>
      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">
            <span className="gold">●</span> AN EXPERIMENT IN LANGUAGE
          </span>
          <h1>
            Small models.
            <br />
            <em>Shakespearean</em>
            <br />
            possibilities.
          </h1>
          <p>
            Three Transformers. Built from scratch.
            <br />A little computation, a world of unfinished stories.
          </p>
          <div className="hero-actions">
            <a className="primary" href="#studio">
              Open the playground <span>↗</span>
            </a>
            <Link href="/research" className="secondary">
              Explore the research →
            </Link>
          </div>
          <div className="hero-caption">
            <span>16 RESEARCH CONFIGURATIONS</span>
            <span>100% OUR OWN WEIGHTS</span>
          </div>
        </div>
        <div
          className="hero-art"
          aria-label="Illustration of a token passing through attention blocks"
        >
          <div className="art-top">
            <span>THE NEXT TOKEN IS AN OPEN QUESTION</span>
            <span>FIG. 01</span>
          </div>
          <div className="token-trace">
            <span>To</span>
            <span>be</span>
            <span>,</span>
            <span>or</span>
            <span>not</span>
            <span>to</span>
            <span>be</span>
            <span className="token-unknown">?</span>
          </div>
          <svg
            viewBox="0 0 440 220"
            className="attention-art"
            aria-hidden="true"
          >
            <g fill="none" stroke="currentColor" strokeWidth="1">
              {[0, 1, 2, 3, 4, 5, 6].map((i) => (
                <g key={i}>
                  <path
                    d={`M${40 + i * 56},10 C${40 + i * 56},100 220,60 220,190`}
                  />
                  <path
                    d={`M${40 + i * 56},10 C${40 + i * 56},100 110,60 110,150`}
                  />
                </g>
              ))}
            </g>
            <circle cx="220" cy="190" r="7" fill="currentColor" />
            <circle cx="110" cy="150" r="5" fill="currentColor" />
          </svg>
          <p className="art-quote">
            “What’s past
            <br />
            is <em>prologue.</em>”
          </p>
          <div className="art-bottom">
            <span>ATTENTION → POSSIBILITY</span>
            <span>TRAINED ON TINY SHAKESPEARE</span>
          </div>
        </div>
      </section>
      <div className="intro-rule">
        <span>NOT A GENERAL-PURPOSE ASSISTANT.</span>
        <p>A transparent window into what small language models learn.</p>
        <Link href="/research#limits">Know the limitations ↗</Link>
      </div>
      <Studio />
      <section className="research-teaser">
        <span className="eyebrow">BEYOND THE NEXT WORD</span>
        <h2>
          Small enough to understand.
          <br />
          <em>Interesting enough to explore.</em>
        </h2>
        <p>
          Every attention head, every layer, every result is open to inspection.
          Explore what changes when a model becomes deeper, wider, or learns to
          see positions differently.
        </p>
        <Link className="secondary" href="/research">
          Inside the research →
        </Link>
      </section>
    </>
  );
}
