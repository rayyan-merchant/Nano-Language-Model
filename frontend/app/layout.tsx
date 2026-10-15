import type { Metadata } from "next";
import Link from "next/link";
import "@fontsource-variable/cormorant-garamond";
import "@fontsource-variable/cormorant-garamond/wght-italic.css";
import "@fontsource-variable/manrope";
import "./globals.css";
export const metadata: Metadata = {
  title: "NanoLM Studio — Small models, open possibilities",
  description:
    "A writing studio for three from-scratch Shakespeare Transformer models. Genuine local inference and transparent research.",
};
export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <a className="skip" href="#main">
          Skip to content
        </a>
        <header className="nav">
          <Link className="brand" href="/" aria-label="NanoLM Studio home">
            <span className="mark">n.</span>NanoLM <b>Studio</b>
          </Link>
          <nav aria-label="Main navigation">
            <Link href="/#studio">Playground</Link>
            <Link href="/compare">Compare</Link>
            <Link href="/research">
              Research <span aria-hidden="true">↗</span>
            </Link>
          </nav>
          <span className="nav-note">BUILT FROM SCRATCH</span>
        </header>
        <main id="main" tabIndex={-1}>{children}</main>
        <footer>
          <Link className="brand" href="/">
            NanoLM Studio<span className="gold">.</span>
          </Link>
          <p>A small language model. An open research question.</p>
          <a
            href="https://github.com/rayyan-merchant/Nano-Language-Model"
            target="_blank"
            rel="noreferrer"
          >
            Source & methodology ↗
          </a>
          <span>Single-seed research · No external LLM</span>
        </footer>
      </body>
    </html>
  );
}
