import Studio from "@/components/Studio";
export default function Compare() {
  return (
    <>
      <section className="page-heading">
        <span className="eyebrow">THE COMPARISON LAB</span>
        <h1>
          Same opening.
          <br />
          <em>Different possibilities.</em>
        </h1>
        <p>
          Hold the prompt and sampling settings constant. Read the genuine
          continuations side by side. A lower validation score doesn’t guarantee
          a better story.
        </p>
      </section>
      <Studio compare />
    </>
  );
}
