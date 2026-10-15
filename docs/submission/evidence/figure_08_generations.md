# Figure 8 — Predetermined genuine examples

All five B01 greedy prompts in receipt order, without filtering or rewriting. Selected step 3000, 32 new BPE tokens, seed 42, temperature 0.7 (ignored by greedy), context 128. Final submission CPU demonstration, not the original 256-token research grid.

## Prompt `'To be, or not to be'`

```text
To be, or not to be a brid.

PARIS:
Anon, I am a man, and thou hast a bawd.

ROMEO
```

## Prompt `'ROMEO:\n'`

```text
ROMEO:
I'll tell thee, I'll not be a king,
And nothing but a man.

JULIET:
I know not, sir, if
```

## Prompt `'JULIET:\n'`

```text
JULIET:
I'll not be a king, and I'll not be a king.

JULIET:
I know not, sir, I'll not be a man
```

## Prompt `'First Citizen:\n'`

```text
First Citizen:
The gods do not speak of the city,
And nothing but one.

First Senator:
Alas,
```

## Prompt `'The king'`

```text
The king of York,
And, by the king, and the king of York,
And, by the king, and the king of York,
```

Source: [unaltered generation receipt](../verification/colab-final/receipts/generation/receipt.json). Repeated phrases and incomplete endings are preserved; 32-token truncation limits coherence judgments.
