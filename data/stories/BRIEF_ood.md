# Out-of-distribution emotion test set: writing brief

Purpose: an INDEPENDENT test of the emotion vectors. The vectors were built from long
third-person stories written by another model. If a vector truly encodes an emotion (and not
the vocabulary or scenarios of its training stories), it must also pick out that emotion in
short texts of a completely different form, written by a different writer. That is you.

## What to write

For each of the 19 emotions below, 12 short texts (228 in total):

afraid, angry, anxious, calm, desperate, enthusiastic, exasperated, fearful, gloomy, grateful,
guilty, happy, hopeful, hostile, loving, nervous, proud, reflective, sad

Definitions that keep near-synonyms apart: afraid = sudden fright now; fearful = dread of what
may come; anxious = diffuse worry about outcomes; nervous = jitters before a specific moment;
angry = a flash of anger at a specific wrong; hostile = sustained antagonism towards a person;
exasperated = worn-out irritation at something repeated; gloomy = heavy, dark outlook;
sad = sorrow over a loss; reflective = quiet thinking back over something.

Each text:
- 25 to 70 words, FIRST PERSON or a spoken line or a short message, present tense mostly.
- Rotate the form across each emotion's 12 items: 4 inner monologue, 3 spoken to someone,
  3 text message or email, 2 diary line.
- Never use the target emotion's own word family (no "angry", "anger", "angrily" for angry;
  no "fear", "feared", "fearful" for fearful; no "hope", "hoping", "hopeful" for hopeful;
  no "love", "loved", "loving" for loving; no "calm" for calm; no "pride", "proud" for proud;
  no "reflect", "reflection" for reflective; and so on). Show it, do not name it.
- Everyday topics that are DIFFERENT from typical story plots: use work chat, sport, cooking,
  weather, studying, gaming, gardening, pets, commuting, phone problems, family calls.
- Benign only: no violence, weapons, self-harm, crime, drugs, sexual content, illness or death.
- No em or en dashes.

## Output

`data/stories/ood/validation.jsonl`, one JSON object per line:
```json
{"id": "ood-angry-01", "emotion": "angry", "form": "monologue | spoken | message | diary", "text": "..."}
```

Check before finishing:
```bash
cd /Users/anuj/Agents/PersonalProjects/affect-as-attack/repo
.venv/bin/python scripts/validate_ood.py
```
Fix every FAIL until it prints RESULT: PASS.
