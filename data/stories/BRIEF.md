# Emotion story corpus: writing brief

This corpus is the input to **Module 1 (emotion vector extraction)** of the Affect as Attack project.
For each emotion we read paired stories through Llama 3.1 8B Instruct, take the mean hidden state
over the story tokens, and subtract the neutral twin from the emotional story. The average of
those differences is the emotion vector. So the ONLY systematic difference between the two
stories in a pair must be the emotion. Everything else (character, place, events, length,
sentence style) should match as closely as possible.

## The 20 categories

19 emotions, each written as 60 PAIRS (emotional story + neutral twin):

afraid, angry, anxious, calm, desperate, enthusiastic, exasperated, fearful, gloomy, grateful,
guilty, happy, hopeful, hostile, loving, nervous, proud, reflective, sad

Plus `neutral`: 60 STANDALONE emotionally flat stories (no twins).

`afraid` and `fearful` are near-synonyms on purpose (the proposal's vocabulary keeps both).
Write `afraid` as a sudden, situational fright (a noise, a near miss, a strange dog) and
`fearful` as a slower, anticipatory dread about something that may happen. Likewise keep
`anxious` (diffuse worry about outcomes) apart from `nervous` (jittery, before a specific
moment such as a speech or an interview), and `angry` (a flash of anger at a specific wrong)
apart from `hostile` (sustained antagonism towards a person) and `exasperated` (worn-out
irritation at something repeated).

## Each emotional story

- 90 to 140 words, past tense, third person, one named main character.
- A concrete everyday scenario. The character clearly experiences the target emotion, shown
  through thoughts, body sensations, actions and dialogue.
- **Never use the target emotion's own word family** (for example no "angry", "anger",
  "angrily" in an `angry` story). The validator lists the banned forms. Convey the emotion,
  do not name it. Other emotion words are allowed but keep the target emotion dominant.
- Benign content only: no violence beyond mild (a slammed door is fine), no weapons, no
  self-harm, no crime, no drugs, no sexual content, no medical emergencies, no deaths. Hostile
  and angry stories are about interpersonal friction, not harm. This matters: if the stories
  touched harmful topics, the vector would mix "emotion" with "harm" and spoil the experiment.

## Each neutral twin

- Same character name, same setting, the same concrete events in the same order wherever the
  emotion is not the event itself, and a length within 80 to 125 percent of its emotional story.
- Emotionally flat: plain factual narration of what happens. No emotion words at all, no
  evaluative language ("lovely", "awful"), no smiling, laughing, crying, sighing, trembling or
  racing hearts. The validator has the full banned list.
- It must not read like a deleted-scene summary. It is a complete, ordinary little story.

## Neutral category (standalone)

60 stories, 90 to 140 words, flat affect, everyday events, varied characters and settings,
same banned emotion lexicon as the neutral twins.

## Diversity (checked by the validator, read by a human)

- 60 different core scenarios per emotion. No two pairs in one emotion share the same situation.
- Spread settings: home, work, school, university, shops and markets, transport, travel, sport,
  hospitals and clinics (non-emergency), farms and villages, cities, online life, hobbies,
  family events, weather and nature.
- Spread characters: ages from children to elderly, genders, and names from many cultures
  (South Asian, East Asian, African, European, Latin American, Middle Eastern, and more).
- Vary sentence openings. Do not start more than three stories in a file with the same four words.
- Do not reuse sentences or stock phrases across stories. Each story is written fresh.
  **Do not generate stories with code or templates.** Write each one as a writer would.
- No em dashes or en dashes anywhere (use commas, full stops or parentheses).

## Output format

One JSONL file per category in `data/stories/raw/`, UTF-8, one JSON object per line.

Emotion files (`<emotion>.jsonl`, 60 lines):

```json
{"id": "angry-01", "emotion": "angry", "setting": "short label, e.g. bus depot", "character": "Asha", "emotional": "...", "neutral": "..."}
```

Neutral file (`neutral.jsonl`, 60 lines):

```json
{"id": "neutral-01", "emotion": "neutral", "setting": "...", "character": "...", "text": "..."}
```

Ids run 01 to 60 with two digits.

## Self-check before you finish

```bash
cd /Users/anuj/Agents/PersonalProjects/affect-as-attack/repo
.venv/bin/python scripts/validate_stories.py data/stories/raw/<emotion>.jsonl
```

Fix every FAIL line and re-run until the file prints `RESULT: PASS`. Then read five random
pairs yourself and ask: would a careful reader feel the emotion in the first story and nothing
in the second? If not, rewrite them.
