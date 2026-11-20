# Round 2 feedback: start the emotion early

Your files pass the round-1 checks, but measuring the corpus showed one systematic problem in
seven emotions: desperate, enthusiastic, exasperated, fearful, proud, reflective, sad.

In these files the emotional story and its neutral twin share about 78 percent of their opening
word for word; the emotion only appears in the last one or two sentences. The other twelve
emotions (written by other writers) share only 8 to 32 percent. Because the vector is the mean
over the WHOLE story, your stories contribute a much weaker and differently placed emotion
signal, which would fake differences between emotions in the experiment.

What to change, for every pair in your files:
- The emotion must be felt from the first or second sentence and run through the whole story
  (thoughts, body sensations, actions, dialogue), as in the twelve other emotions. Example of the
  target style: the emotional story opens with the character's state, then the events unfold
  through it.
- Keep the neutral twin a matched, flat account of the same character, place and events.
  Twins may share some sentences, but the first half of the two stories must NOT be identical.
- Keep everything else from the brief (length 90 to 140 words, the banned words, benign content,
  no dashes, same ids, setting and character fields). You may keep each pair's scenario.

The validator now enforces this: no pair may share more than 50 percent of its opening, and a
file's median shared opening must be at most 35 percent. Run it until RESULT: PASS.
