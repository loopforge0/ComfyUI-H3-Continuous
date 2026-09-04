"""Kate again, but cut for drift measurement rather than for the rant.

Six 4-second shots instead of six 8-second ones. That is the point: a 4 s segment
is the hardest case this pack has, because the same chain length buys you twice
the joins, and every join is a chance for the take to walk away from itself.

Everything shared between shots -- the subject definition, the style, the framing,
the delivery rules, the carry-in clause, the room tone -- is imported from
``kate_rant6_prompts`` rather than retyped. The README asks for those blocks to be
repeated *verbatim* across segments, and importing them is the only way to be sure
they are. What changes per shot is the line she speaks and the beat she ends on.

The script is deliberately short: 11-13 words a shot. A 107-frame segment holds
about 3.9 s of speech after the closing beat, and at her measured 3.1 words per
second that is 12 words. Overfill it and the model invents filler; underfill it and
it invents babble to cover the gap. Both were real failures on earlier builds.

Every shot still ends on a NON-SPEAKING beat, because the next segment replays the
last 0.92 s and a replayed half-word cannot be resumed cleanly.
"""

from kate_rant6_prompts import (
    CARRY,
    DELIVERY,
    FRAMING,
    KATE,
    RETAIN,
    ROOM,
    ROOM_CONT,
    STYLE,
    _prompt,
)

__all__ = ["PROMPTS", "SECONDS", "HANDOFF", "KATE", "RETAIN", "STYLE"]

# 4.0 s snaps up to 107 frames (4.46 s). The handoff is 22 frames (0.92 s), which is
# 21% of the segment -- the fifth the README argues for. The pack's 39-frame default
# would be 36% here, well past the quarter where seams measurably degrade, so a short
# segment MUST shorten its handoff to match.
SECONDS = [4.0] * 6
HANDOFF = 0.917

_SHOTS = [
    # (summary, opening clause, line, closing beat, sound)
    ("A young woman films herself in closeup on her phone and launches straight into "
     "an irritated rant about reality dating shows.",
     "She is already mid-thought as the shot opens, leaning slightly in toward the "
     "lens, and goes straight into the rant without any preamble, ",
     "So I am going to rant here, because these reality shows",
     "With the line finished she lets out a short audible breath through her nose, "
     "closes her mouth and holds a flat weary look straight down the lens as the "
     "shot ends.",
     ROOM),

    ("The same unbroken closeup continues: she names what the shows claim to be about.",
     "Coming straight out of the held breath, she picks the rant back up just after "
     "the one-second mark, ",
     "are supposedly about people trying to find true love",
     "On the words 'true love' she lifts both hands beside her face and bends her "
     "index and middle fingers into sarcastic air quotes, and she ends the shot with "
     "her mouth closed, her eyebrows high and both hands still held up beside her face.",
     ROOM_CONT),

    ("The same unbroken closeup continues: she moves on to the games the contestants play.",
     "Still holding the air quotes up beside her face from the replayed opening, she "
     "drops both hands just after the one-second mark and carries on, ",
     "and all these ridiculous social games you have to play",
     "She ends the shot with her mouth closed, one eyebrow raised and her head tilted "
     "slightly to one side.",
     ROOM_CONT),

    ("The same unbroken closeup continues: she compares the games to her manager's offsites.",
     "Carrying straight on out of the head tilt, she picks the line up again just "
     "after the one-second mark, faster and more irritated, ",
     "my manager plays the same games at her offsite every quarter",
     "In the beat after the line she taps the air once with one flat hand as if "
     "ticking the point off, then lets the hand fall, closes her mouth and holds a "
     "flat stare down the lens as the shot ends.",
     ROOM_CONT),

    ("The same unbroken closeup continues: she turns the comparison into an "
     "incredulous question.",
     "Carrying straight on out of the flat stare, she asks the question just after "
     "the one-second mark, ",
     "but I do not see her getting married. What is the point",
     "She lifts both hands palms-up into a wide baffled shrug in the beat after the "
     "question and holds them there, her eyebrows pushed right up and her mouth "
     "closed, as the shot ends.",
     ROOM_CONT),

    ("The same unbroken closeup continues: she lands the last complaint and stops.",
     "Still holding the palms-up shrug from the replayed opening, she drops her hands "
     "just after the one-second mark and finishes in a flat deadpan, ",
     "of any of it. Honestly, I am so tired of this",
     "With the line finished she pushes both hands up into her hair and rakes them "
     "back through it in one quick exasperated movement, then drops them, closes her "
     "mouth and holds a long flat stare straight down the lens as the shot ends.",
     ROOM_CONT),
]


def _build(index, summary, opening, line, beat, sound):
    # CARRY only belongs on a segment that actually has a replayed opening.
    carry = "" if index == 0 else CARRY + " "
    shot = "%s %s%s%s<d>[English] %s</d>. %s" % (
        FRAMING, carry, DELIVERY + " ", opening, line, beat)
    return _prompt(summary, shot, sound)


PROMPTS = [_build(i, *shot) for i, shot in enumerate(_SHOTS)]


if __name__ == "__main__":
    for i, (p, s) in enumerate(zip(PROMPTS, SECONDS)):
        words = p.split("<d>[English]")[1].split("</d>")[0].split()
        print("shot %d  %.2fs  %2d words  %.2f w/s  prompt %d chars"
              % (i + 1, s, len(words), len(words) / s, len(p)))
