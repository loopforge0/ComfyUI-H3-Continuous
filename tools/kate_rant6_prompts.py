"""Kate's rant, six segments.

Why six and not seven
---------------------
The seven-segment build under-filled its shots. Kate speaks at a measured 3.1 words
per second (segment 1 of that build: 18 words over 5.81 s of speech-active audio), but
segments 2-4 were given only 1.25 / 1.41 / 1.88 words per second of runtime. That left
each of them three-and-a-half seconds with nothing scripted in it -- and because every
prompt also demanded she talk "continuously", the model filled the gap with
speech-shaped babble. That is where "watch the derm strame durame" and "piding the
check" came from: invented filler, not a rendering artifact.

So the same 104 words, verbatim, are redistributed across six shots at 2.67-2.83 words
per second of runtime, which is 3.1 w/s of speech plus a 0.4-0.95 s closing beat. There
is no unscripted time left for the model to fill.

Two further defences against the same failure:
  * DELIVERY no longer says "continuously". It says she speaks the given line and adds
    no other words -- an explicit prohibition, not just an absence of instruction.
  * Every shot still ends on a NON-SPEAKING beat, because the next segment replays the
    last 1.625 s and a replayed half-word cannot be resumed cleanly.

Segment lengths are not uniform
-------------------------------
A flat 6 x 192 cannot hold this text: the closing sentence breaks into two ten-word
clauses that will not recombine under a 192-frame cap, which forces the earlier shots
past their own capacity. Sizing each shot to its own line fixes it. Five of the six are
at or below the proven 192 frames; the two 209s match segment 6 of the previous run,
which rendered without trouble on the same card.
"""

KATE = (
    "the young woman in <Picture 1>, in her early twenties, with shoulder-length wavy "
    "mid-brown hair worn loose with soft lilac-purple tones running through the "
    "mid-lengths and ends, fair clear skin, blue-grey eyes, dark natural brows and a "
    "slim build, wearing a chunky oatmeal-and-brown waffle-knit crew-neck sweater. Only "
    "her face, hair, colouring and sweater are taken from <Picture 1>; the calm closed-"
    "mouth half-smile and the still portrait pose in that image are not retained, because "
    "she is talking rapidly and animatedly throughout the target video."
)

STYLE = (
    "The target video is photorealistic live-action shot vertically for social media, "
    "filmed on a phone front camera held at arm's length just above eye level in a warm "
    "book-lined room, in soft daylight from a window to her left, with a shallow depth of "
    "field so the wooden bookshelves behind her are softly blurred, and with no on-screen "
    "text and no subtitles anywhere."
)

FRAMING = (
    "The camera is handheld at arm's length and holds the same tight head-and-shoulders "
    "closeup for the whole segment with no cut, the top of her head near the top of frame "
    "and the bottom of frame at her collarbones, drifting only very slightly the way a "
    "hand-held phone does and never zooming, cutting or changing its distance."
)

# The anti-babble clause is the last sentence. The seven-segment build said "talks fast,
# continuously" and gave the model four unscripted seconds to honour it; it obliged by
# inventing words. Here the script is dense enough that there is no gap, and the prompt
# forbids filling one anyway.
DELIVERY = (
    "She looks straight down the lens and talks fast, at a quick ranting pace, with "
    "mounting exasperation, in a bright, expressive young female voice, her jaw and lips "
    "opening and closing distinctly on every syllable so every word is clearly formed, "
    "never settling into a fixed open smile or a laugh that would freeze her mouth. Her "
    "eyebrows, eyes and the set of her mouth carry the irritation, and each hand movement "
    "lands in the gap between phrases rather than underneath a line. She speaks exactly "
    "the words given below and no others: she does not ad-lib, does not repeat herself "
    "and does not add any further dialogue to fill out the shot."
)

CARRY = (
    "The segment opens on an exact replay of the closing moments of the preceding take and "
    "carries that motion straight through, so the join is invisible. She holds the exact "
    "body pose, head angle, gaze direction, hand position and facial expression carried in "
    "by the replayed opening and continues out of it without any reset or restart."
)

RETAIN = (
    "<Subject 1> (appears in [Shot 1]): fully_preserved - her face, blue-grey eyes, "
    "shoulder-length wavy mid-brown hair with its lilac-purple tones and her chunky "
    "oatmeal waffle-knit sweater are retained exactly; the calm still portrait pose of "
    "<Picture 1> is not retained."
)

ROOM = ("Close indoor room tone in a soft furnished room, her voice loud and close on the "
        "phone microphone, with faint clothing rustle as she gestures and no other sound.")
ROOM_CONT = ("The close indoor room tone carries straight on from the preceding segment "
             "without interruption: her voice loud and close on the phone microphone, with "
             "faint clothing rustle as she gestures and no other sound.")


def _prompt(summary, shot, sound):
    return """subject_definitions:
<Subject 1> is %s

summary:
%s

retention_analysis:
%s

detailed_description:
%s
[Shot 1] %s

overall_soundscape:
%s

non_diegetic_music:
N/A""" % (KATE, summary, RETAIN, STYLE, shot, sound)


# --- segment 1 : 158 frames (6.58 s), 18 words -------------------------------------
P1 = _prompt(
    "A young woman films herself in closeup on her phone and launches straight into an "
    "irritated rant about reality dating shows.",
    FRAMING + " " + DELIVERY + " She is already mid-thought as the shot opens, leaning "
    "slightly in toward the lens, and goes straight into the rant without any preamble, "
    "<d>[English] So I'm going to rant a little here, but I'm just soooo tired of these "
    "so-called reality shows</d>. She stretches the word 'soooo' out with her eyes closing "
    "and her head tipping back for a moment, and on the words 'so-called reality shows' "
    "her mouth pulls to one side in distaste. With the line finished she lets out a short "
    "audible breath through her nose, closes her mouth and holds a flat weary look "
    "straight down the lens as the shot ends.",
    ROOM)

# --- segment 2 : 192 frames (8.00 s), 17 words -------------------------------------
# The air quotes are the one gesture allowed to run under the line: they are meaningless
# anywhere else. They stay up at the end so segment 3 inherits a held, readable pose.
P2 = _prompt(
    "The same unbroken closeup continues: she carries the rant on to the contestants and "
    "the games they play.",
    FRAMING + " " + CARRY + " " + DELIVERY + " Coming straight out of the held breath, she "
    "picks the rant back up just after the one-and-three-quarter-second mark, "
    "<d>[English] where people are trying to find true love. And all these social games "
    "you have to play,</d>. On the words 'true love' she lifts both hands beside her face "
    "and bends her index and middle fingers into sarcastic air quotes, and she keeps them "
    "raised through the rest of the line. She ends the shot with her mouth closed, her "
    "eyebrows high and both hands still held up in the air quotes beside her face.",
    ROOM_CONT)

# --- segment 3 : 209 frames (8.71 s), 19 words -------------------------------------
P3 = _prompt(
    "The same unbroken closeup continues: she compares the show's games to her manager's "
    "work offsites.",
    FRAMING + " " + CARRY + " " + DELIVERY + " Still holding the air quotes up beside her "
    "face from the replayed opening, she drops both hands just after the "
    "one-and-three-quarter-second mark and carries on, faster and more irritated, "
    "<d>[English] my manager plays the same games during her offsite every 3 months, but "
    "I don't see her getting married</d>. In the small break after the word 'months' she "
    "taps the air once with one flat hand as if ticking the point off. She ends the shot "
    "with her mouth closed, one eyebrow raised and her head tilted slightly to one side.",
    ROOM_CONT)

# --- segment 4 : 158 frames (6.58 s), 14 words -------------------------------------
P4 = _prompt(
    "The same unbroken closeup continues: she finishes the comparison and turns it into an "
    "incredulous question.",
    FRAMING + " " + CARRY + " " + DELIVERY + " Carrying straight on out of the head tilt, "
    "she finishes the thought just after the one-and-three-quarter-second mark, "
    "<d>[English] and running off with her co-workers. What do you even put on your "
    "resume?</d>. She lifts both hands palms-up into a wide baffled shrug in the beat "
    "after the question and holds them there, her eyebrows pushed right up and her mouth "
    "closed, as the shot ends.",
    ROOM_CONT)

# --- segment 5 : 175 frames (7.29 s), 16 words -------------------------------------
P5 = _prompt(
    "The same unbroken closeup continues: she answers her own question by listing absurd "
    "resume lines.",
    FRAMING + " " + CARRY + " " + DELIVERY + " Still holding the palms-up shrug from the "
    "replayed opening, she drops her hands just after the one-and-three-quarter-second "
    "mark and answers her own question in a flat deadpan, <d>[English] left 3 people at "
    "the altar and was a runner-up bride? And... was it a requirement</d>. On the words "
    "'left 3 people' she counts three fingers up on one hand, then lets the hand fall. "
    "The line stops mid-thought: she ends the shot with her mouth closed, her brows "
    "drawn together and her eyes flicking upward as if the next complaint is arriving.",
    ROOM_CONT)

# --- segment 6 : 209 frames (8.71 s), 20 words -------------------------------------
P6 = _prompt(
    "The same unbroken closeup continues: she lands the final complaint about the "
    "contestants' wardrobes and stops.",
    FRAMING + " " + CARRY + " " + DELIVERY + " Coming straight out of the upward eye "
    "flick, she completes the sentence without pausing just after the "
    "one-and-three-quarter-second mark, delivering it fast and flat in one breath, "
    "<d>[English] that all your clothes should fit in a shoe box that you had to skip "
    "getting any jeans and shirts?</d>. On the words 'shoe box' she sketches a small box "
    "in the air with both hands held close together. With the line finished she pushes "
    "both hands up into her hair and rakes them back through it in one quick exasperated "
    "movement, then drops them, closes her mouth and holds a long flat stare straight "
    "down the lens as the shot ends.",
    ROOM_CONT)


PROMPTS = [P1, P2, P3, P4, P5, P6]

# frames: 158, 192, 209, 158, 175, 209  (seconds are snapped up to the 17k+5 grid)
SECONDS = [6.58, 8.00, 8.71, 6.58, 7.29, 8.71]
