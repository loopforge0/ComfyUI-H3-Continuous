"""The three-segment stage example: one singer, one song, one unbroken take.

The smallest chain worth shipping. Three 8-second H3 Render Segment nodes at the
default 1.625 s handoff make a ~20.75 s continuous performance -- enough to show every
rule the pack depends on, short enough to render in one sitting.

Three things this example exists to demonstrate
-----------------------------------------------
1. **Verbatim repetition.** STYLE, FRAMING, KATE and STAGE below are pasted into all
   three prompts unchanged, word for word. That repetition -- not the anchor -- is what
   keeps the spotlight the same colour and her sweater the same sweater across a join.

2. **Nothing new before 1.7 s.** Segments 2 and 3 open on a fixed replay of the
   previous segment's last 1.625 s. Each of their prompts therefore spends its opening
   sentence *finishing a move that is already in progress*, and schedules its first
   sung word after that. A shot that starts a new action at 0.0 s is fighting frames it
   cannot change.

3. **Sing, then move -- never both.** H3 spends its motion budget once. A dance beat
   underneath a sung line freezes the mouth mid-word; a sung line underneath a dance
   beat flattens the dance. So every segment alternates: sing a phrase with the body
   quiet, then dance in the gap. Each segment also ends on a NON-singing beat, because
   the next one replays those frames and a replayed half-word cannot be resumed.

Why the song is unaccompanied
-----------------------------
`non_diegetic_music` is `N/A` in all three on purpose, and the stage's band gear is
explicitly unplayed. Three independently generated score beds would not match across
the joins -- the key would move, the tempo would drift, and the two seams would become
the most audible thing in the video. So she sings a cappella in a room with a long
reverb tail. Put the music under the finished cut in post, where it is one continuous
track.
"""

# --- the blocks repeated verbatim in every segment ----------------------------------
# Change one of these and you must change it in all three. That is the point.

KATE = (
    "the young woman in <Picture 1>, in her early twenties, with shoulder-length wavy "
    "mid-brown hair worn loose with soft lilac-purple tones running through the "
    "mid-lengths and ends, fair clear skin, blue-grey eyes, dark natural brows and a "
    "slim build, wearing a chunky oatmeal-and-brown waffle-knit crew-neck sweater and "
    "dark slim jeans. Only her face, hair, colouring and sweater are taken from "
    "<Picture 1>; the calm closed-mouth half-smile and the still seated portrait pose "
    "in that image are not retained, because she is singing and dancing throughout the "
    "target video."
)

STAGE = (
    "the theatre stage in <Picture 2>: a full-width deep red velvet curtain hanging in "
    "heavy vertical folds across the back, a glossy honey-coloured wooden floor, a "
    "single warm amber followspot throwing a visible cone of light straight down onto "
    "centre stage and a dappled gobo pattern across the boards around it, a chrome "
    "vocal microphone on a black straight stand at centre, a drum kit on a low riser "
    "upstage behind the microphone, a two-tier keyboard rig and stool stage-left, a "
    "black monitor wedge and a small warm practical lamp stage-right, and an electric "
    "guitar on a stand at the far stage-right edge. Only the layout, shapes, materials, "
    "colours and lighting of that image are used. No musicians are present, and the "
    "drum kit, keyboards and guitar are never touched or played."
)

STYLE = (
    "The target video is photorealistic live-action concert footage shot vertically, "
    "filmed on a single locked-off camera at chest height about four metres back from "
    "centre stage, lit only by the warm amber followspot against the deep red curtain "
    "with the rest of the stage falling away into shadow, with a shallow depth of field "
    "so the drum kit and the curtain behind her are softly blurred, and with no "
    "on-screen text and no subtitles anywhere."
)

FRAMING = (
    "The camera is locked off and holds the same full-length wide shot for the whole "
    "segment with no cut, her whole body inside the cone of light with headroom above "
    "her and the dappled stage floor visible below her feet, never panning, zooming, "
    "cutting or changing its distance."
)

# The motion-budget rule is the second-to-last sentence and the anti-babble clause is
# the last one. Both are load-bearing -- see the module docstring.
DELIVERY = (
    "She sings unaccompanied in a clear, warm young female voice at a relaxed "
    "mid-tempo, her jaw and lips opening and closing distinctly on every syllable so "
    "every word of the lyric is clearly formed, and she never settles into a fixed open "
    "smile or a laugh that would freeze her mouth. While she is singing her body stays "
    "quiet and she moves only her head, shoulders and one hand; every full dance move "
    "lands in the gap between sung lines rather than underneath one. She sings exactly "
    "the words given below and no others: she does not ad-lib, does not repeat herself, "
    "does not hum and does not add any further vocal to fill out the shot."
)

CARRY = (
    "The segment opens on an exact replay of the closing moments of the preceding take "
    "and carries that motion straight through, so the join is invisible. She holds the "
    "exact body pose, head angle, gaze direction, arm position and facial expression "
    "carried in by the replayed opening and continues out of it without any reset or "
    "restart, and no new action begins before the 1.7-second mark."
)

RETAIN = (
    "<Subject 1> (appears in [Shot 1]): fully_preserved - her face, blue-grey eyes, "
    "shoulder-length wavy mid-brown hair with its lilac-purple tones and her chunky "
    "oatmeal waffle-knit sweater are retained exactly; the calm still portrait pose of "
    "<Picture 1> is not retained.\n"
    "<Subject 2> (appears in [Shot 1]): partially_preserved - the red velvet curtain, "
    "honey wooden floor, amber followspot and gobo dapple, centre microphone stand, "
    "drum riser, keyboard rig, monitor wedge and guitar stand are retained in the same "
    "layout; the empty framing of <Picture 2> is not retained, because <Subject 1> is "
    "performing in the light."
)

ROOM = (
    "A large empty theatre's room tone with a long natural reverb tail off the wooden "
    "stage and the back wall, her unaccompanied voice carrying out into that space, "
    "with the soft scuff and tap of her shoes on the boards and the faint rustle of her "
    "sweater as she moves. No instruments are played, there is no audience and there is "
    "no other sound."
)

ROOM_CONT = (
    "The large empty theatre's room tone carries straight on from the preceding segment "
    "without interruption: the same long natural reverb tail off the wooden stage and "
    "the back wall, her unaccompanied voice carrying out into that space, with the soft "
    "scuff and tap of her shoes on the boards and the faint rustle of her sweater as "
    "she moves. No instruments are played, there is no audience and there is no other "
    "sound."
)


def _prompt(summary, shot, sound):
    return """subject_definitions:
<Subject 1> is %s
<Subject 2> is %s

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
N/A""" % (KATE, STAGE, summary, RETAIN, STYLE, shot, sound)


# --- segment 1: 8 s, all new --------------------------------------------------------
# Opens cold, so it is the only one free to start an action at 0.0 s. It also
# establishes the stage, and is the only absolute reference the other two inherit.
P1 = _prompt(
    "[reference generation] <Subject 1> stands alone in the followspot at the centre "
    "microphone of the empty stage <Subject 2>, sings the opening line of an "
    "unaccompanied song, then steps back off the microphone and opens into the first "
    "move of a slow dance.",

    FRAMING + " " + DELIVERY + " The shot opens on her already standing at the chrome "
    "microphone at centre stage inside the cone of amber light, feet together, her "
    "right hand resting lightly on the top of the black stand and her left hand at her "
    "side, looking straight down the lens. She draws a breath and sings the opening "
    "line, her body still and only her shoulders lifting with the phrase, "
    "<d>[English] I have been waiting on the light to find me.</d> "
    "She closes her mouth on the last word. Only then does she move: she lets go of the "
    "microphone stand, takes one unhurried step back and to her right along the boards, "
    "and begins a slow half-turn away from the lens with both arms drifting up and out "
    "to shoulder height, her hair swinging out behind her. The segment ends mid-turn, "
    "her body angled away, arms still raised, her mouth closed and singing nothing.",

    ROOM)


# --- segment 2: 8 s, first 1.625 s is a replay of segment 1's tail ------------------
# The opening sentence finishes segment 1's half-turn rather than starting anything.
P2 = _prompt(
    "[reference generation] <Subject 1> comes out of the turn she is already in, sings "
    "the second line of the unaccompanied song, then travels across the followspot on "
    "the empty stage <Subject 2> in a short step sequence.",

    CARRY + " " + FRAMING + " " + DELIVERY + " She completes the slow half-turn she is "
    "already in, bringing her face back round to the lens and letting both raised arms "
    "fall softly to her sides as she comes square again, a little to the right of the "
    "microphone stand. Settled and facing front, she sings the second line, her body "
    "quiet and only her chin lifting into it, "
    "<d>[English] and here it is, warm on my hands tonight.</d> "
    "She closes her mouth, and on the silence she turns her palms up in front of her "
    "and dances: two smooth travelling steps to her left back across the cone of light, "
    "a low sweep of both arms out and down that follows the step, and a quick clean "
    "pivot on the ball of her left foot that brings her round to face the lens again. "
    "The segment ends just out of the pivot, her weight settling onto her right foot, "
    "arms low and open, her mouth closed and singing nothing.",

    ROOM_CONT)


# --- segment 3: 8 s, last of the chain -- handoff_seconds is 0 ----------------------
# Ends on held stillness rather than a move cut off mid-air, because nothing follows it.
P3 = _prompt(
    "[reference generation] <Subject 1> settles out of the pivot, sings the closing "
    "line of the unaccompanied song, and finishes on a full turn that comes down into "
    "stillness in the followspot on the empty stage <Subject 2>.",

    CARRY + " " + FRAMING + " " + DELIVERY + " Her weight finishes settling out of the "
    "pivot and she comes to a balanced stand facing the lens, arms low and open, back "
    "at the centre of the cone of amber light. She lifts her head and sings the closing "
    "line, her body still and both hands opening a little wider at her waist, "
    "<d>[English] so let it burn, I am not going home.</d> "
    "She closes her mouth on the last word. Into the silence she dances the finish: a "
    "single full turn on the spot, rising onto the balls of her feet as she comes round "
    "with both arms sweeping up overhead, her hair lifting and falling with the turn. "
    "She comes down out of it facing the lens, arms lowering slowly to her sides, and "
    "holds there completely still with her chest rising once, her mouth closed and "
    "singing nothing, the amber followspot steady on her and the red curtain unmoving "
    "behind her. The segment ends on that held stillness.",

    ROOM_CONT)


PROMPTS = [P1, P2, P3]

# <Picture 1> is Kate, <Picture 2> is the stage -- the same two images in all three
# segments, which is what holds her face and the room together across the joins.
IMAGES = [["kate.png", "stage.png"] for _ in range(3)]
