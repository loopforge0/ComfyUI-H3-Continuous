# kate rant

Six renders of continuous speech, the drift example. 6 renders at 480x864, chained with
`MiniMaxH3AddGuide`.

Prompts are copied verbatim out of the session manifest, which records the string that was
actually sent alongside the seed and the frame counts. That is a better source than the
generator scripts in `tools/`, which can be edited after a run.

## Wiring

| Tag | Image | What it is |
| --- | --- | --- |
| `<Picture 1>` | `refs/kate.jpg` | character plate |
| none | - | `MiniMaxH3AddGuide`, last 39 frames of the previous segment plus its audio |

The guide clip carries no tag, so the prompts describe it in prose without labelling it.

**Which Kate plate this used is not confirmed.** `assets/characters/kate.png` and `kate_closeup.png` are near-identical crops of the same shot and the runner that queued this session is not in the repo. `kate_closeup.png` is staged here because its framing matches the take, but check before publishing.

## Segments

`handoff` is the number of closing frames passed forward to anchor the next segment, so the
last one is always 0. Segment lengths are not uniform here: each is sized to the words it has to carry, because runtime with nothing scripted in it is where the model invents speech-shaped filler.

| # | frames | seconds | handoff | seed |
| --- | --- | --- | --- | --- |
| 1 | 158 | 6.583s | 39 | `8824375` |
| 2 | 192 | 8.0s | 39 | `8834348` |
| 3 | 209 | 8.708s | 39 | `8844321` |
| 4 | 158 | 6.583s | 39 | `8854294` |
| 5 | 175 | 7.292s | 39 | `8864267` |
| 6 | 209 | 8.708s | 0 | `8874240` |
