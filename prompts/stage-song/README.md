# stage song

Singing on a stage, three renders. 3 renders at 480x864, chained with
`MiniMaxH3AddGuide`.

Prompts are copied verbatim out of the session manifest, which records the string that was
actually sent alongside the seed and the frame counts. That is a better source than the
generator scripts in `tools/`, which can be edited after a run.

## Wiring

| Tag | Image | What it is |
| --- | --- | --- |
| `<Picture 1>` | `refs/kate.jpg` | the singer |
| `<Picture 2>` | `refs/stage.jpg` | the stage |
| none | - | `MiniMaxH3AddGuide`, last 39 frames of the previous segment plus its audio |

The guide clip carries no tag, so the prompts describe it in prose without labelling it.

Wiring confirmed: `tools/kate_stage_prompts.py` declares `IMAGES = [["kate.png", "stage.png"] for _ in range(3)]`.

## Segments

`handoff` is the number of closing frames passed forward to anchor the next segment, so the
last one is always 0.

| # | frames | seconds | handoff | seed |
| --- | --- | --- | --- | --- |
| 1 | 192 | 8.0s | 39 | `22318` |
| 2 | 192 | 8.0s | 39 | `32291` |
| 3 | 192 | 8.0s | 0 | `42264` |
