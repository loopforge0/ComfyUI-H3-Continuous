# mia influencer

Talking to camera, unbroken. 4 renders at 480x864, chained with
`MiniMaxH3AddGuide`.

Prompts are copied verbatim out of the session manifest, which records the string that was
actually sent alongside the seed and the frame counts. That is a better source than the
generator scripts in `tools/`, which can be edited after a run.

## Wiring

| Tag | Image | What it is |
| --- | --- | --- |
| `<Picture 1>` | `refs/mia.jpg` | three-panel character sheet |
| none | - | `MiniMaxH3AddGuide`, last 39 frames of the previous segment plus its audio |

The guide clip carries no tag, so the prompts describe it in prose without labelling it.



## Segments

`handoff` is the number of closing frames passed forward to anchor the next segment, so the
last one is always 0.

| # | frames | seconds | handoff | seed |
| --- | --- | --- | --- | --- |
| 1 | 192 | 8.0s | 39 | `4422180` |
| 2 | 192 | 8.0s | 39 | `4432153` |
| 3 | 192 | 8.0s | 39 | `4442126` |
| 4 | 192 | 8.0s | 0 | `4452099` |
