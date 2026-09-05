# Market Street

One continuous front-tracking walk down Market Street in San Francisco, seven renders long,
built twice: once locally on an RTX 3060 at 864x480, once on a rented RTX 5090 at 1376x768.

`3060/` and `5090/` are two independent prompt sets for the same seven shots. They share the
shot plan, the beats, the characters and the continuity block; about 70% of the wording is
shared, measured token by token. The 5090 set was reworked for the longer shot that card could
hold and adds two instructions the local set does not carry, one holding the camera steady and
one keeping her expression neutral between the scripted beats.

## Wiring

Tag numbers come from the order references are wired into the node, not from anything in the
prompt text. For every shot in both sets:

| Wired input | Tag | Image |
| --- | --- | --- |
| `ref_images.ref_image_0` | `<Picture 1>` | `refs/block-N.jpg`, the block shot from overhead |
| `ref_images.ref_image_1` | `<Picture 2>` | `refs/amber.jpg` |
| `ref_images.ref_image_2..4` | `<Picture 3..5>` | that shot's guest, in the order listed below |
| `MiniMaxH3AddGuide` | none | last 39 frames of the previous shot, plus its audio |

The guide clip carries no tag. It never reaches the tokenizer, so the prompts describe it in
prose and must not label it - a `<Video 1>` there would be an unresolved reference label.

## Shots

| # | Block ref | Guest | Beat |
| --- | --- | --- | --- |
| 1 | `block-1.jpg` | - | Establish; she comes off a crosswalk and settles into stride |
| 2 | `block-2.jpg` | sofia | High-five on the move, then peels away |
| 3 | `block-3.jpg` | - | F-line streetcar crosses behind; pigeons scatter |
| 4 | `block-4.jpg` | allie | Selfie taken without either of them stopping |
| 5 | `block-5.jpg` | - | Dappled light under street trees; squints into the sun |
| 6 | `block-6.jpg` | dany | Falls into step, dances alongside, then drops back |
| 7 | `block-7.jpg` | sofia + allie + dany | All three meet her; camera arcs to reveal the clock tower |

## Settings

| | 3060 | 5090 |
| --- | --- | --- |
| resolution | 864x480 | 1376x768 |
| length | 192 frames, 8.00s | 243 frames, 10.13s |
| anchor | 39 frames | 39 frames |
| steps | 20 | 20 |
| seed | `1078725334526148 + shot * 9973` | same, except shot 6 |

Shot 6 on the 5090 run used seed `8613402297755119`. It landed on the eighth attempt, and the
seed is the one that worked rather than the one the formula gives.

| shot | 3060 seed | 5090 seed |
| --- | --- | --- |
| 1 | `1078725334536121` | `1078725334536121` |
| 2 | `1078725334546094` | `1078725334546094` |
| 3 | `1078725334556067` | `1078725334556067` |
| 4 | `1078725334566040` | `1078725334566040` |
| 5 | `1078725334576013` | `1078725334576013` |
| 6 | `1078725334585986` | `8613402297755119` |
| 7 | `1078725334595959` | `1078725334595959` |

Every shot replays its 39-frame anchor, so only 153 (3060) or 204 (5090) frames per shot are
new. That is why seven eight-second renders come to a 46s cut rather than 56s.
