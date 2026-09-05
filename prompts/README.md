# Prompts

Every prompt behind the takes in the video, one file per render. Twenty seven of them across
five takes, copied byte for byte from what was sent rather than tidied up afterwards.

| Folder | Take | Renders |
| --- | --- | --- |
| `market-street/3060/` | The walk, local RTX 3060 | 7 |
| `market-street/5090/` | The walk, rented RTX 5090 | 7 |
| `mia-influencer/` | Talking to camera | 4 |
| `kate-rant/` | Six renders of speech, the drift example | 6 |
| `stage-song/` | Singing on a stage | 3 |

## The format

All twenty seven use the same six sections, written against MiniMax's own
[video prompt writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md):

```
subject_definitions:     what each <Subject N> is, and which <Picture N> it comes from
summary:                 one line: what the target video is
retention_analysis:      per subject, what is kept exactly and what is not
detailed_description:    the shot itself, second by second
overall_soundscape:      diegetic audio
non_diegetic_music:      score, or N/A
```

`non_diegetic_music` is `N/A` in all twenty seven. Separately generated score beds do not match
across a join, so music goes on over the finished cut.

Line endings are LF here. Some of the source files on the machine that made them are CRLF, but
the runner reads them in text mode, so the string that reached the model had LF either way and
that is what is stored.

## The one thing a prompt cannot tell you

`<Picture 1>`, `<Picture 2>` and so on are **positional**. The number comes from the order the
reference images are wired into the node, not from anything written in the prompt. Read a
prompt without knowing that order and `<Picture 2>` is unresolvable.

Each folder's README carries its wiring table. Start there.

## Reference images

`refs/` in each folder holds the images those prompts point at, resized to 1024px on the long
edge and saved as JPEG at quality 92. The block references are aerial photographs rather than
map tiles, so nothing here carries text that JPEG would soften. Full resolution originals are
not published: H3 does not need them and they come to 38 MB.
