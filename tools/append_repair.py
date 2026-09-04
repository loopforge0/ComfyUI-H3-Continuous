"""Append the repair nodes to the example workflow, leaving its layout untouched.

The pack ships one workflow. It opens as a working three-shot example, and the repair
path lives in the same file, below the chain and bypassed, so fixing a bad segment does
not mean finding a second file and re-pointing five loaders at the same models.

    python tools/append_repair.py

Everything already in the file keeps its position, size, group and wiring -- this only
adds nodes below the existing graph, at ids and link ids past the ones in use. The
repair block ships **bypassed** (mode 4), so Run renders the chain and pays nothing for
the repair half until you un-bypass it.

Re-running is safe: if the file already has an H3 Repair Segment, it stops.
"""

import argparse
import json
import os
import urllib.request

from build_workflows import build_repair, BLUE, RED
from adopt_workflow import refit

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The repair half is these node types; everything else in the repair workflow is the
# models column, which the example already has and must not get a second copy of.
REPAIR_TYPES = ("H3LoadSession", "H3RepairSegment", "H3ChainToVideo", "SaveVideo",
                "PreviewAny")

BYPASS = 4

REPAIR_NOTE = """# Repair - fixing one bad segment

**This half is bypassed.** It costs nothing until you turn it on. To use it: select
these nodes, press **Ctrl+B** to un-bypass, then select the **shots** above and
**Ctrl+B** them off, so only one half runs. Running both renders both, and pays GPU
time for both.

One segment came out wrong - a fluffed line, a hand through a lamppost. Normally
re-rendering segment N changes its ending, so every segment after it stops joining and
has to be re-rendered too. On a seven-shot chain, fixing shot 2 costs six renders.

**H3 Repair Segment pins both ends.** The opening is pinned to the previous segment's
handoff clip; the ending is pinned to *the very clip the next segment already opens on*.
The replacement lands between two fixed points and nothing downstream moves.

## Do this
1. Type the `session_name` you rendered with into **H3 Load Session**. It prints what is
   in that session - segment numbers, lengths, seeds - on the preview beside it.
2. Set `segment_number` to the one you are replacing (1-based, same as `seg_NN.mp4`).
3. Rebuild that shot on the **H3 Repair Segment** node: same reference images, same
   length, fixed prompt.
4. Run. The old file is kept as `seg_NN.replaced.mp4`, the node's `video` output plays
   the replacement immediately, and the whole cut is re-joined.

It shares the **H3 Chain Settings** node above, so the models, sampler, canvas size and
`chain_seed` are already the ones the chain used. They have to be.

## Notes
- Leave `pin_ending` **on**. Turn it off only for the very last segment, or when you
  mean to re-render everything after this one.
- The pinned ending is anchored as picture only. Its audio comes from the render you are
  throwing away, and re-anchoring that would drag back exactly what you are removing.
- Keep the replacement the same length as the original. Repairs are for a bad take, not
  for a re-cut.
- `seg_NN.tail.mp4` is deliberately **not** rewritten. It is what segment N+1 opens on,
  and the repair was pinned to it."""


def append_repair(graph, info, top=None, gap=400):
    """Add the repair block below everything already in ``graph``. Returns node count."""
    if any(n["type"] == "H3RepairSegment" for n in graph["nodes"]):
        return 0

    settings = next(n for n in graph["nodes"] if n["type"] == "H3ChainSettings")

    if top is None:
        bottom = max([n["pos"][1] + n["size"][1] for n in graph["nodes"]]
                     + [g["bounding"][1] + g["bounding"][3] for g in graph["groups"]])
        top = round((bottom + gap) / 20.0) * 20

    # Build a throwaway repair workflow off the live schema, then lift its repair half
    # across. Typing the nodes out by hand here would let them drift from the real ones.
    src = build_repair(info)
    src = {"nodes": src.nodes, "links": src.links, "groups": src.groups}
    donor = {n["id"]: n for n in src["nodes"] if n["type"] in REPAIR_TYPES}

    # the donor's repair half starts at y=40; rebase it onto `top`
    dy = top - min(n["pos"][1] for n in donor.values())
    dx = 0

    next_id = max(n["id"] for n in graph["nodes"]) + 1
    next_link = max([l[0] for l in graph["links"]] + [0]) + 1
    remap = {}

    for old_id in sorted(donor):
        node = json.loads(json.dumps(donor[old_id]))
        remap[old_id] = node["id"] = next_id
        node["order"] = next_id
        next_id += 1
        node["pos"] = [node["pos"][0] + dx, node["pos"][1] + dy]
        node["mode"] = BYPASS
        for inp in node["inputs"]:
            inp["link"] = None
        for out in node["outputs"]:
            out["links"] = None
        graph["nodes"].append(node)

    added = {n["id"]: n for n in graph["nodes"] if n["id"] in remap.values()}

    def connect(origin, origin_slot, target, target_slot, link_type):
        nonlocal next_link
        lid = next_link
        next_link += 1
        graph["links"].append([lid, origin["id"], origin_slot,
                               target["id"], target_slot, link_type])
        target["inputs"][target_slot]["link"] = lid
        links = origin["outputs"][origin_slot].get("links") or []
        links.append(lid)
        origin["outputs"][origin_slot]["links"] = links

    # carry over every link that ran between two repair-half nodes
    for lid, oid, oslot, tid, tslot, ltype in src["links"]:
        if oid in remap and tid in remap:
            connect(added[remap[oid]], oslot, added[remap[tid]], tslot, ltype)

    # and hook the repair segment onto the settings node this file already has
    repair = next(n for n in added.values() if n["type"] == "H3RepairSegment")
    slot = next(i for i, inp in enumerate(repair["inputs"]) if inp["name"] == "settings")
    connect(settings, 0, repair, slot, "H3_SETTINGS")

    notes = [n for n in graph["nodes"] if n["type"] == "MarkdownNote"
             and n["pos"][0] < min(d["pos"][0] for d in donor.values())]
    note_x = min(n["pos"][0] for n in notes) if notes else 40
    note_w = max(n["size"][0] for n in notes) if notes else 980
    graph["nodes"].append({
        "id": next_id, "type": "MarkdownNote", "pos": [note_x, top], "size": [note_w, 760],
        "flags": {}, "order": next_id, "mode": 0, "inputs": [], "outputs": [],
        "title": "Repair", "properties": {}, "widgets_values": [REPAIR_NOTE],
        "color": RED, "bgcolor": RED,
    })
    next_id += 1

    # Reuse the donor's own group boxes, rebased by the same dy as its nodes, so the
    # boxes cannot drift away from what they are meant to enclose.
    def holds(g, node):
        x, y, w, h = g["bounding"]
        return x <= node["pos"][0] <= x + w and y <= node["pos"][1] <= y + h

    others = [n for n in src["nodes"] if n["id"] not in donor]
    for g in src["groups"]:
        x, y, w, h = g["bounding"]
        if not any(holds(g, n) for n in donor.values()):
            continue
        if any(holds(g, n) for n in others):
            continue          # the models-column group; this file already has one
        graph["groups"].append({
            "id": len(graph["groups"]) + 1,
            "title": "B - REPAIR (bypassed) - " + g["title"].split(" - ", 1)[-1],
            "bounding": [x, y + dy, w, h], "color": BLUE, "flags": {}})

    graph["last_node_id"] = next_id - 1
    graph["last_link_id"] = next_link - 1
    return len(remap) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--file", default=os.path.join(
        HERE, "workflows", "h3_continuous.json"))
    a = ap.parse_args()

    info = json.load(urllib.request.urlopen(a.server + "/object_info", timeout=60))
    with open(a.file, encoding="utf-8") as f:
        graph = json.load(f)

    before = len(graph["nodes"])
    positions = {n["id"]: list(n["pos"]) for n in graph["nodes"]}

    added = append_repair(graph, info)
    if not added:
        raise SystemExit("%s already has an H3 Repair Segment -- nothing to do" % a.file)

    moved = [n["id"] for n in graph["nodes"]
             if n["id"] in positions and n["pos"] != positions[n["id"]]]
    if moved:
        raise SystemExit("refusing to write: existing nodes moved: %s" % moved)

    ds = refit(graph)
    with open(a.file, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    print("%s: %d nodes -> %d (%d added, 0 moved)"
          % (os.path.basename(a.file), before, len(graph["nodes"]), added))
    print("  viewport: scale=%.2f offset=%s" % (ds["scale"], ds["offset"]))


if __name__ == "__main__":
    main()
