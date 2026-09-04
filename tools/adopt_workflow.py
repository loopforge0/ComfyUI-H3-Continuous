"""Take a workflow you rearranged in ComfyUI and make it shippable.

`build_workflows.py` generates the graphs, but a graph laid out by a program is never
as readable as one a person has dragged into shape. When you rearrange a shipped
workflow and save it, run this to pull your version back into the repo:

    python tools/adopt_workflow.py h3_continuous

It keeps everything you changed -- node positions, sizes, group bounds, widget values --
and fixes only the two things a hand-saved workflow gets wrong for someone opening it
for the first time:

1. **The saved viewport.** ComfyUI stores wherever you were last looking in `extra.ds`
   and restores it verbatim on open. Yours is usually parked on whatever you were
   editing, which for a new user opens on blank canvas. Re-anchored to the top-left of
   the graph at a readable zoom.

2. **Model metadata.** `properties.models` on each loader is what makes ComfyUI offer
   to download missing models when the workflow opens, instead of showing an empty
   dropdown. ComfyUI does not write it when you save, so it is re-attached here.

Pass --to-comfy to copy the result back over the ComfyUI-side copy as well.
"""

import argparse
import json
import os
import shutil

from build_workflows import Builder, model_entry

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def comfy_workflows():
    """Where ComfyUI saves workflows.

    Installed normally, this pack lives in ComfyUI/custom_nodes/<pack>, so ComfyUI's
    root is two levels up and there is nothing to configure. Set COMFYUI_DIR (or pass
    --src) when the checkout lives somewhere else, as it does when you develop the pack
    outside the install and mirror it in.
    """
    root = os.environ.get("COMFYUI_DIR") or os.path.dirname(os.path.dirname(HERE))
    path = os.path.join(root, "user", "default", "workflows")
    return path if os.path.isdir(path) else None


def attach_models(graph):
    """Put properties.models back on every loader that names a known H3 file."""
    touched = []
    for node in graph["nodes"]:
        found = [m for m in (model_entry(v) for v in (node.get("widgets_values") or [])) if m]
        props = node.setdefault("properties", {})
        if found:
            props["models"] = found
            touched.append((node.get("title") or node["type"], found[0]["name"]))
        else:
            props.pop("models", None)
    return touched


def refit(graph):
    """Re-anchor extra.ds on the graph's top-left, using the builder's own rule."""
    b = Builder.__new__(Builder)
    b.nodes, b.groups = graph["nodes"], graph.get("groups", [])
    graph.setdefault("extra", {})["ds"] = b._fit()
    return graph["extra"]["ds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="workflow name, with or without .json")
    found = comfy_workflows()
    ap.add_argument("--src", default=found, required=found is None,
                    help="ComfyUI's workflows folder. Found automatically when this "
                         "pack sits in ComfyUI/custom_nodes; otherwise set COMFYUI_DIR "
                         "or pass this.")
    ap.add_argument("--to-comfy", action="store_true",
                    help="also copy the adopted file back over the ComfyUI-side copy")
    a = ap.parse_args()

    name = a.name if a.name.endswith(".json") else a.name + ".json"
    src = os.path.join(a.src, name)
    dst = os.path.join(HERE, "workflows", name)
    if not os.path.exists(src):
        raise SystemExit("no such file: %s" % src)

    with open(src, encoding="utf-8") as f:
        graph = json.load(f)

    if not graph.get("id"):
        # See build_workflows.dump: a null id makes ComfyUI-Manager's handleFile hook
        # throw, and the workflow then silently fails to open via File -> Open.
        raise SystemExit("%s has no id -- open and re-save it in ComfyUI first" % name)

    before = json.load(open(dst, encoding="utf-8")) if os.path.exists(dst) else None
    touched = attach_models(graph)
    ds = refit(graph)

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    print("adopted %s -> %s" % (src, dst))
    if before:
        moved = sum(1 for n, o in zip(graph["nodes"], before["nodes"]) if n["pos"] != o["pos"])
        print("  %d of %d nodes moved" % (moved, len(graph["nodes"])))
    print("  viewport: scale=%.2f offset=%s" % (ds["scale"], ds["offset"]))
    for title, model in touched:
        print("  models: %-42s %s" % (title[:42], model))

    if a.to_comfy:
        shutil.copyfile(dst, src)
        print("  copied back to %s" % src)


if __name__ == "__main__":
    main()
