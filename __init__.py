"""ComfyUI-H3-Continuous: seamless multi-segment MiniMax H3 video.

Chain any number of H3 renders into one unbroken take. Each segment opens on an
exact replay of the previous segment's last frames, pinned with MiniMaxH3AddGuide
rather than merely conditioned on -- which is the difference between a joined shot
and a cut.
"""

from .h3_continuous import H3ContinuousExtension


async def comfy_entrypoint() -> H3ContinuousExtension:
    return H3ContinuousExtension()
