from typing import NamedTuple

from imagehash import ImageHash
from scenedetect import FrameTimecode


class Scene(NamedTuple):
    start: FrameTimecode
    end: FrameTimecode


class HashedScene(NamedTuple):
    scene: Scene
    hash: ImageHash
