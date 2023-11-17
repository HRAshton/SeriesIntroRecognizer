from typing import NamedTuple

from imagehash import ImageHash
from scenedetect import FrameTimecode


class Scene(NamedTuple):
    start: FrameTimecode
    end: FrameTimecode

    def duration_secs(self):
        return self.end.get_seconds() - self.start.get_seconds()


class HashedScene(NamedTuple):
    scene: Scene
    hash: ImageHash
