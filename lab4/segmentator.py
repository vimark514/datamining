import numpy as np


class Segmentator:
    def __init__(self, image, block_size):
        self.image = image
        self.block_size = block_size

    def segment_image(self):
        h, w = self.image.shape
        segments = []

        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                segment = self.image[y:min(y + self.block_size, h), x:min(x + self.block_size, w)]
                segments.append(segment)

        return segments
