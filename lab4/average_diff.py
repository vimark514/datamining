import cv2


class AverageDifferenceCalculator:
    def __init__(self, image):
        self.image = image

    def calculate(self):
        mean, stddev = cv2.meanStdDev(self.image)
        return mean
