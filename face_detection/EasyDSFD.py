from dsfd.detect import DSFDDetector


class EasyDSFD:

    def __init__(self):
        self.detector = DSFDDetector()

    def detect(self, frame):
        detections = self.detector.detect_face(frame, confidence_threshold=.5, shrink=1.0)
        return detections


