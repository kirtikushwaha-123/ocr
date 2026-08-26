"""
models/yolo_region_detector.py

OPTIONAL FUTURE MODULE - NOT USED IN VERSION 1.

This module is intentionally a stub / scaffold. There is currently no
trained YOLO model (`best.pt`, `ingredients.pt`, `nutrition.pt`, etc.)
because there is no real annotated dataset of package photographs yet.

Once you have collected and annotated 500-1000+ real package images with:
    class 0 = ingredients
    class 1 = nutrition

...you can train a YOLO model (e.g. with the `ultralytics` package) and
point this module at the resulting weights file. The rest of the PicWise
pipeline (main.py) will then be able to call `YoloRegionDetector.detect()`
INSTEAD of (or as a prior for) the OCR-anchor heuristics in
`detection/ingredient_region.py` and `detection/nutrition_region.py`,
by passing `--use-yolo` and `--yolo-weights <path>` on the CLI once that
flag is wired up.

This module is never imported by main.py by default, and importing it
does not require `ultralytics` to be installed - the dependency is only
required if you actually instantiate and use `YoloRegionDetector`.
"""


class YoloRegionDetector:
    """
    Thin wrapper around an Ultralytics YOLO model, trained to detect two
    classes: 0 = ingredients region, 1 = nutrition region.

    Usage (once you have real trained weights):

        from models.yolo_region_detector import YoloRegionDetector

        detector = YoloRegionDetector(weights_path="weights/picwise_yolo.pt")
        regions = detector.detect(image)
        # regions -> {"ingredients": {"bbox": [...], "confidence": 0.93},
        #             "nutrition":   {"bbox": [...], "confidence": 0.88}}
    """

    CLASS_NAMES = {0: "ingredients", 1: "nutrition"}

    def __init__(self, weights_path, confidence_threshold=0.35, device=None):
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "The 'ultralytics' package is required to use YoloRegionDetector. "
                "Install it with: pip install ultralytics\n"
                f"(original error: {e})"
            )

        import os

        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(
                f"YOLO weights file not found: {self.weights_path}. "
                "You must train a model first (see module docstring) - "
                "PicWise Version 1 does not ship with any trained weights."
            )

        self._model = YOLO(self.weights_path)

    def detect(self, image):
        """
        Runs YOLO inference on `image` (BGR numpy array) and returns:

            {
                "ingredients": {"bbox": [x1,y1,x2,y2], "confidence": float} or None,
                "nutrition":   {"bbox": [x1,y1,x2,y2], "confidence": float} or None,
            }

        Only the highest-confidence detection per class is returned. If a
        class isn't detected above confidence_threshold, its value is None.
        """
        self._load_model()

        predict_kwargs = {"conf": self.confidence_threshold}
        if self.device:
            predict_kwargs["device"] = self.device

        results = self._model.predict(image, **predict_kwargs)

        best_by_class = {}

        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = [float(v) for v in box.xyxy[0]]

                label = self.CLASS_NAMES.get(cls_id)
                if label is None:
                    continue

                current_best = best_by_class.get(label)
                if current_best is None or conf > current_best["confidence"]:
                    best_by_class[label] = {"bbox": xyxy, "confidence": conf}

        return {
            "ingredients": best_by_class.get("ingredients"),
            "nutrition": best_by_class.get("nutrition"),
        }


def is_weights_available(weights_path):
    import os

    return bool(weights_path) and os.path.isfile(weights_path)
