"""
ocr/detector_interface.py

Defines clean interface abstractions for Region Detectors, Text Detectors,
and Text Recognizers. This allows future deep learning models (YOLO, DBNet,
CRAFT, SVTR, TrOCR) to be plugged in later without rewriting the pipeline.
"""

from abc import ABC, abstractmethod

class RegionDetector(ABC):
    """
    Abstract interface for detecting Ingredients or Nutrition regions on packaging.
    """
    @abstractmethod
    def detect(self, image, items=None, **kwargs):
        """
        Detects the region boundary.
        
        Args:
            image: numpy array (BGR or Grayscale image)
            items: optional list of already-extracted OCR line/item dicts
            
        Returns:
            dict containing:
                "bbox": [x1, y1, x2, y2] or None
                "confidence": float
                "method": str
                "matched_items": list
        """
        pass

class TextDetector(ABC):
    """
    Abstract interface for detecting text locations (localization only).
    """
    @abstractmethod
    def detect_text_areas(self, image, **kwargs):
        """
        Locates text lines/words polygons in the image.
        
        Args:
            image: numpy array
            
        Returns:
            list of polygons: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ...]
        """
        pass

class TextRecognizer(ABC):
    """
    Abstract interface for recognizing text within cropped image segments.
    """
    @abstractmethod
    def recognize_text(self, image, boxes, **kwargs):
        """
        Recognizes text within specified boxes.
        
        Args:
            image: numpy array
            boxes: list of polygons or bounding boxes
            
        Returns:
            list of dicts containing:
                "text": str
                "confidence": float
        """
        pass
