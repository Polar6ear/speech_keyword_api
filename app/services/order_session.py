import time
from matplotlib import text
import numpy as np


class OrderSession:
    def __init__(self, silence_threshold: float = 4.0):
        self.items = {}
        self.detected_keywords = {}
        self.last_speech_time = time.time()
        self.silence_threshold = silence_threshold
        self.previous_tail = np.array([], dtype=np.float32)
        self.last_emitted_text = ""
        self.last_sent_end_time = 0.0

    def update_order(self, item: str, qty: int):
        """Max quantity rakho — Window 1 ka result best hota hai"""
        if item not in self.items or qty > self.items[item]:
            self.items[item] = qty

    def is_duplicate_keyword(self, keyword: str, start_time: float) -> bool:
        """Same keyword 3 sec mein dobara detect hua toh skip karo"""
        if keyword in self.detected_keywords:
            if abs(start_time - self.detected_keywords[keyword]) < 3.0:
                return True
        self.detected_keywords[keyword] = start_time
        return False

    def mark_speech(self):
        """Jab bhi speech detect ho — timestamp update karo"""
        self.last_speech_time = time.time()

    def is_complete(self) -> bool:
        """Silence threshold cross ho gayi = order complete"""
        return (
            time.time() - self.last_speech_time > self.silence_threshold
            and bool(self.items)
        )
    
    HALLUCINATION_PHRASES = [
        "food order system", "items", "thank you",
        "thanks for watching", "please subscribe"
    ]

    def is_hallucination(self, text: str) -> bool:
        text_lower = text.lower().strip()
        return any(phrase in text_lower for phrase in self.HALLUCINATION_PHRASES)

    def get_order_list(self) -> list:
        """Final order list return karo"""
        return [
            {"item": k, "quantity": v}
            for k, v in self.items.items()
        ]

    def reset(self):
        """Order complete hone ke baad reset karo"""
        self.items = {}
        self.detected_keywords = {}
        self.last_sent_end_time = 0.0
        self.previous_tail = np.array([], dtype=np.float32)
        self.last_speech_time = time.time()

        # last_emitted_text reset nahi karte — hallucination filter ke liye
    def flush_context(self):
        """Silence ke baad audio context clear karo"""
        self.previous_tail = np.array([], dtype=np.float32)
        self.last_emitted_text = ""
        self.last_sent_end_time = 0.0