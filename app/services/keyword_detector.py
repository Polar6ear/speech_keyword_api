import re
from rapidfuzz import fuzz
import jellyfish 
import logging
logger = logging.getLogger(__name__)

def clean_word(word: str) -> str:
    # remove punctuation + lowercase
    return re.sub(r"[^\w\s]", "", word.lower()).strip()


def detect_keyword(transcription_result: dict, target_keywords: list) -> list:
    detected = []
    segments = transcription_result.get("segments", [])
    spoken_words = []

    for segment in segments:
        for word_info in segment.get("words", []):
            cleaned = clean_word(word_info["word"])
            if cleaned:
                spoken_words.append({
                    "word": cleaned,
                    "start": word_info["start"],
                    "end": word_info["end"]
                })

    for target in target_keywords:

        target_clean = clean_word(target)
        target_len = len(target_clean.split())

        for i in range(len(spoken_words) - target_len + 1):

            phrase_words = spoken_words[i:i + target_len]
            phrase = " ".join([w["word"] for w in phrase_words])

            if len(target_clean) <= 3:
                similarity = fuzz.ratio(phrase, target_clean)
                threshold = 92

            elif target_len == 1:
                similarity = fuzz.ratio(phrase, target_clean)
                threshold = 87
            else:
                similarity = fuzz.token_set_ratio(phrase, target_clean)
                threshold = 80

            phonetic_match = False
            if 70 <= similarity < threshold:
                try:
                    phonetic_match = (
                        jellyfish.soundex(phrase) == jellyfish.soundex(target_clean)
                    )
                except:
                    phonetic_match = False
                    
            if 65 <= similarity < threshold:
                logger.debug(f"Borderline match: {phrase} ~ {target_clean} ({similarity})")

            if similarity >= threshold or (similarity >= 70 and phonetic_match):

                already_exists = any(
                    d["keyword"] == target
                    and abs(d["start"] - phrase_words[0]["start"]) < 1.0
                    for d in detected
                )

                if already_exists:
                    continue

                detected.append({
                    "keyword": target,
                    "match_word": phrase,
                    "start": phrase_words[0]["start"],
                    "end": phrase_words[-1]["end"],
                    "confidence": round(max(similarity / 100, 0.9 if phonetic_match else similarity / 100),3),
                    "match_type": "exact" if similarity == 100 else "fuzzy"
                })

    return detected