import re
from rapidfuzz import fuzz


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

            similarity = fuzz.token_set_ratio(phrase, target_clean)

            if target_len == 1:
                similarity = fuzz.ratio(phrase, target_clean)
                threshold = 85
            else:
                similarity = fuzz.token_set_ratio(phrase, target_clean)
                threshold = 80

            if similarity >= threshold:

                already_exists = any(
                    d["keyword"] == target
                    and abs(d["start"] - phrase_words[0]["start"]) < 0.2
                    for d in detected
                )

                if already_exists:
                    continue

                detected.append({
                    "keyword": target,
                    "match_word": phrase,
                    "start": phrase_words[0]["start"],
                    "end": phrase_words[-1]["end"],
                    "confidence": round(similarity / 100, 3),
                    "match_type": "exact" if similarity == 100 else "fuzzy"
                })

    return detected