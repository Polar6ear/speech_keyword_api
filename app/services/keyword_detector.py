from rapidfuzz import fuzz, process

def detect_keyword(transcription_result: dict, target_keywords: list) -> list:
    detected = []
    segments = transcription_result.get("segments", [])
    spoken_words = []

    for segment in segments:
        for word_info in segment.get("word", []):
            spoken_words.append({
                "word": word_info["word"].strip().lower(),
                "start": word_info["start"],
                "end": word_info["end"]
            })


    #Matching if words exist or not
    for target in target_keywords:
        target_clean = target.lower()

        for spoken in spoken_words:
            spoken_word = spoken["word"]


            if spoken_word == target_clean:
                detected.append({
                    "keyword": target,
                    "match_word": spoken_word,
                    "start": spoken["start"],
                    "end": spoken["end"],
                    "confidence": 1.0,
                    "match_type": "exact"
                })
                continue

            
            similarity = fuzz.ration(spoken_word, target_clean)

            if similarity >= 85:
                detected.append({
                    "keyword": target,
                    "match_word": spoken_word,
                    "start": spoken["start"],
                    "end": spoken["end"],
                    "confidence": similarity / 100,
                    "match_type": "fuzzy"
                })

    return detected