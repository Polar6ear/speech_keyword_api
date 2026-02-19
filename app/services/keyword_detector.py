from rapidfuzz import fuzz

def detect_keyword(transcription_result: dict, target_keywords: list) -> list:
    detected = []
    segments = transcription_result.get("segments", [])
    spoken_words = []

    for segment in segments:
        for word_info in segment.get("words", []):
            spoken_words.append({
                "word": word_info["word"].strip().lower(),
                "start": word_info["start"],
                "end": word_info["end"]
            })

    combined_words = []

    # 2-word combinations
    for i in range(len(spoken_words) - 1):
        phrase = spoken_words[i]["word"] + " " + spoken_words[i + 1]["word"]
        combined_words.append({
            "word": phrase,
            "start": spoken_words[i]["start"],
            "end": spoken_words[i+1]["end"]
        })

    # 3-word combinations
    for i in range(len(spoken_words) - 2):
        phrase = (
            spoken_words[i]["word"] + " " +
            spoken_words[i + 1]["word"] + " " +
            spoken_words[i + 2]["word"]
        )
        combined_words.append({
            "word": phrase,
            "start": spoken_words[i]["start"],
            "end": spoken_words[i+2]["end"]
        })

    all_spoken_words = spoken_words + combined_words

    for target in target_keywords:
        target_clean = target.lower()

        for spoken in all_spoken_words:
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

            similarity = fuzz.token_set_ratio(spoken_word, target_clean)

            if similarity >= 82:

                already_exists = any(
                    d["keyword"] == target and abs(d["start"] - spoken["start"]) < 0.2
                    for d in detected
                )

                if already_exists:
                    continue

                detected.append({
                    "keyword": target,
                    "match_word": spoken_word,
                    "start": spoken["start"],
                    "end": spoken["end"],
                    "confidence": similarity / 100,
                    "match_type": "fuzzy"
                })

    return detected



