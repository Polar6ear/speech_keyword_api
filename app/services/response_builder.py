def build_detection_response(detected_keywords: list):
    summary = {}

    for item in detected_keywords:
        word = item["keyword"]
        timestamp = (item.get("start"), item.get("end"))

        if word not in summary:
            summary[word] = {
                "count": 0,
                "timestamp": [],
            }

        summary[word]["count"] += 1
        if timestamp:
            summary[word]["timestamp"].append(timestamp)

    response = {
        "total_keywords_found": sum(v["count"] for v in summary.values()),
        "unique_words": len(summary),
        "keywords": summary  
    } 

    return response
