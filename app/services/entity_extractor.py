import re

WORD_TO_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "some": 2, "few": 2, "couple": 2, "double": 2
}


def extract_order_entities(text: str, keywords: list):
    """
    Extract ordered items and their quantities from transcribed text.

    Scans for food keywords and nearby numeric tokens (digits or word-numbers),
    then associates quantities with the closest keyword within a 2-word window.
    Items without an explicit quantity default to 1.
    """
    text_lower = text.lower()
    words = text_lower.split()

    # Find all keyword positions (supports multi-word keywords)
    items = []
    for keyword in keywords:
        start = 0
        while True:
            idx = text_lower.find(keyword, start)
            if idx == -1:
                break
            word_pos = len(text_lower[:idx].split())
            items.append((word_pos, keyword))
            start = idx + 1

    # Find all number positions
    numbers = []
    for i, word in enumerate(words):
        clean = re.sub(r"[^\w\s]", "", word)
        if clean.isdigit():
            numbers.append((i, int(clean)))
        elif clean in WORD_TO_NUM:
            numbers.append((i, WORD_TO_NUM[clean]))

    results = []
    used_items = set()

    # Associate each number with its nearest keyword (within 2 words)
    for num_idx, num in numbers:
        closest_item = None
        min_distance = float("inf")

        for item_idx, item in items:
            distance = abs(item_idx - num_idx)
            if distance < min_distance and distance <= 3:
                min_distance = distance
                closest_item = item

        if closest_item and closest_item not in used_items:
            results.append({"item": closest_item, "quantity": num})
            used_items.add(closest_item)

    # Keywords with no associated number default to quantity 1
    for _, item in items:
        if item not in used_items:
            results.append({"item": item, "quantity": 1})
            used_items.add(item)

    return results
