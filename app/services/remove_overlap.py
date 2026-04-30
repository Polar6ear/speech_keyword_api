def remove_overlap(prev_text: str, current_text: str) -> str:
    """
    Remove overlapping suffix/prefix between two consecutive transcript chunks.

    Sliding windows cause the same spoken text to appear in multiple
    transcription results. This function finds the longest suffix of
    prev_text that matches a prefix of current_text (minimum 5 chars),
    then strips that overlap from current_text before returning.
    """
    prev_text = prev_text.strip()
    current_text = current_text.strip()

    if not prev_text:
        return current_text

    max_overlap = min(len(prev_text), len(current_text))
    best_overlap = 0

    for i in range(5, max_overlap + 1):
        if prev_text[-i:] == current_text[:i]:
            best_overlap = i

    new_text = current_text[best_overlap:]
    return new_text.strip()
