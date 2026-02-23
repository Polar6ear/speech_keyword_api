def remove_overlap(prev_text, current_text):
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