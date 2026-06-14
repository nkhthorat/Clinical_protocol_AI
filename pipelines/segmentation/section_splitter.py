import re
from typing import List, Dict


def split_protocol_sections(text: str) -> List[Dict[str, str]]:
    """
    Split protocol text into sections using numbered headings.

    Returns:
        [
            {
                "section": section_title,
                "text": section_content
            }
        ]
    """

    heading_pattern = (
        r'\n\s*(\d+(?:\.\d+)*)\s+'
        r'([A-Z][A-Za-z0-9\s\-/(),]+)\n'
    )

    matches = list(re.finditer(heading_pattern, text))

    sections = []

    for i, match in enumerate(matches):

        section_number = match.group(1)
        section_title = match.group(2).strip()

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        section_text = text[start:end].strip()

        sections.append(
            {
                "section": f"{section_number} {section_title}",
                "text": section_text
            }
        )

    return sections