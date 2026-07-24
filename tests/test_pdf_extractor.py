from investigation_rag.extractors.pdf_extractor import extract_pdf_layout_blocks, pdf_layout_to_markdown


def test_pdf_layout_to_markdown_infers_headings_and_bullets():
    layout = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "bbox": [72, 72, 300, 90],
                        "spans": [{"text": "Investigation Summary", "size": 18, "flags": 16, "font": "Arial-Bold"}],
                    },
                    {
                        "bbox": [72, 112, 460, 126],
                        "spans": [{"text": "This paragraph remains normal text.", "size": 10, "flags": 0, "font": "Arial"}],
                    },
                    {
                        "bbox": [90, 138, 460, 152],
                        "spans": [{"text": "\u2022 First finding", "size": 10, "flags": 0, "font": "Arial"}],
                    },
                ],
            }
        ]
    }

    markdown = pdf_layout_to_markdown(layout)

    assert markdown == "# Investigation Summary\n\nThis paragraph remains normal text.\n\n- First finding"


def test_extract_pdf_layout_blocks_preserves_relative_coordinates_and_spans():
    layout = {
        "width": 600,
        "height": 800,
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "bbox": [60, 80, 300, 96],
                        "spans": [{"text": "Investigation Summary", "size": 18, "flags": 18, "font": "Arial-BoldItalic"}],
                    }
                ],
            }
        ],
    }

    blocks = extract_pdf_layout_blocks(layout, page_number=2)

    assert blocks[0]["text"] == "Investigation Summary"
    assert blocks[0]["page_number"] == 2
    assert blocks[0]["rel_bbox"] == [0.1, 0.1, 0.5, 0.12]
    assert blocks[0]["bold"] is True
    assert blocks[0]["italic"] is True
    assert blocks[0]["spans"][0]["font_size"] == 18


def test_extract_pdf_layout_blocks_restores_spaces_between_visually_separated_spans():
    layout = {
        "width": 600,
        "height": 800,
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "bbox": [60, 80, 300, 96],
                        "spans": [
                            {"text": "Rapid", "bbox": [60, 80, 88, 96], "size": 10, "flags": 0, "font": "Times"},
                            {"text": "progress", "bbox": [92, 80, 135, 96], "size": 10, "flags": 0, "font": "Times"},
                            {"text": "in", "bbox": [139, 80, 148, 96], "size": 10, "flags": 0, "font": "Times"},
                            {"text": "AI", "bbox": [152, 80, 164, 96], "size": 10, "flags": 0, "font": "Times"},
                        ],
                    },
                    {
                        "bbox": [60, 110, 150, 126],
                        "spans": [
                            {"text": "Dario Amodei", "bbox": [60, 110, 124, 126], "size": 10, "flags": 0, "font": "Times"},
                            {"text": "∗", "bbox": [125, 108, 130, 118], "size": 7, "flags": 0, "font": "Times"},
                        ],
                    },
                ],
            }
        ],
    }

    blocks = extract_pdf_layout_blocks(layout, page_number=1)
    markdown = pdf_layout_to_markdown(layout)

    assert blocks[0]["text"] == "Rapid progress in AI"
    assert blocks[1]["text"] == "Dario Amodei∗"
    assert "Rapid progress in AI" in markdown


def test_extract_pdf_layout_blocks_prefers_word_boundaries_when_spans_are_collapsed():
    layout = {
        "width": 600,
        "height": 800,
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "bbox": [60, 80, 260, 96],
                        "spans": [
                            {"text": "For", "bbox": [60, 80, 76, 96], "size": 10, "flags": 0, "font": "Times"},
                            {"text": "concreteness,", "bbox": [76, 80, 130, 96], "size": 10, "flags": 0, "font": "Times"},
                            {"text": "we", "bbox": [130, 80, 140, 96], "size": 10, "flags": 0, "font": "Times"},
                            {"text": "will", "bbox": [140, 80, 154, 96], "size": 10, "flags": 0, "font": "Times"},
                        ],
                    }
                ],
            }
        ],
    }
    words = [
        (60, 80, 76, 96, "For", 0, 0, 0),
        (80, 80, 130, 96, "concreteness,", 0, 0, 1),
        (134, 80, 144, 96, "we", 0, 0, 2),
        (148, 80, 162, 96, "will", 0, 0, 3),
    ]

    blocks = extract_pdf_layout_blocks(layout, page_number=1, words=words)
    markdown = pdf_layout_to_markdown(layout, words=words)

    assert blocks[0]["text"] == "For concreteness, we will"
    assert markdown == "For concreteness, we will"


def test_pdf_layout_to_markdown_leaves_long_sentences_as_text():
    layout = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "bbox": [72, 72, 460, 86],
                        "spans": [
                            {"text": "This important sentence is bold but should remain prose.", "size": 10, "flags": 16, "font": "Arial-Bold"}
                        ],
                    }
                ],
            }
        ]
    }

    markdown = pdf_layout_to_markdown(layout)

    assert markdown == "This important sentence is bold but should remain prose."


def test_pdf_layout_to_markdown_merges_numbered_section_heading_and_skips_page_number():
    layout = {
        "width": 600,
        "height": 800,
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {"bbox": [90, 120, 100, 136], "spans": [{"text": "2", "size": 14, "flags": 16, "font": "Arial-Bold"}]},
                    {
                        "bbox": [115, 120, 320, 136],
                        "spans": [{"text": "Overview of Research Problems", "size": 14, "flags": 16, "font": "Arial-Bold"}],
                    },
                    {"bbox": [90, 160, 440, 174], "spans": [{"text": "This is normal body text.", "size": 10, "flags": 0, "font": "Arial"}]},
                    {"bbox": [296, 720, 304, 732], "spans": [{"text": "2", "size": 10, "flags": 0, "font": "Arial"}]},
                ],
            }
        ],
    }

    markdown = pdf_layout_to_markdown(layout)

    assert markdown == "## 2 Overview of Research Problems\n\nThis is normal body text."


def test_pdf_layout_to_markdown_finds_same_size_centered_heading():
    layout = {
        "width": 600,
        "height": 800,
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {"bbox": [80, 100, 520, 114], "spans": [{"text": "Keywords: safety, systems", "size": 11, "flags": 0, "font": "Times"}]},
                    {"bbox": [278, 130, 322, 144], "spans": [{"text": "Abstract", "size": 11, "flags": 0, "font": "Times"}]},
                    {"bbox": [90, 160, 500, 174], "spans": [{"text": "The body begins here.", "size": 11, "flags": 0, "font": "Times"}]},
                ],
            }
        ],
    }

    markdown = pdf_layout_to_markdown(layout)

    assert markdown == "Keywords: safety, systems\n\n### Abstract\n\nThe body begins here."


def test_pdf_layout_to_markdown_does_not_promote_affiliation_lines():
    layout = {
        "width": 600,
        "height": 800,
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "bbox": [55, 120, 520, 134],
                        "spans": [{"text": "1MIT FutureTech, Massachusetts Institute of Technology, Cambridge, MA", "size": 11, "flags": 0, "font": "Arial"}],
                    },
                    {
                        "bbox": [55, 160, 520, 174],
                        "spans": [{"text": "2School of Psychology, The University of Queensland", "size": 11, "flags": 0, "font": "Arial"}],
                    },
                ],
            }
        ],
    }

    markdown = pdf_layout_to_markdown(layout)

    assert markdown == (
        "1MIT FutureTech, Massachusetts Institute of Technology, Cambridge, MA\n\n"
        "2 School of Psychology, The University of Queensland"
    )


def test_pdf_layout_to_markdown_reflows_two_column_taxonomy_rows():
    layout = {
        "width": 840,
        "height": 595,
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {"bbox": [62, 262, 188, 275], "spans": [{"text": "5.2 Loss of human agency and", "size": 9, "flags": 0, "font": "Arial"}]},
                    {"bbox": [192, 262, 769, 275], "spans": [{"text": "Delegating by humans of key decisions to AI systems", "size": 9, "flags": 0, "font": "Arial"}]},
                    {"bbox": [78, 273, 120, 286], "spans": [{"text": "autonomy", "size": 9, "flags": 0, "font": "Arial"}]},
                    {"bbox": [192, 273, 705, 286], "spans": [{"text": "leading to humans feeling disempowered", "size": 9, "flags": 0, "font": "Arial"}]},
                    {"bbox": [62, 312, 180, 325], "spans": [{"text": "6.1 Power centralization and", "size": 9, "flags": 0, "font": "Arial"}]},
                    {"bbox": [192, 312, 753, 325], "spans": [{"text": "AI-driven concentration of power and resources", "size": 9, "flags": 0, "font": "Arial"}]},
                    {"bbox": [78, 323, 161, 336], "spans": [{"text": "unfair distribution of", "size": 9, "flags": 0, "font": "Arial"}]},
                    {"bbox": [78, 334, 111, 347], "spans": [{"text": "benefits", "size": 9, "flags": 0, "font": "Arial"}]},
                ],
            }
        ],
    }

    markdown = pdf_layout_to_markdown(layout)

    assert "### 5.2 Loss of human agency and autonomy" in markdown
    assert "Delegating by humans of key decisions to AI systems" in markdown
    assert "### 6.1 Power centralization and unfair distribution of benefits" in markdown
