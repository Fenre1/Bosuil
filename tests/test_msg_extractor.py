from investigation_rag.extractors.msg_extractor import html_to_markdown, markdown_to_plain_text


def test_html_email_body_converts_basic_structure_to_markdown():
    markdown = html_to_markdown("<h1>Status</h1><p>Hello <strong>there</strong></p><ul><li>One</li></ul>")

    assert "# Status" in markdown
    assert "Hello **there**" in markdown
    assert "- One" in markdown


def test_markdown_email_body_can_fallback_to_plain_search_text():
    plain = markdown_to_plain_text("# Status\n\nHello **there**\n\n- One")

    assert plain == "Status\n\nHello there\nOne"
