from pipeline.oe import parse_folder


def _entry(file_id, title):
    return (f'<div class="flip-entry" id="entry-{file_id}"><a href="https://drive.google.com/'
            f'file/d/{file_id}/view?usp=drive_web"><div class="flip-entry-title">{title}</div></a></div>')


def test_parse_folder_pairs_ids_with_year_titles():
    html = (_entry("id2025", "2025_LoL_esports_match_data_from_OraclesElixir.csv")
            + _entry("idOther", "readme.txt")
            + _entry("id2026", "2026_LoL_esports_match_data_from_OraclesElixir.csv"))
    assert parse_folder(html) == {2025: "id2025", 2026: "id2026"}
