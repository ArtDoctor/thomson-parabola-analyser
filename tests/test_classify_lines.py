from oblisk.analysis.classification import classify_lines


def test_classify_lines_unique_species_per_parabola() -> None:
    """If two parabolae match the same species within tolerance, only the closer one keeps that label."""
    a_H = 100.0
    a_list = [100.0, 101.0]
    match_tol = 0.03
    out = classify_lines(a_list, a_H, match_tol)
    assert len(out) == 2
    labels = [str(row['label']) for row in out]
    assert labels.count('H^1+') == 1
    assert labels.count('?') == 1
    assert out[0]['label'] == 'H^1+'
    assert out[1]['label'] == '?'


def test_classify_lines_no_conflict_when_different_species() -> None:
    a_H = 100.0
    r_c2 = 5.956656587174187
    a_list = [100.0, r_c2 * a_H]
    match_tol = 0.03
    out = classify_lines(a_list, a_H, match_tol)
    assert out[0]['label'] == 'H^1+'
    assert out[1]['label'] == 'C^2+'
