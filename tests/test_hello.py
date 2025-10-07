from tools.see_the_errors import report_error

def test_example():
    assert 1 + 1 == 2

def test_error_reporting():
    try:
        assert 1 + 1 == 3
    except AssertionError as e:
        report_error(e)