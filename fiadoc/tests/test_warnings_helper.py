import warnings

import pytest

from fiadoc.tests._warnings import assert_warnings


def test_required_pattern_matched():
    with assert_warnings(required=['hello world']):
        warnings.warn('hello world from parser', UserWarning)


def test_required_pattern_missing_fails():
    with pytest.raises(BaseException) as excinfo:
        with assert_warnings(required=['never emitted']):
            pass
    assert 'never emitted' in str(excinfo.value)


def test_unexpected_warning_fails_with_location():
    with pytest.raises(BaseException) as excinfo:
        with assert_warnings():
            warnings.warn('surprise', UserWarning)
    msg = str(excinfo.value)
    assert 'surprise' in msg
    assert 'UserWarning' in msg
    assert 'test_warnings_helper.py' in msg


def test_allowed_pattern_tolerates_but_does_not_require():
    with assert_warnings(allowed=['maybe']):
        pass
    with assert_warnings(allowed=['maybe']):
        warnings.warn('maybe later', UserWarning)


def test_category_filter_passes_other_classes_through():
    with assert_warnings(required=['stay'], category=UserWarning):
        warnings.warn('deprecated thing', DeprecationWarning)
        warnings.warn('stay tuned', UserWarning)


def test_required_and_allowed_combined():
    with assert_warnings(required=['must'], allowed=['maybe']):
        warnings.warn('must happen', UserWarning)
        warnings.warn('maybe also', UserWarning)


def test_unexpected_alongside_required_still_fails():
    with pytest.raises(BaseException) as excinfo:
        with assert_warnings(required=['must']):
            warnings.warn('must happen', UserWarning)
            warnings.warn('unexpected too', UserWarning)
    assert 'unexpected too' in str(excinfo.value)
