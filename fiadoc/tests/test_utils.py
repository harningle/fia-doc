import pytest

from fiadoc.utils import download_pdf


def test_download_pdf_saves_valid_pdf(requests_mock, tmp_path, monkeypatch):
    monkeypatch.setenv('FIADOC_CACHE_DIR', str(tmp_path / 'cache'))
    url = 'https://example.com/test.pdf'
    pdf = b'%PDF-1.4\n1 0 obj\n<<>>\nendobj\n'
    requests_mock.get(url, content=pdf, headers={'Content-Type': 'application/pdf'})

    out_path = tmp_path / 'doc.pdf'
    download_pdf(url, out_path)

    assert out_path.read_bytes() == pdf
    cached = list((tmp_path / 'cache' / 'downloads').glob('*.pdf'))
    assert len(cached) == 1
    assert cached[0].read_bytes() == pdf


def test_download_pdf_uses_cache(requests_mock, tmp_path, monkeypatch):
    monkeypatch.setenv('FIADOC_CACHE_DIR', str(tmp_path / 'cache'))
    url = 'https://example.com/test.pdf'
    pdf = b'%PDF-1.4\n1 0 obj\n<<>>\nendobj\n'
    requests_mock.get(url, content=pdf, headers={'Content-Type': 'application/pdf'})

    first = tmp_path / 'first.pdf'
    second = tmp_path / 'second.pdf'
    download_pdf(url, first)
    requests_mock.reset()

    download_pdf(url, second)

    assert first.read_bytes() == pdf
    assert second.read_bytes() == pdf


def test_download_pdf_rejects_non_pdf(requests_mock, tmp_path, monkeypatch):
    monkeypatch.setenv('FIADOC_CACHE_DIR', str(tmp_path / 'cache'))
    url = 'https://example.com/test.pdf'
    requests_mock.get(url, content=b'<html>blocked</html>', headers={'Content-Type': 'text/html'})

    with pytest.raises(ValueError, match='Expected a PDF'):
        download_pdf(url, tmp_path / 'doc.pdf')
