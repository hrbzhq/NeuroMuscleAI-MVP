import zipfile
import io
import os
from batch_utils import parse_labels_csv, expand_zip_bytes


def test_parse_labels_csv_ok():
    csv_bytes = b"img1.jpg,0\nimg2.png,1\n"
    mapping = parse_labels_csv(csv_bytes)
    assert mapping["img1.jpg"] == 0
    assert mapping["img2.png"] == 1


def test_parse_labels_csv_bad_label():
    csv_bytes = b"img1.jpg,abc\n"
    try:
        parse_labels_csv(csv_bytes)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_expand_zip_bytes(tmp_path):
    # create a zip in memory with two fake image files
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('a/img1.jpg', b'JPEGDATA')
        z.writestr('b/notimage.txt', b'IGNORE')
        z.writestr('img2.png', b'PNGDATA')
    entries = expand_zip_bytes(buf.getvalue())
    names = [e[0] for e in entries]
    assert 'a/img1.jpg' in names
    assert 'img2.png' in names
    assert all(isinstance(e[1], (bytes, bytearray)) for e in entries)
