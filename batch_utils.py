import io
import zipfile
import csv
from typing import Dict, List, Tuple


def parse_labels_csv(content: bytes) -> Dict[str, int]:
    """Parse CSV bytes of form filename,label and return a mapping.

    Raises ValueError on parse errors.
    """
    s = content.decode('utf-8')
    reader = csv.reader(s.splitlines())
    mapping = {}
    for row in reader:
        if len(row) < 2:
            continue
        fname = row[0].strip()
        try:
            lbl = int(row[1].strip())
        except Exception as e:
            raise ValueError(f"Invalid label for {fname}: {e}")
        mapping[fname] = lbl
    return mapping


def expand_zip_bytes(zip_bytes: bytes) -> List[Tuple[str, bytes]]:
    """Return list of (filename, bytes) entries for images in the zip.

    Only extracts files with common image extensions.
    """
    out = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for zi in z.infolist():
            if zi.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                out.append((zi.filename, z.read(zi)))
    return out
