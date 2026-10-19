import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit


def normalize_url(url):
    """Same URL cleaning as the training notebook."""
    url = unicodedata.normalize("NFKC", str(url).strip())
    url = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", url)
    if "://" not in url:
        url = "http://" + url
    try:
        p = urlsplit(url)
        host = (p.hostname or "").lower().encode("idna").decode("ascii")
        if p.port:
            host += ":" + str(p.port)
        return urlunsplit((p.scheme.lower(), host, p.path, p.query, p.fragment))
    except Exception:
        return url


def normalize_frame(frame):
    """Pipeline step: clean the URL column of a DataFrame."""
    frame = frame.copy()
    frame["URL"] = frame["URL"].map(normalize_url)
    return frame