"""Reuse completed local scans only with matching inputs, code and settings."""
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re


def digest_file(path):
    digest = sha256()
    before = path.stat()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
        raise ValueError('An input changed while checking saved scans. Retry when the build finishes.')
    return digest.hexdigest()


def scan_identity(paths, profile):
    inputs = []
    for feature in paths:
        for path in ((feature,) if feature.name.startswith('imdb-train-') else (feature,feature.with_name(feature.name.replace('-features.npz','-images.npz')))):
            inputs.append((str(path.resolve()),digest_file(path)))
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root/'detectors'/'label_flip').glob('*.py'))
    sources += [root/'detectors'/'output_connector.py']
    sources += sorted((root/'poison_features').glob('*.py'))
    # Leila: the MNIST adapter reuses the tested pixel experiment's scoring functions.
    sources += [root/'experiments'/'scan_mnist_pixels.py']
    identity = dict(schema=1, inputs=inputs, profile=profile, python=platform.python_version(),
        packages={name:version(name) for name in ('numpy','scipy','scikit-learn','cleanlab','threadpoolctl')},
        code=[(str(path.relative_to(root)),digest_file(path)) for path in sources])
    return sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def save_cache_record(output_dir, identity):
    # Leila: write completion metadata last; failed/partial scans never become cache hits.
    record = dict(identity=identity,result_sha256=digest_file(output_dir/'results.json'))
    temporary = output_dir/'cache.tmp'
    temporary.write_text(json.dumps(record),encoding='utf-8')
    temporary.replace(output_dir/'cache.json')


def find_cached_scan(feature_file):
    from .web_scan import ARTIFACTS, feature_pair, web_profile
    root = ARTIFACTS/'label_flip_scans'
    candidates = list(root.glob('*/cache.json'))
    if not candidates: return None
    # Leila: use the MNIST pixel profile for MNIST cache identity.
    paths = feature_pair(feature_file)
    if paths[0].name.startswith('imdb-train-'):
        from .web_imdb import profile
        config = profile()
    elif len(paths) == 1:
        from .web_mnist import profile
        config = profile()
    else:
        config = web_profile()
    identity = scan_identity(paths,config)
    for path in sorted(candidates,key=lambda p:p.stat().st_mtime_ns,reverse=True):
        if not re.fullmatch('[0-9a-f]{32}',path.parent.name) or path.resolve().parent.parent != root.resolve():
            continue
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            result = path.with_name('results.json')
            if (record['identity'] == identity and result.is_file()
                    and digest_file(result) == record['result_sha256']):
                return path.parent.name
        except (OSError,ValueError,KeyError,TypeError):
            continue
    return None
