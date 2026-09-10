"""Leila: presentation-only dataset settings, separate from detector inputs."""
import re


def describe_dataset(source, samples=None):
    name = re.split(r'[\\/]', str(source))[-1]
    header = re.match(r'^(cifar10|mnist|imdb)-(train|test)-([^-]+)-', name)
    if not header:
        return {'description': 'Dataset information unavailable for this saved input.'}
    dataset, split, size = header.groups()
    attack = re.search(r'-(targeted_label_flip|label_flip|blended_injection|backdoor|none)-', name)
    kind = attack[1] if attack else 'unknown'
    titles = {'none':'Clean data', 'label_flip':'Next-class label flip',
              'targeted_label_flip':'Targeted label flip', 'blended_injection':'Blended noise injection',
              'backdoor':'Backdoor phrase' if dataset=='imdb' else 'Backdoor patch'}
    parts = [{'cifar10':'CIFAR-10','mnist':'MNIST','imdb':'IMDB'}[dataset], titles.get(kind,'Attack unspecified')]
    count = re.search(r'-n(\d+)-', name)
    rate = re.search(r'-(\d{3})-seed', name)
    if kind not in ('none','unknown'):
        if count:
            parts.append(f'{int(count[1]):,} attacked samples')
            if samples: parts.append(f'{100*int(count[1])/samples:g}% poisoning')
        elif rate: parts.append(f'{int(rate[1])}% poisoning')
    if samples is not None:
        parts.append(f'{samples:,} original ' + ('reviews' if dataset=='imdb' else 'images'))
    elif size.isdigit(): parts.append(f'{int(size):,} samples')
    if size=='full': parts.append(f'Full {"training" if split=="train" else "test"} dataset')
    target=re.search(r'-t(\d+)-',name); source_label=re.search(r'-s(\d+)-',name)
    if kind=='targeted_label_flip' and target and source_label:
        parts.append(f'Label {source_label[1]} → {target[1]}')
    elif kind in ('backdoor','blended_injection') and target:
        parts.append(f'Target label {target[1]}')
    alpha=re.search(r'-a([\d.]+)-',name)
    if kind=='blended_injection' and alpha: parts.append(f'Blend strength {alpha[1]}')
    seed=re.search(r'-seed(\d+)',name)
    if seed: parts.append(f'Attack seed {seed[1]}')
    return {'description':' · '.join(parts), 'dataset':dataset, 'attack':kind,
            'provenance':'Saved dataset filename and recorded sample count'}
