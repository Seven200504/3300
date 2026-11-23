"""Lossless CSV bundles; readers accept both loose outputs and bundled tables."""
from pathlib import Path
import io
import zipfile
import pandas as pd

def read_csv(path, *args, **kwargs):
    if isinstance(path, (str, Path)):
        p = Path(path)
        bundle = p.parent / 'tables.zip'
        if not p.exists() and p.suffix == '.csv' and bundle.exists():
            with zipfile.ZipFile(bundle) as z:
                return pd.read_csv(io.BytesIO(z.read(p.name)), *args, **kwargs)
    return pd.read_csv(path, *args, **kwargs)

def consolidate(root):
    for folder in (Path(root) / 'output').glob('r[567]'):
        files = sorted(folder.glob('*.csv'))
        if not files:
            continue
        target = folder / 'tables.zip'
        contents = {}
        if target.exists():
            with zipfile.ZipFile(target) as z:
                contents.update({n: z.read(n) for n in z.namelist()})
        contents.update({p.name: p.read_bytes() for p in files})
        temp = folder / 'tables.tmp.zip'
        with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, data in sorted(contents.items()):
                z.writestr(name, data)
        with zipfile.ZipFile(temp) as z:
            assert z.testzip() is None
            assert all(z.read(name) == data for name, data in contents.items())
        temp.replace(target)
        for p in files:
            p.unlink()
        print(f'{folder.name}: {len(contents)} CSV tables consolidated into tables.zip')

if __name__ == '__main__':
    consolidate(Path(__file__).resolve().parents[1])
