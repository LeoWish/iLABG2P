"""Interactive Windows/Linux launcher. All activities use the current Python environment."""
from __future__ import annotations
import os
import subprocess
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*args):
    print('\n[RUN]', sys.executable, *args, flush=True)
    r = subprocess.run([sys.executable, *args], cwd=ROOT)
    print('[EXIT]', r.returncode, flush=True)
    return r.returncode


def example():
    src = ROOT / 'examples' / 'DEMO_SYNTHETIC_NOT_BIOLOGICAL.fna'
    dst = ROOT / 'test_data' / src.name
    if dst.exists() and dst.read_bytes() != src.read_bytes():
        print('[BLOCKED] Example destination exists with different content:', dst)
        return
    shutil.copy2(src, dst)
    try:
        run('test.py')
        pred = ROOT / 'results' / 'predictions' / 'New_Genome_Predictions.csv'
        if pred.exists():
            shutil.copy2(pred, ROOT/'examples'/'DEMO_Predictions.csv')
            print('This is a SOFTWARE smoke test, NOT biologically meaningful predictions.')
    finally:
        dst.unlink(missing_ok=True)


def main():
    os.chdir(ROOT)
    while True:
        print('''\n=== iLABG2P | Windows / Linux ===
1. Predict new FASTA genomes from test_data (default Top1)
2. Run a synthetic example (software function only)
3. Check package, frozen folds and saved models
4. Smoke test (small end-to-end run)
5. Full run: 16,780 configurations + reports + Supplementary Material
6. Refit Top10 from existing v2 OOF results
0. Exit\n''')
        option = input('Select / 请选择: ').strip()
        if option == '0':
            break
        elif option == '1':
            run('test.py')
        elif option == '2':
            example()
        elif option == '3':
            run('verify_package.py')
        elif option == '4':
            run('run.py', '--smoke')
        elif option == '5':
            yes = input('Full 16,780-configuration OOF training may take many hours. Type YES: ')
            if yes == 'YES':
                run('run.py')
        elif option == '6':
            run('run.py', '--refit-only')
        else:
            print('Invalid selection.')


if __name__ == '__main__':
    main()
