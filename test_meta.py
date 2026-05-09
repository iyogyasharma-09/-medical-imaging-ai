import sys
sys.path.insert(0, '.')
from pathlib import Path
from config import BRATS2023_TRAIN_MANIFEST, BRATS2023_VAL_MANIFEST, BRATS2023_TEST_MANIFEST
from imported_files.brats2023_dataset import load_manifest

for split, path in [('train', BRATS2023_TRAIN_MANIFEST),
                    ('val',   BRATS2023_VAL_MANIFEST),
                    ('test',  BRATS2023_TEST_MANIFEST)]:
    df = load_manifest(path, label_col='label_grade')
    df_gli = df[df['task'] == 'GLI'].copy()
    path_col = next((c for c in ['data_path', 'subject_dir', 'path']
                     if c in df_gli.columns), None)
    missing = df_gli[path_col].apply(
        lambda p: not Path(str(p)).exists()
    ).sum() if path_col else 0
    n_lgg = int((df_gli['label_grade'] == 0).sum())
    n_hgg = int((df_gli['label_grade'] == 1).sum())
    print(f"{split}: n={len(df_gli)}  LGG={n_lgg}  HGG={n_hgg}  missing={missing}  path_col={path_col}")

print("Metadata check OK")
