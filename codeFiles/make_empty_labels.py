from pathlib import Path
IMG_EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
ROOT=Path(__file__).resolve().parents[1]
for split in ['train','valid','test']:
    img_dir=ROOT/split/'images'; lab_dir=ROOT/split/'labels'; lab_dir.mkdir(parents=True, exist_ok=True)
    if not img_dir.exists(): continue
    made=0
    for img in img_dir.rglob('*'):
        if img.suffix.lower() in IMG_EXTS:
            lab=lab_dir/(img.stem+'.txt')
            if not lab.exists():
                lab.write_text('', encoding='utf-8'); made+=1
    print(f'{split}: created {made} empty label files')
