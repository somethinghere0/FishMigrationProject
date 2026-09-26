from __future__ import annotations
from pathlib import Path
import argparse

IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
ROOT = Path(__file__).resolve().parents[1]

def count_split(split):
    img_dir = ROOT / split / 'images'
    lab_dir = ROOT / split / 'labels'
    imgs = sorted([p for p in img_dir.rglob('*') if p.suffix.lower() in IMG_EXTS]) if img_dir.exists() else []
    labs = sorted([p for p in lab_dir.rglob('*.txt')]) if lab_dir.exists() else []
    missing = []
    empty = 0
    nonempty = 0
    bad = []
    for img in imgs:
        lab = lab_dir / (img.stem + '.txt')
        if not lab.exists():
            missing.append(img.name)
        else:
            txt = lab.read_text(encoding='utf-8').strip()
            if txt:
                nonempty += 1
                for i,line in enumerate(txt.splitlines(),1):
                    parts=line.split()
                    if len(parts)!=5:
                        bad.append((lab.name,i,line))
                    else:
                        try:
                            cls,x,y,w,h=parts
                            vals=[float(x),float(y),float(w),float(h)]
                            if int(cls)!=0 or any(v<0 or v>1 for v in vals):
                                bad.append((lab.name,i,line))
                        except Exception:
                            bad.append((lab.name,i,line))
            else:
                empty += 1
    return {'split':split,'images':len(imgs),'labels':len(labs),'missing':missing,'empty_labels':empty,'nonempty_labels':nonempty,'bad':bad}

def main():
    for split in ['train','valid','test']:
        s=count_split(split)
        print(f"\n[{split}]")
        print(f"images: {s['images']} | labels: {s['labels']} | non-empty labels: {s['nonempty_labels']} | empty labels: {s['empty_labels']}")
        if s['missing']:
            print(f"missing labels for {len(s['missing'])} images, first 10: {s['missing'][:10]}")
        if s['bad']:
            print(f"bad label lines: {len(s['bad'])}, first 5: {s['bad'][:5]}")
    print('\nDone. Empty .txt files are OK for no-fish frames. Missing .txt files should be created as empty files.')

if __name__=='__main__':
    main()
