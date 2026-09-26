"""Convert Roboflow-style COCO JSON annotations into YOLO txt labels.
Use only if labels/*.txt are missing but _annotations.coco.json exists.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def convert_split(split: str):
    split_dir=ROOT/split
    json_path=split_dir/'_annotations.coco.json'
    img_dir=split_dir/'images'
    lab_dir=split_dir/'labels'
    lab_dir.mkdir(parents=True, exist_ok=True)
    if not json_path.exists():
        print(f'{split}: no {json_path}, skipped')
        return
    data=json.loads(json_path.read_text(encoding='utf-8'))
    images={img['id']:img for img in data.get('images',[])}
    anns_by_img={img_id:[] for img_id in images}
    for ann in data.get('annotations',[]):
        anns_by_img.setdefault(ann['image_id'],[]).append(ann)
    for img_id,img in images.items():
        file_name=Path(img['file_name']).name
        w=float(img['width']); h=float(img['height'])
        lines=[]
        for ann in anns_by_img.get(img_id,[]):
            # force all fish categories to class 0
            x,y,bw,bh=ann['bbox']
            xc=(x+bw/2)/w; yc=(y+bh/2)/h; nw=bw/w; nh=bh/h
            # clamp
            vals=[max(0,min(1,v)) for v in [xc,yc,nw,nh]]
            lines.append('0 ' + ' '.join(f'{v:.6f}' for v in vals))
        (lab_dir/(Path(file_name).stem+'.txt')).write_text('\n'.join(lines), encoding='utf-8')
    print(f'{split}: wrote YOLO labels for {len(images)} images')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['train','valid','test'])
    args=ap.parse_args()
    for s in args.splits: convert_split(s)
if __name__=='__main__': main()
