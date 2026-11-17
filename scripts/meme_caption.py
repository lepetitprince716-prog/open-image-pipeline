#!/usr/bin/env python3
"""Overlay classic top/bottom meme captions (Impact, white fill + black outline).

A meme is image + caption; SD garbles in-image text, so the readable joke is added
here in post. Writes "<name>.meme.png" next to the input (original untouched).

Usage: python3 scripts/meme_caption.py <png> --top "TEXT" --bottom "TEXT"
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FONT="/System/Library/Fonts/Supplemental/Impact.ttf"

def fit_font(draw, text, max_w, start):
    size=start
    while size>16:
        f=ImageFont.truetype(FONT, size)
        if draw.textlength(text, font=f) <= max_w: return f
        size-=4
    return ImageFont.truetype(FONT, 16)

def wrap(draw, text, font, max_w):
    words=text.split(); lines=[]; cur=""
    for w in words:
        t=(cur+" "+w).strip()
        if draw.textlength(t, font=font) <= max_w: cur=t
        else: lines.append(cur); cur=w
    if cur: lines.append(cur)
    return lines

def draw_block(img, text, where):
    if not text: return
    d=ImageDraw.Draw(img); W,H=img.size
    text=text.upper(); margin=int(W*0.04); max_w=W-2*margin
    font=fit_font(d, text, max_w, int(H*0.10))
    lines=wrap(d, text, font, max_w)
    lh=font.size*1.12; stroke=max(2,font.size//12)
    total=lh*len(lines)
    y=int(H*0.02) if where=="top" else H-total-int(H*0.03)
    for ln in lines:
        w=d.textlength(ln, font=font); x=(W-w)/2
        d.text((x,y), ln, font=font, fill="white", stroke_width=stroke, stroke_fill="black")
        y+=lh

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--top",default=""); ap.add_argument("--bottom",default="")
    a=ap.parse_args()
    p=Path(a.image); img=Image.open(p).convert("RGB")
    draw_block(img, a.top, "top"); draw_block(img, a.bottom, "bottom")
    out=p.with_suffix(".meme.png"); img.save(out)
    print("->", out.name)
