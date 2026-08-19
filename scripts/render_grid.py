"""Render the prediction grid for the portfolio card.

Nine held-out faces, sampled with a fixed seed across the age range so the grid
represents the validation set rather than a flattering corner of it. Each cell
shows the original photo with the predicted and actual age and gender beneath.
"""
import os, json, numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

HERE  = os.path.dirname(__file__)
SRC   = os.path.join(HERE, "..", "UTKFace", "images_flat", "part1")
ART   = os.path.join(HERE, "..", "artifacts")
OUT   = os.path.join(ART, "prediction-grid.png")

S, PAD, CAP = 900, 14, 50            # canvas, gutter, caption band
BG, FG, MUT = "#1b2236", "#eef2ff", "#8fa0c8"
OK, BAD     = "#4d8bff", "#ff6b6b"

fdir = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
F  = ImageFont.truetype(os.path.join(fdir, "DejaVuSans-Bold.ttf"), 21)
Fs = ImageFont.truetype(os.path.join(fdir, "DejaVuSans.ttf"), 17)
Fl = ImageFont.truetype(os.path.join(fdir, "DejaVuSans.ttf"), 13)

d = np.load(os.path.join(ART, "val_preds.npz"), allow_pickle=True)
ta, pa, tg, pg, paths = d["true_age"], d["pred_age"], d["true_gender"], d["pred_gender"], d["paths"]

# one face from each of nine age bands, so the grid spans the range the model is
# actually asked about instead of clustering on the dataset's young median
rng = np.random.default_rng(7)
bands = [(1, 5), (6, 12), (13, 19), (20, 26), (27, 34), (35, 44), (45, 54), (55, 69), (70, 116)]
pick = []
for lo, hi in bands:
    idx = np.where((ta >= lo) & (ta <= hi))[0]
    pick.append(int(rng.choice(idx)))

canvas = Image.new("RGB", (S, S), BG)
draw   = ImageDraw.Draw(canvas)
cell   = (S - PAD * 4) // 3
img_h  = cell - CAP

for n, i in enumerate(pick):
    r, c = divmod(n, 3)
    x = PAD + c * (cell + PAD)
    y = PAD + r * (cell + PAD)
    face = Image.open(os.path.join(SRC, str(paths[i]))).convert("RGB").resize((cell, img_h), Image.LANCZOS)
    canvas.paste(face, (x, y))

    # Each cell states its own prediction against the truth. No aggregate score and
    # no commentary - the numbers are what they are and the reader can read them.
    ty = y + img_h + 4
    g_pred = "M" if int(pg[i]) == 0 else "F"
    g_true = "M" if int(tg[i]) == 0 else "F"
    draw.text((x + 2, ty), "pred", font=Fl, fill=MUT)
    draw.text((x + 46, ty), f"{pa[i]:.0f}", font=F, fill=FG)
    draw.text((x + 46 + draw.textlength(f"{pa[i]:.0f}", font=F) + 5, ty + 3),
              g_pred, font=Fs, fill=FG)
    draw.text((x + 2, ty + 23), "actual", font=Fl, fill=MUT)
    draw.text((x + 46, ty + 22), f"{int(ta[i])}", font=Fs, fill=MUT)
    draw.text((x + 46 + draw.textlength(f"{int(ta[i])}", font=Fs) + 5, ty + 22),
              g_true, font=Fs, fill=MUT)

canvas.save(OUT)
m = json.load(open(os.path.join(ART, "metrics.json")))
print("wrote", OUT)
print("model", m["best"], "| MAE %.2f yrs | gender %.3f" % (m["results"][m["best"]]["mae"], m["results"][m["best"]]["acc"]))
for i in pick:
    print(f"  actual {int(ta[i]):3d} -> pred {pa[i]:5.1f}   gender {'MF'[int(tg[i])]} -> {'MF'[int(pg[i])]}")
