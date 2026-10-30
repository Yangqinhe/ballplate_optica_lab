"""Display-only CCD readout geometry; never alter acquisition or actor inputs."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REGION_COLORS = ("#ffcd57", "#ff894f", "#ff6396", "#cb8cff",
                 "#58d9ed", "#65b0ff", "#73e69a", "#c8e76b")
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def checked_regions(shape, regions):
    h, w = shape
    checked = []
    for region in regions:
        if len(region) != 4 or any(int(v) != v for v in region):
            raise ValueError("integer half-open [y0,y1,x0,x1] regions required")
        y0, y1, x0, x1 = map(int, region)
        if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
            raise ValueError("CCD readout region outside the native image")
        checked.append((y0, y1, x0, x1))
    if not checked:
        raise ValueError("readout geometry is empty")
    return checked


def native_regions(shape, corrected_regions):
    """Undo the decoder's 180-degree rotation, preserving half-open boundaries."""
    h, w = shape
    return [(h-y1, h-y0, w-x1, w-x0)
            for y0, y1, x0, x1 in checked_regions(shape, corrected_regions)]


def readout_statistics(raw, corrected_regions, reference):
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.dtype != np.uint8:
        raise ValueError("native Mono8 image required")
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("positive fixed reference required")
    regions = native_regions(raw.shape, corrected_regions)
    mask = np.zeros(raw.shape, dtype=bool)
    sums = []
    for y0, y1, x0, x1 in regions:
        sums.append(int(raw[y0:y1, x0:x1].sum(dtype=np.uint64)))
        mask[y0:y1, x0:x1] = True
    total = int(raw.sum(dtype=np.uint64))
    inside = int(raw[mask].sum(dtype=np.uint64))
    return dict(native_regions=regions, region_counts=sums,
                features=np.asarray(sums, dtype=np.float32)/np.float32(reference),
                total_counts=total, inside_counts=inside,
                count_fraction=inside/total if total else None,
                unique_pixels=int(mask.sum()), area_fraction=float(mask.mean()))


def render_ccd(raw, corrected_regions, gain=1., size=560, show_regions=True,
               labels=True, corrected_view=False):
    """Return a marked RGB preview; all painting is on a newly allocated image."""
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.dtype != np.uint8:
        raise ValueError("native Mono8 image required")
    if not np.isfinite(gain) or gain <= 0 or int(size) != size or size < 32:
        raise ValueError("positive gain and integer preview size >=32 required")
    regions = (checked_regions(raw.shape, corrected_regions) if corrected_view
               else native_regions(raw.shape, corrected_regions))
    values = raw[::-1, ::-1] if corrected_view else raw
    values = np.clip(values.astype(np.float32)*gain, 0, 255).astype(np.uint8)
    image = Image.fromarray(values).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.BOX)
    canvas = Image.new("RGB", (size, size), "#08101b")
    ox, oy = (size-image.width)//2, (size-image.height)//2
    canvas.paste(image, (ox, oy))
    if not show_regions:
        return np.asarray(canvas)
    draw = ImageDraw.Draw(canvas)
    sx, sy = image.width/raw.shape[1], image.height/raw.shape[0]
    font_size = 12 if size >= 400 else 8
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except OSError:
        font = ImageFont.load_default()
    for index, (y0, y1, x0, x1) in enumerate(regions):
        color = REGION_COLORS[index % len(REGION_COLORS)]
        left, top = ox+math.floor(x0*sx), oy+math.floor(y0*sy)
        right = max(left, ox+math.ceil(x1*sx)-1)
        bottom = max(top, oy+math.ceil(y1*sy)-1)
        draw.rectangle((left, top, right, bottom), outline=color,
                       width=2 if size >= 400 else 1)
        if labels and size >= 400:
            text = f"R{index+1}"
            draw.text((left+3, top+2), text, fill=color, font=font,
                      stroke_width=1, stroke_fill="#08101b")
    return np.asarray(canvas)
