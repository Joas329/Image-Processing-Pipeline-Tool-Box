import os
import cv2
import argparse
import numpy as np
import astroalign as aa

from PIL import Image
from glob import glob
from concurrent.futures import ProcessPoolExecutor

RAW_PATH = "/media/joas329/My Passport/OPTICAL/exp_149/chunk_010"
PNG_PATH = os.path.join(RAW_PATH, "pngs")
SAMPLES_PATH = os.path.join(PNG_PATH, "samples")

# per-worker globals: the background model, reference mask and pipeline state are
# shipped once per worker by the pool initializer instead of with every frame
_BG = None
_REF01 = None
_NSIGMA = 1.0
_REF_IDX = 0
_SAMPLE_IDX = 0
_SAMPLES_DIR = None

#####################
# Worker Pool State #
#####################
def _init_worker(background, ref01, nsigma, ref_idx, sample_idx, samples_dir):
    global _BG, _REF01, _NSIGMA, _REF_IDX, _SAMPLE_IDX, _SAMPLES_DIR
    _BG = background
    _REF01 = ref01
    _NSIGMA = nsigma
    _REF_IDX = ref_idx
    _SAMPLE_IDX = sample_idx
    _SAMPLES_DIR = samples_dir

###############
# RAW Loading #
###############
def load_flir_raw(path, width, height, pixel_format="BayerRG8"):
    data = np.fromfile(path, dtype=np.uint8) # keep uint8

    if pixel_format == "BayerRG8":
        expected_size = width * height
        if data.size != expected_size:
            raise ValueError(f"Expected {expected_size} bytes for a {width}x{height} image, but got {data.size} bytes.")
        return data.reshape((height, width))

    elif pixel_format == "RGB8":
        expected_size = width * height * 3
        if data.size != expected_size:
            raise ValueError(f"Expected {expected_size} bytes for a {width}x{height} image, but got {data.size} bytes.")
        return data.reshape((height, width, 3))

    raise ValueError(f"Unsupported pixel format: {pixel_format}")

#################
# Median Filter #
#################
def median_filter(frame, kernel_size=3):
    return cv2.medianBlur(frame, kernel_size)

####################
# Background Model #
####################
def build_background_model(raw_files, width, height, pixel_format="BayerRG8", n_frames=50, strip_rows=256):
    bg_files = raw_files[-n_frames:]
    print(f"Using {len(bg_files)} frames to build background model")

    # keep uint8; a float32 stack of 50 full frames is ~2.5 GB
    bg_stack = [median_filter(load_flir_raw(p, width, height, pixel_format)) for p in bg_files]
    stack = np.stack(bg_stack, axis=0) # (N, H, W) uint8
    background = np.empty(stack.shape[1:], dtype=np.float32)

    # median strip-by-strip so np.median never partitions the whole stack at once
    for y in range(0, stack.shape[1], strip_rows):
        y1 = min(y + strip_rows, stack.shape[1])
        background[y:y1] = np.median(stack[:, y:y1, :], axis=0)

    print(f"Background computed from {stack.shape[0]} frames")
    return background

###################################
# BG Subtract + Sigma Threshold #
###################################
def sigma_threshold(frame, background, nsigma):
    # subtract in float and keep negatives, so sky stays symmetric around 0 and n*sigma means real sigmas
    residual = frame.astype(np.float32) - background

    # frame is background-subtracted (sky ~ 0), so detection is simply
    # "n local noise sigmas above zero". per-pixel rms means a noisy region demands a proportionally taller peak, unlike a single global cut.
    RMS_FLOOR = 1.0
    mean = cv2.blur(residual, (25, 25))
    sq_mean = cv2.blur(residual * residual, (25, 25))
    rms = np.sqrt(np.clip(sq_mean - mean * mean, 0, None))
    rms = np.maximum(rms, RMS_FLOOR)
    img_thresh = (residual > nsigma * rms).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_thresh)
    areas = stats[1:, cv2.CC_STAT_AREA]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]

    MIN_AREA = 10
    MAX_AREA = 150
    MAX_ASPECT = 3.0
    MIN_COMPACT = 0.3

    valid = []
    for idx in range(len(areas)):
        area = areas[idx]
        w = widths[idx]
        h = heights[idx]
        aspect = max(w, h) / (min(w, h) + 1e-6)
        compactness = area / (w * h + 1e-6)

        if MIN_AREA <= area <= MAX_AREA and aspect <= MAX_ASPECT and compactness >= MIN_COMPACT:
            valid.append(idx + 1)

    mask = np.isin(labels, valid)
    img_clean = np.where(mask, 255, 0).astype(np.uint8)
    return cv2.medianBlur(img_clean, 3)

######################
# Image Registration #
######################
def register(mask, ref01):
    # keep the binary nature; align this frame's star mask onto the reference mask
    mask01 = mask.astype(np.float32) / 255.0
    transform, (src_pts, _) = aa.find_transform(mask01, ref01, detection_sigma=2.0, max_control_points=50, min_area=10)
    registered, _ = aa.apply_transform(transform, mask01, ref01, fill_value=0.0) # fill borders with black, not median gray

    # preserve binary nature — no renormalization
    return np.clip(registered * 255, 0, 255).astype(np.uint8), len(src_pts)

##################
# Image Stacking #
##################
def normalize_stack(sum_stack):
    # binary frames sum well past 255, so a raw uint8 cast overflows (mod 256); scale the density map instead
    peak = np.percentile(sum_stack, 99.9)
    return np.clip(255.0 * sum_stack / (peak + 1e-6), 0, 255).astype(np.uint8)

#############
# Operators #
#############
def op_median(frame, idx):
    return median_filter(frame)

def op_sigma(frame, idx):
    # fused background subtraction + sigma threshold + blob cleanup
    return sigma_threshold(frame, _BG, _NSIGMA)

def op_register(frame, idx):
    if idx == _REF_IDX: # reference maps onto itself
        return frame
    reg, _ = register(frame, _REF01)
    return reg

# ordered operator set, applied one by one per frame; comment a line to drop that stage
OPERATORS = [("median", op_median), ("sigma", op_sigma), ("register", op_register)]

############################
# In-Memory Chunk Pipeline #
############################
def _process_frame(args):
    # walk the operator set one by one, all in memory; only the sample frame writes intermediates
    idx, raw_path, width, height, pixel_format = args

    try:
        frame = load_flir_raw(raw_path, width, height, pixel_format)
        for name, op in OPERATORS:
            frame = op(frame, idx)
            if idx == _SAMPLE_IDX: # one image per operator
                cv2.imwrite(os.path.join(_SAMPLES_DIR, f"{name}.png"), frame)
        return idx, frame, None
    except Exception as e:
        return idx, None, str(e)

####################
# Chunk Processing #
####################
def process_chunk(chunk_dir, width=4096, height=3000, pixel_format="BayerRG8", nsigma=1.0, reference_index=200, sample_index=0, max_workers=2, n_frames=50, shape=(3000, 4096)):
    global _BG, _REF01, _NSIGMA, _REF_IDX, _SAMPLE_IDX, _SAMPLES_DIR
    raw_files = sorted(glob(os.path.join(chunk_dir, "*.raw")))
    if not raw_files:
        raise RuntimeError(f"No RAW files found in {chunk_dir}")

    n = len(raw_files)
    ref_idx = reference_index if reference_index < n else n // 2
    sample_idx = sample_index if sample_index != ref_idx else (ref_idx + 1) % n # keep the registered sample meaningful
    samples_dir = os.path.join(chunk_dir, "pngs", "samples")
    os.makedirs(samples_dir, exist_ok=True)
    background = build_background_model(raw_files, width, height, pixel_format, n_frames=n_frames)

    # parent-side state so the pre-register operators can build the reference in this process
    _BG, _NSIGMA, _REF_IDX, _SAMPLE_IDX, _SAMPLES_DIR = background, nsigma, ref_idx, sample_idx, samples_dir

    # reference = every operator before register, run on the reference frame, so all frames align to a matching target
    ref_frame = load_flir_raw(raw_files[ref_idx], width, height, pixel_format)
    for name, op in OPERATORS:
        if name == "register":
            break
        ref_frame = op(ref_frame, ref_idx)
    ref_mask = ref_frame
    _REF01 = ref_mask.astype(np.float32) / 255.0
    cv2.imwrite(os.path.join(samples_dir, "background.png"), np.clip(background, 0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join(samples_dir, "reference_sigma.png"), ref_mask)

    sum_stack = np.zeros(shape, dtype=np.float32)
    count, failed = 0, 0

    #######################
    # Track Stacked Paths #
    #######################
    stacked_paths = []
    tasks = [(i, raw_files[i], width, height, pixel_format) for i in range(n)]
    print(f"Streaming {n} frames: {' -> '.join(name for name, _ in OPERATORS)}")

    # results stream back one at a time and get summed then dropped, so RAM stays flat
    initargs = (background, _REF01, nsigma, ref_idx, sample_idx, samples_dir)
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=initargs) as executor:
        for idx, reg_u8, err in executor.map(_process_frame, tasks):
            if reg_u8 is None:
                failed += 1
                print(f"[{idx}] failed | {err}")
                continue

            if reg_u8.shape != shape:
                failed += 1
                print(f"[{idx}] failed | incorrect registered shape {reg_u8.shape}")
                continue

            sum_stack += reg_u8
            stacked_paths.append(os.path.abspath(raw_files[idx]))
            count += 1

    if count == 0:
        raise RuntimeError("No frames registered")

    ####################
    # Save Stack Paths #
    ####################
    stack_paths_file = os.path.join(chunk_dir, "stack_image_paths.txt")
    with open(stack_paths_file, "w") as f:
        for path in stacked_paths:
            f.write(path + "\n")
    print(f"Saved {len(stacked_paths)} stack image paths -> {stack_paths_file}")

    ##############
    # Save Stack #
    ##############
    out_path = os.path.join(chunk_dir, "pngs", "stacked_sum.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, normalize_stack(sum_stack))
    print(f"Stacked {count}/{n} (failed {failed}) -> {out_path}")

    return {"raw": n, "registered": count, "failed": failed, "stacked": out_path, "stack_paths": stack_paths_file}

################
# GIF Creation #
################
def create_gif_from_raw(raw_files, output_gif_path, width=4096, height=3000, pixel_format="BayerRG8", fps=10, scale=0.25):
    if len(raw_files) == 0:
        raise ValueError("No RAW files found.")

    print(f"Found {len(raw_files)} RAW files")
    duration = int(1000 / fps)
    frames = []

    for p in raw_files:
        raw = load_flir_raw(p, width, height, pixel_format)
        rgb = cv2.cvtColor(raw, cv2.COLOR_BAYER_RG2RGB)
        img = Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE)
        if scale != 1.0:
            w, h = img.size
            img = img.resize((int(w * scale), int(h * scale)))
        frames.append(img)

    frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=True)
    print(f"GIF saved to: {output_gif_path}")

#############
# Main Loop #
#############
def main():
    parser = argparse.ArgumentParser(description="Astronomy image processing pipeline")
    parser.add_argument("--step", type=str, required=True, choices=["all", "GIF"], help="Pipeline step to run")
    parser.add_argument("--raw_path", type=str, default=RAW_PATH)
    parser.add_argument("--png_path", type=str, default=PNG_PATH)
    parser.add_argument("--max_workers", type=int, default=4)
    args = parser.parse_args()

    raw_path = args.raw_path
    png_path = args.png_path

    ################
    # GIF Creation #
    ################
    if args.step == "GIF":
            raw_files = sorted(glob(os.path.join(raw_path, "*.raw")))
            output_path = os.path.join(png_path, "animated_dir.gif")
            os.makedirs(png_path, exist_ok=True)
            create_gif_from_raw(raw_files, output_path, width=4096, height=3000, pixel_format="BayerRG8", fps=10, scale=0.25)

    #################
    # Full Pipeline #
    #################
    if args.step == "all":
        process_chunk(raw_path, width=4096, height=3000, pixel_format="BayerRG8", nsigma=1.0, reference_index=200, max_workers=args.max_workers)

if __name__ == "__main__":
    main()