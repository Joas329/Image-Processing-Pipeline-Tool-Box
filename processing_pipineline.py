import os
import cv2
import argparse
import numpy as np
import astroalign as aa

from PIL import Image
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

RAW_PATH = "/media/joas329/My Passport/OPTICAL/exp_75/chunk_000"
PNG_PATH = "/media/joas329/My Passport/OPTICAL/exp_75/chunk_000/pngs"
MEDIAN_FILTER_PATH = os.path.join(PNG_PATH, "median_filtered")
BG_SUB_PATH = os.path.join(PNG_PATH, "bg_subtraction")
GAUSSIAN_DENOISED_PATH = os.path.join(PNG_PATH, "gaussian_denoised")
INTENSITY_THRESH_PATH = os.path.join(PNG_PATH, "intensity_thresholded")
REGISTERED_PATH = os.path.join(PNG_PATH, "registered")

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

####################################################
############### Parallel Processing ################
####################################################
def run_parallel(fn, tasks: list, max_workers: int = 4, label: str = "Processing") -> list[str]:

    total = len(tasks)
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fn, task) for task in tasks]

        for i, future in enumerate(as_completed(futures), start=1):
            path, ok, out_path, msg = future.result()

            if ok:
                print(f"[{label}] {i}/{total}: {os.path.basename(out_path)}")
                results.append((path, out_path))
            else:
                print(f"[{label}] FAILED {i}/{total}: {os.path.basename(path)} | {msg}")

    return [out for _, out in sorted(results, key=lambda x: os.path.basename(x[0]))]

####################################
########### RAW --> PNG ############
####################################

def process_raw_to_png(args):
    path, out_dir, width, height, pixel_format, debayer = args

    try:
        name = os.path.splitext(os.path.basename(path))[0]

        img = load_flir_raw(path, width, height, pixel_format)

        img_8 = img.astype(np.uint8)

        if debayer:
            rgb = cv2.cvtColor(img_8, cv2.COLOR_BAYER_RG2RGB)
            save_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            save_img = img_8

        out_path = os.path.join(out_dir, f"{name}.png")
        ok = cv2.imwrite(out_path, save_img)

        if not ok:
            return path, False, None, "cv2.imwrite failed"

        return path, True, out_path, None

    except Exception as e:
        return path, False, None, str(e)

def load_flir_raw(path, width, height, pixel_format="BayerRG8"):
    data =  np.fromfile(path, dtype=np.uint8)

    if pixel_format == "BayerRG8":
        expected_size = width * height
        if data.size != expected_size:
            raise ValueError(f"Expected {expected_size} bytes for a {width}x{height} image, but got {data.size} bytes.")
        img=data.reshape((height, width))
        return img.astype(np.float32)

    elif pixel_format == "RGB8":
        expected_size = width * height * 3
        if data.size != expected_size:
            raise ValueError(f"Expected {expected_size} bytes for a {width}x{height} image, but got {data.size} bytes.")
        img=data.reshape((height, width, 3))
        return img.astype(np.float32)

    raise ValueError(f"Unsupported pixel format: {pixel_format}")

def raw_to_png(raw_path, width, height, pixel_format="BayerRG8", debayer=False, max_workers=4):
    out_dir = os.path.join(raw_path, "pngs")
    os.makedirs(out_dir, exist_ok=True)

    raw_files = sorted(glob(os.path.join(raw_path, "*.raw")))

    if not raw_files:
        raise RuntimeError(f"No .raw files found in {raw_path}")

    print(f"[RAW -> PNG] Found {len(raw_files)} .raw files. Converting to PNG...")

    tasks = [
        (path, out_dir, width, height, pixel_format, debayer)
        for path in raw_files
    ]

    png_files = run_parallel(
        process_raw_to_png,
        tasks,
        max_workers=max_workers,
        label="RAW -> PNG"
    )

    print(f"[INFO] Done converting {len(png_files)} RAW files.")
    return png_files

######################################
########### Median Filter ############
######################################
def process_median_filter(args):
    path, out_dir, kernel_size = args

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return path, False, None, "read error"

    filtered = cv2.medianBlur(img, kernel_size)

    name = os.path.basename(path)
    out_path = os.path.join(out_dir, name)

    ok = cv2.imwrite(out_path, filtered)
    if not ok:
        return path, False, None, "write error"

    return path, True, out_path, None


def median_filter_parallel(input_dir, kernel_size=3, max_workers=4):
    filtered_dir = os.path.join(input_dir, "median_filtered")
    os.makedirs(filtered_dir, exist_ok=True)

    png_files = sorted(glob(os.path.join(input_dir, "*.png")))

    if not png_files:
        raise RuntimeError(f"No PNG files found in {input_dir}")

    print(f"[Median Filter] Found {len(png_files)} images")

    tasks = [(path, filtered_dir, kernel_size) for path in png_files]

    filtered_files = run_parallel(
        process_median_filter,
        tasks,
        max_workers=max_workers,
        label="Median Filter"
    )

    print(f"[Median Filter] Done. Created {len(filtered_files)} images")

    return filtered_files

####################################################
########### Main Background Subtraction ############
####################################################

png_files = sorted(glob(os.path.join(PNG_PATH, "*.png")))
total = len(png_files)
print(f"Found {total} PNG files")

def build_background_model(files, n_frames=50):
    bg_files = files[-n_frames:]
    print(f"Using {len(bg_files)} images to build background model")

    bg_stack = []

    for path in bg_files:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("Skipping unreadable:", path)
            continue
        bg_stack.append(img.astype(np.float32))

    if len(bg_stack) == 0:
        raise RuntimeError("No readable images for background model")

    background = np.median(np.stack(bg_stack, axis=0), axis=0)
    print(f"Background computed from {len(bg_stack)} frames")

    return background

def process_bg_subtraction(args):
    path, background, out_dir = args

    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return path, False, None, "read error"

    img_bg_sub = img.astype(np.float32) - background
    img_bg_sub = np.clip(img_bg_sub, 0, None)

    # Normalize to 8-bit properly before saving
    p1, p99 = np.percentile(img_bg_sub, [1, 99.7])
    img_norm = np.clip((img_bg_sub - p1) / (p99 - p1 + 1e-6), 0, 1)
    img_8 = (img_norm * 255).astype(np.uint8)

    name = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{name}_bgsub.png")
    cv2.imwrite(out_path, img_8)

    return path, True, out_path, None


###########################################
########### Gaussian Denoising ############
###########################################
def process_gaussian_denoise(args):
    path, out_dir, kernel_size, sigma = args

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return path, False, None, "read error"

    denoised = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    name = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{name}_gauss.png")
    cv2.imwrite(out_path, denoised)

    return path, True, out_path, None


###############################################
########### Intensity Thresholding ############
###############################################
def process_threshold(args):
    path, out_dir = args

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return path, False, None, "read error"

    img = img.astype(np.float32)

    threshold = np.percentile(img, 99.95)
    img_thresh = np.where(img >= threshold, img, 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_thresh)

    areas   = stats[1:, cv2.CC_STAT_AREA]
    widths  = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]

    MIN_AREA    = 10
    MAX_AREA    = 150
    MAX_ASPECT  = 3.0
    MIN_COMPACT = 0.3

    valid = []
    for idx in range(len(areas)):
        area   = areas[idx]
        w      = widths[idx]
        h      = heights[idx]
        aspect = max(w, h) / (min(w, h) + 1e-6)
        compactness = area / (w * h + 1e-6)

        if MIN_AREA <= area <= MAX_AREA and aspect <= MAX_ASPECT and compactness >= MIN_COMPACT:
            valid.append(idx + 1)

    mask      = np.isin(labels, valid)
    img_clean = np.where(mask, 255, 0).astype(np.uint8)
    img_clean = cv2.medianBlur(img_clean, 3)

    name     = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{name}_thresholded.png")
    cv2.imwrite(out_path, img_clean)

    return path, True, out_path, None

###########################################
########### Image Registration ############
###########################################
def find_matching_image(input_dir, raw_path):
    base = os.path.splitext(os.path.basename(raw_path))[0]

    matches = sorted(glob(os.path.join(input_dir, f"{base}*.png")))

    if len(matches) == 0:
        raise FileNotFoundError(f"No PNG found for base name: {base}")

    if len(matches) > 1:
        print(f"Warning: multiple matches for {base}, using: {os.path.basename(matches[0])}")

    return matches[0]

def load_registered_images(out_dir, raw_files):
    registered_images = []

    for path in raw_files:
        name = os.path.splitext(os.path.basename(path))[0]
        img_path = os.path.join(out_dir, f"{name}_registered.png")

        if not os.path.exists(img_path):
            print(f"Missing: {img_path}")
            registered_images.append(None)
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Failed to load: {img_path}")
            registered_images.append(None)
            continue

        # Convert back to float32 [0,1]
        img = img.astype(np.float32) / 255.0

        registered_images.append(img)

    return registered_images


def prep_for_registration(img):
    img = img.astype(np.float32)

    # background subtraction
    bg = np.percentile(img, 10)
    img = img - bg
    img[img < 0] = 0

    # optional denoise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # normalize robustly first
    p_low, p_high = np.percentile(img, [50, 99.9])
    img = np.clip((img - p_low) / (p_high - p_low + 1e-6), 0, 1)

    # logarithmic stretch
    alpha = 100.0
    img = np.log1p(alpha * img) / np.log1p(alpha)

    return img.astype(np.float32)


def register_one_image(args):
    i, raw_path, input_dir, ref_img, ref_reg, reference_index, out_dir = args

    base = os.path.splitext(os.path.basename(raw_path))[0]

    try:
        img_path = find_matching_image(input_dir, raw_path)

        print(f"[{i}] Processing {os.path.basename(img_path)}")

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(f"Could not load {img_path}")

        # Keep as float [0, 1], don't renormalize, preserve binary nature
        img = img.astype(np.float32) / 255.0

        if i == reference_index:
            registered = img
            status = "reference"
            matched = None
        else:
            transform, (src_pts, dst_pts) = aa.find_transform(
                img,
                ref_reg,
                detection_sigma=2.0,
                max_control_points=50,
                min_area=10,
            )

            registered, footprint = aa.apply_transform(
                transform,
                img,
                ref_img,
                fill_value=0.0,  # fill borders with black, not median gray
            )

            status = "registered"
            matched = len(src_pts)

        out_path = os.path.join(out_dir, f"{base}_registered.png")

        # Preserve binary nature — no renormalization
        registered_8 = np.clip(registered * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(out_path, registered_8)

        return {
            "index": i,
            "status": status,
            "matched": matched,
            "error": None,
        }

    except Exception as e:
        return {
            "index": i,
            "status": "failed",
            "matched": None,
            "error": str(e),
        }

def register_images_sigma_clipped_parallel(
    raw_files,
    sigma_dir,
    out_dir,
    reference_index=0,
    batch_size=32,
    max_workers=6,
):
    os.makedirs(out_dir, exist_ok=True)
    n = len(raw_files)

    ref_path = find_matching_image(sigma_dir, raw_files[reference_index])
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        raise RuntimeError(f"Could not load reference image: {ref_path}")

    ref_img = ref_img.astype(np.float32) / 255.0
    ref_reg = ref_img

    results_log = []
    successful_paths = []  # ← track only successful registrations

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        print(f"\nProcessing batch {batch_start} to {batch_end - 1}")

        tasks = [
            (i, raw_files[i], sigma_dir, ref_img, ref_reg, reference_index, out_dir)
            for i in range(batch_start, batch_end)
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(register_one_image, t) for t in tasks]

            for future in as_completed(futures):
                result = future.result()
                i = result["index"]

                if result["status"] == "registered":
                    print(f"[{i}] registered | stars={result['matched']}")
                    base = os.path.splitext(os.path.basename(raw_files[i]))[0]
                    successful_paths.append(os.path.join(out_dir, f"{base}_registered.png"))

                elif result["status"] == "reference":
                    print(f"[{i}] reference")
                    base = os.path.splitext(os.path.basename(raw_files[i]))[0]
                    successful_paths.append(os.path.join(out_dir, f"{base}_registered.png"))

                else:
                    print(f"[{i}] failed | {result['error']}")

                results_log.append(result)

    print(f"\nSuccessfully registered: {len(successful_paths)}/{n} frames")

    manifest_path = os.path.join(out_dir, "successful_frames.txt")
    with open(manifest_path, "w") as f:
        for p in sorted(successful_paths):
            f.write(p + "\n")
    print(f"Manifest saved: {manifest_path}")

    return results_log, successful_paths

#######################################
########### Image Stacking ############
#######################################
def stack_registered_images(
    output_path,
    expected_shape=(3000, 4096),
    input_dir=None,
    registered_files=None,       # ← pass explicit list if available
    pattern="*_registered.png"
):
    if registered_files is None:
        if input_dir is None:
            raise ValueError("Must provide either input_dir or registered_files")
        registered_files = sorted(glob(os.path.join(input_dir, pattern)))

    if not registered_files:
        raise RuntimeError("No registered images to stack")

    print(f"[Stacking] Stacking {len(registered_files)} successfully registered images")

    sum_stack = np.zeros(expected_shape, dtype=np.float32)
    count = 0
    skipped = []

    for path in registered_files:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped.append((path, "failed load"))
            continue
        if img.shape != expected_shape:
            skipped.append((path, img.shape))
            continue
        sum_stack += img.astype(np.float32)
        count += 1

    print(f"[Stacking] Stacked: {count} | Skipped: {len(skipped)}")

    if count == 0:
        raise RuntimeError("No valid images were stacked")

    stacked_8 = sum_stack.astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, stacked_8)
    print(f"[Stacking] Saved: {output_path}")
    return output_path

#######################################
############ GIF Creation #############
#######################################
def create_gif_from_dir(input_dir, output_gif_path, fps=10, scale=0.25):
    image_paths = sorted(glob(os.path.join(input_dir, "*.png")))

    if len(image_paths) == 0:
        raise ValueError("No PNG files found in directory.")

    print(f"Found {len(image_paths)} images")

    duration = int(1000 / fps)
    frames = []

    for p in image_paths:
        img = Image.open(p).convert("P", palette=Image.ADAPTIVE)

        if scale != 1.0:
            w, h = img.size
            img = img.resize((int(w * scale), int(h * scale)))

        frames.append(img)

    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True
    )

    print(f"GIF saved to: {output_gif_path}")


def main():
    parser = argparse.ArgumentParser(description="Astronomy image processing pipeline")

    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["raw_to_png", "bg_sub", "gaussian", "median", "threshold", "register", "stack","all", "GIF"],
        help="Pipeline step to run"
    )

    parser.add_argument("--raw_path", type=str, default=RAW_PATH)
    parser.add_argument("--png_path", type=str, default=PNG_PATH)
    parser.add_argument("--max_workers", type=int, default=4)

    args = parser.parse_args()

    raw_path = args.raw_path
    png_path = args.png_path

    # -----------------
    # GIF Creation (Optional)
    # -----------------
    output_path = os.path.join(PNG_PATH,"animated_dir.gif")
    if args.step == "GIF":
        create_gif_from_dir(
            input_dir=INTENSITY_THRESH_PATH,
            output_gif_path=output_path,
            fps=10,
            scale=0.25
        )

    # -----------------
    # RAW -> PNG
    # -----------------
    if args.step in ["raw_to_png", "all"]:
        raw_to_png(
            raw_path,
            width=4096,
            height=3000,
            pixel_format="BayerRG8",
            debayer=False,
            max_workers=args.max_workers
        )

    # Refresh PNG files after optional conversion
    png_files = sorted(glob(os.path.join(png_path, "*.png")))

    if len(png_files) == 0 and args.step not in ["raw_to_png"]:
        raise RuntimeError(f"No PNG files found in {png_path}")

    # -----------------
    # Median filtering
    # -----------------
    if args.step in ["median", "all"]:

        median_files = median_filter_parallel(
            PNG_PATH,
            kernel_size=3,
            max_workers=args.max_workers
        )

    # -----------------
    # Background subtraction
    # -----------------
    if args.step in ["bg_sub", "all"]:
        bg_dir = BG_SUB_PATH
        os.makedirs(bg_dir, exist_ok=True)

        median_files = sorted(glob(os.path.join(MEDIAN_FILTER_PATH, "*.png")))

        if len(median_files) == 0:
            raise RuntimeError(f"No median-filtered PNG files found in {MEDIAN_FILTER_PATH}")

        background = build_background_model(median_files, n_frames=50)

        tasks = [(path, background, bg_dir) for path in median_files]

        bg_sub_files = run_parallel(
            process_bg_subtraction,
            tasks,
            max_workers=args.max_workers,
            label="BG Subtraction"
        )

        print(f"Done. Created {len(bg_sub_files)} background-subtracted images.")
    # -----------------
    # Gaussian denoising
    # -----------------
    if args.step in ["gaussian", "all"]:
        bg_dir = os.path.join(png_path, "bg_subtraction")
        bg_sub_files = sorted(glob(os.path.join(bg_dir, "*.png")))

        denoise_dir = os.path.join(png_path, "gaussian_denoised")
        os.makedirs(denoise_dir, exist_ok=True)

        tasks = [(path, denoise_dir, 3, 0) for path in bg_sub_files]

        denoised_files = run_parallel(
            process_gaussian_denoise,
            tasks,
            max_workers=args.max_workers,
            label="Gaussian denoise"
        )

        print(f"Done. Created {len(denoised_files)} denoised images.")

    # -----------------
    # Intensity thresholding
    # -----------------
    if args.step in ["threshold", "all"]:
        bg_dir = os.path.join(png_path, "bg_subtraction")
        bg_sub_files = sorted(glob(os.path.join(bg_dir, "*.png")))

        os.makedirs(INTENSITY_THRESH_PATH, exist_ok=True)

        tasks = [(path, INTENSITY_THRESH_PATH) for path in bg_sub_files]

        thresholded_files = run_parallel(
            process_threshold,
            tasks,
            max_workers=args.max_workers,
            label="Thresholding"
        )

        print(f"Done. Created {len(thresholded_files)} thresholded images.")

    # -----------------
    # Image registration
    # -----------------
    if args.step in ["register", "all"]:
        raw_files = sorted(glob(os.path.join(raw_path, "*.raw")))

        if len(raw_files) == 0:
            raise RuntimeError(f"No RAW files found in {raw_path}")

        os.makedirs(REGISTERED_PATH, exist_ok=True)

        registration_log, successful_paths = register_images_sigma_clipped_parallel(
            raw_files=raw_files,
            sigma_dir=INTENSITY_THRESH_PATH,
            out_dir=REGISTERED_PATH,
            reference_index=200,
            batch_size=32,
            max_workers=args.max_workers,
        )
        print(f"Registration finished. {len(successful_paths)} frames ready for stacking.")

    # -----------------
    # Image Stacking
    # -----------------
    if args.step in ["stack", "all"]:
        if 'successful_paths' in dir():
            files_to_stack = successful_paths
        else:
            manifest_path = os.path.join(REGISTERED_PATH, "successful_frames.txt")
            if not os.path.exists(manifest_path):
                raise RuntimeError(
                    f"No manifest found at {manifest_path}. "
                    f"Run --step register first, or provide successful_paths."
                )
            with open(manifest_path) as f:
                files_to_stack = [line.strip() for line in f if line.strip()]

        print(f"Stacking {len(files_to_stack)} frames from manifest")

        stack_registered_images(output_path=os.path.join(PNG_PATH, "stacked_sum.png"), expected_shape=(3000, 4096), registered_files=files_to_stack)

if __name__ == "__main__":
    main()

