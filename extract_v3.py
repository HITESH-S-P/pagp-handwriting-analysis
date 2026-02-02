import cv2
import numpy as np

def extract_lines(binary, min_line_height=20):
    height, width = binary.shape
    horiz_proj = np.sum(binary > 0, axis=1).astype(float)
    thresh = np.max(horiz_proj) * 0.05
    lines = []
    start = None
    for i in range(height):
        if horiz_proj[i] > thresh:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_line_height:
                    lines.append((start, i))
                start = None
    if start is not None and height - start >= min_line_height:
        lines.append((start, height))
    return lines

def dilate(image, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def start(image_path):
    """
    Extracts raw feature values from the handwriting image using computer vision techniques.
    Returns a list of raw values for: baseline_angle, top_margin, letter_size, line_spacing, word_spacing, pen_pressure, slant_angle
    """

    # Read image
    image_bytes = open(image_path, "rb").read()
    if len(image_bytes) == 0:
        raise ValueError("Uploaded image file is empty")

    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode the image")

    height, width = img.shape[:2]

    # Resize for consistency (assume standard width)
    target_width = 1280
    scale = target_width / width
    img = cv2.resize(img, (target_width, int(height * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Binarize
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect if there's content
    if np.sum(binary) < 1000 * 255:  # arbitrary low ink threshold
        raise ValueError("No sufficient handwriting detected in the image")

    # 1. Pen pressure
    ink_pixels = gray[binary > 0]
    avg_intensity = np.mean(ink_pixels)
    pen_pressure = 255 - avg_intensity  # higher value for heavier pressure

    # 2. Top margin
    horiz_proj = np.sum(binary > 0, axis=1).astype(float)
    thresh = np.max(horiz_proj) * 0.05
    top_rows = np.where(horiz_proj > thresh)[0]
    if len(top_rows) == 0:
        raise ValueError("No lines detected")
    first_line_start = top_rows[0]
    top_margin = first_line_start / (height / 10.0)  # scale to fit categorize thresholds (~0-10 range)

    # 3. Extract lines
    lines = extract_lines(binary)
    if len(lines) == 0:
        raise ValueError("No lines detected")

    line_heights = [end - start for start, end in lines]
    letter_size = np.mean(line_heights) / 10.0  # scale to fit ~10-20 range

    # 4. Line spacing (normalized)
    if len(lines) > 1:
        spacings = [lines[i+1][0] - lines[i][1] for i in range(len(lines)-1)]
        line_spacing = np.mean(spacings) / np.mean(line_heights)  # ratio ~1-4
    else:
        line_spacing = 2.5  # default medium

    # 5. Word spacing (average normalized gap)
    word_spacing_ratios = []
    for start_l, end_l in lines:
        line_bin = binary[start_l:end_l, :]
        vert_proj = np.sum(line_bin > 0, axis=0).astype(float)
        v_thresh = np.max(vert_proj) * 0.05
        words = []
        start_w = None
        for j in range(width):
            if vert_proj[j] > v_thresh:
                if start_w is None:
                    start_w = j
            else:
                if start_w is not None:
                    words.append((start_w, j))
                    start_w = None
        if start_w is not None:
            words.append((start_w, width))
        if len(words) > 1:
            gaps = [words[i+1][0] - words[i][1] for i in range(len(words)-1)]
            word_widths = [end_w - start_w for start_w, end_w in words]
            avg_gap = np.mean(gaps)
            avg_width = np.mean(word_widths)
            word_spacing_ratios.append(avg_gap / avg_width if avg_width > 0 else 1.5)
    word_spacing = np.mean(word_spacing_ratios) if word_spacing_ratios else 1.5

    # 6. Baseline angle (average slope)
    baseline_slopes = []
    for start_l, end_l in lines:
        line_bin = binary[start_l:end_l, :]
        cols = []
        lows = []
        for col in range(width):
            column = line_bin[:, col]
            ink_rows = np.where(column > 0)[0]
            if len(ink_rows) > 0:
                lows.append(ink_rows[-1])  # bottom of ink in line relative
                cols.append(col)
        if len(cols) > 20:
            slope = np.polyfit(cols, lows, 1)[0]
            baseline_slopes.append(slope)
    baseline_angle = np.mean(baseline_slopes) if baseline_slopes else 0.0

    # 7. Slant angle (average from vertical strokes)
    # Dilate vertically to connect strokes
    v_dilated = dilate(binary, (5, 30))  # vertical kernel
    ctrs, _ = cv2.findContours(v_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    slant_angles = []
    for ctr in ctrs:
        if len(ctr) < 5:
            continue
        rect = cv2.minAreaRect(ctr)
        angle = rect[2]
        # Normalize to -45 to 45 from vertical (assuming 90 is vertical)
        if angle > 90:
            angle -= 180
        if abs(angle) < 60:  # plausible slant
            slant_angles.append(angle)
    slant_angle = np.mean(slant_angles) if slant_angles else 0.0

    return [
        baseline_angle,
        top_margin,
        letter_size,
        line_spacing,
        word_spacing,
        pen_pressure,
        slant_angle
    ]