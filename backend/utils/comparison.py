import cv2
import numpy as np


def create_comparison(
    original_path,
    mask_path,
    overlay_path,
    output_path,
    model_name,
    tumor_percentage,
    confidence,
    inference_time,
):

    # Read images
    original = cv2.imread(original_path)
    mask = cv2.imread(mask_path)
    overlay = cv2.imread(overlay_path)

    # Resize all to same size
    width = 300
    height = 300

    original = cv2.resize(original, (width, height))
    mask = cv2.resize(mask, (width, height))
    overlay = cv2.resize(overlay, (width, height))

    # White canvas
    canvas = np.ones((520, 940, 3), dtype=np.uint8) * 255

    # Title
    cv2.putText(
        canvas,
        "Brain Tumor Segmentation",
        (220, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )

    # Images
    canvas[70:370, 20:320] = original
    canvas[70:370, 320:620] = mask
    canvas[70:370, 620:920] = overlay

    # Labels
    cv2.putText(canvas, "Original MRI", (90, 395),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.putText(canvas, "Predicted Mask", (365, 395),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.putText(canvas, "Overlay", (730, 395),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Information
    y = 435

    cv2.putText(canvas,
                f"Model: {model_name}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2)

    cv2.putText(canvas,
                f"Tumor Area: {tumor_percentage:.2f}%",
                (20, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 128, 0),
                2)

    cv2.putText(canvas,
                f"Confidence: {confidence:.4f}",
                (20, y + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 0, 0),
                2)

    cv2.putText(canvas,
                f"Inference Time: {inference_time:.2f} ms",
                (20, y + 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                2)

    cv2.imwrite(output_path, canvas)