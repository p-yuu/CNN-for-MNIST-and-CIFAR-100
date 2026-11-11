from pathlib import Path
from collections import Counter


BASE_DIR = Path(__file__).resolve().parent.parent

GROUND_TRUTH_PATH = BASE_DIR / "data" / "test" / "ground_truth.txt"
PREDICTION_PATH = BASE_DIR / "outputs" / "results" / "predictions.txt"


def load_labels(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    labels = {}
    duplicates = []

    with open(file_path, "r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) != 2:
                raise ValueError(f"{file_path.name}, line {line_number}: Invalid format: {line}")

            image_name, class_name = parts

            if image_name in labels:
                duplicates.append(image_name)

            labels[image_name] = class_name

    return labels, duplicates


def evaluate():
    ground_truth, gt_duplicates = load_labels(GROUND_TRUTH_PATH)
    predictions, pred_duplicates = load_labels(PREDICTION_PATH)

    gt_names = set(ground_truth)
    pred_names = set(predictions)

    missing = sorted(gt_names - pred_names)
    extra = sorted(pred_names - gt_names)

    correct = 0
    incorrect = []

    for image_name in sorted(gt_names & pred_names):
        true_label = ground_truth[image_name]
        pred_label = predictions[image_name]

        if true_label == pred_label:
            correct += 1
        else:
            incorrect.append((image_name, true_label, pred_label))

    total = len(ground_truth)
    accuracy = correct / total * 100 if total else 0.0

    print("\nEvaluation Results")
    print("-" * 45)

    print(f"Ground Truth Images: {len(ground_truth)}")
    print(f"Predicted Images:    {len(predictions)}")
    print(f"Correct:             {correct}")
    print(f"Incorrect:           {len(incorrect)}")
    print(f"Missing:             {len(missing)}")
    print(f"Extra:               {len(extra)}")

    print(f"\nTest Accuracy: {accuracy:.2f}%")

    if gt_duplicates:
        print(f"\nDuplicate Ground Truth IDs: {gt_duplicates[:10]}")

    if pred_duplicates:
        print(f"Duplicate Prediction IDs: {pred_duplicates[:10]}")

    if missing:
        print(f"\nMissing predictions: {missing[:10]}")

    if extra:
        print(f"Unexpected predictions: {extra[:10]}")

    if incorrect:
        print("\nFirst 10 incorrect predictions:")

        for image_name, true_label, pred_label in incorrect[:10]:
            print(
                f"{image_name}: "
                f"GT={true_label}, Pred={pred_label}"
            )

    if (gt_duplicates or pred_duplicates or missing or extra):
        print("\nWARNING: Prediction file has duplicate, missing, or extra IDs.")


if __name__ == "__main__":
    evaluate()
