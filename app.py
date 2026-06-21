from transformers import pipeline
from PIL import Image
import pandas as pd
import os

MODEL_NAME = "prithivMLmods/WBC-Type-Classifier"

classifier = pipeline(
    "image-classification",
    model=MODEL_NAME
)

results = []

DATASET_FOLDER = "datasets"

for lab in os.listdir(DATASET_FOLDER):

    lab_path = os.path.join(DATASET_FOLDER, lab)

    if not os.path.isdir(lab_path):
        continue

    for image_file in os.listdir(lab_path):

        if image_file.lower().endswith((".jpg", ".jpeg", ".png")):

            image_path = os.path.join(lab_path, image_file)

            try:
                image = Image.open(image_path)

                prediction = classifier(image)

                top = prediction[0]

                results.append({
                    "lab": lab,
                    "image": image_file,
                    "prediction": top["label"],
                    "confidence": round(top["score"] * 100, 2)
                })

                print(
                    f"{image_file} -> "
                    f"{top['label']} "
                    f"({top['score']*100:.2f}%)"
                )

            except Exception as e:
                print(f"Error on {image_file}: {e}")

df = pd.DataFrame(results)

df.to_csv("results.csv", index=False)

print("\nSaved to results.csv")
