import pandas as pd

df = pd.read_csv("results.csv")

accuracy = (
    df.groupby("lab")
      .apply(lambda x: (x["prediction"] == x["actual"]).mean())
      .reset_index(name="accuracy")
)

print("\nAccuracy by Lab:")
print(accuracy)
