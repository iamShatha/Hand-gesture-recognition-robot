import pandas as pd
import os

file1 = os.path.join(os.path.expanduser("~"), "Desktop", "gestures.csv")
file2 = os.path.join(os.path.expanduser("~"), "Desktop", "gestures (1).csv")
file3 = os.path.join(os.path.expanduser("~"), "Desktop", "gestures (2).csv")

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

combined = pd.concat([df1, df2, df3], ignore_index=True)

output_file = os.path.join(os.path.expanduser("~"), "Desktop", "combined_gestures.csv")
combined.to_csv(output_file, index=False)

print("Done!")
print("Saved to:", output_file)

