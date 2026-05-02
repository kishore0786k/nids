import pandas as pd

# 1) Read the Parquet dataset
df = pd.read_parquet("data/NF-ToN-IoT-V2.parquet")

# 2) Save it as CSV with this exact name
df.to_csv("data/NF-ToN-IoT-V2.csv", index=False)

print("Done: data/NF-ToN-IoT-V2.csv created")
