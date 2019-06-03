import pandas as pd

"WRANGLING"

df = pd.DataFrame(data={"Price": [10000., 25000.]})
df["PriceMinus100"] = df["Price"] - 100. #two cols same types T1


"ANALYSIS"

# The following should raise a warning because the multiplication
# resets the type
print(df[df["Price"] * 9. == df["PriceMinus100"]])


