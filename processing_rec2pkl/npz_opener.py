import numpy as np

# Load the file
data = np.load(r"L:\xl_cl\Albert\20251022_dio\CnL42_0_90_two_grating_passive_static_20251022_160842_DIO_cleaned.npz")

# See what’s inside
print(data.files)

# Access the arrays
rising = data["rising_times"]
falling = data["falling_times"]

print("Rising shape:", rising.shape)
print("Falling shape:", falling.shape)

# Preview first few entries
print("First 10 rising times:", rising[:10])
print("First 10 falling times:", falling[:10])
