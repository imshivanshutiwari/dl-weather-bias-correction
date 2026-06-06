import numpy as np
from datetime import datetime, timedelta

# Define the start and end years
start_year = 2019
end_year = 2023

# List to hold datetime64 values
datetime_list = []

# Loop through each year
for year in range(start_year, end_year + 1):
    start_date = datetime(year, 6, 1, 3)
    end_date = datetime(year, 10, 1, 0)  # Include October 1st, 00:00

    current = start_date
    while current <= end_date:
        dt64 = np.datetime64(current, 'h')  # Hour-level precision
        datetime_list.append(dt64)
        current += timedelta(hours=3)

# Convert list to numpy array
datetime_array = np.array(datetime_list, dtype='datetime64[h]')

# Save to .npy file
np.save("jjas_datetime_3hr_incl_31s_and_oct1.npy", datetime_array)

print("Saved to 'jjas_datetime_3hr_incl_31s_and_oct1.npy'")
