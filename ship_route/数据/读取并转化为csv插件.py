import pandas as pd

# csv路径
csv_output_path = "data_columns_PM.csv"

# Read the CSV file back and create a dictionary in the requested format
loaded_data = pd.read_csv(csv_output_path, header=None)

# Create the dictionary structure similar to P_M
P_M = {
    (col + 1, row + 1): float(loaded_data.iloc[row, col])
    for col in range(loaded_data.shape[1])
    for row in range(loaded_data.shape[0])
}

# Display a sample of the dictionary
i = 0
j = 20
t = 0
for t in range(30):
    P_M_sample = {k: P_M[k] for k in list(P_M)[i:j]}  # Display first 20 entries
    i += 20
    j += 20
    print(P_M_sample)
print(P_M[1, 100])
print(P_M[3, 5])
print(P_M[6, 76])
