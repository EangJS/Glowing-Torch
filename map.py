import pandas as pd

id2label = {0: "None of the given categories", 1: "Regular and Straight Jeans", 2: "Other Jeans", 3: "Basic, Cotton, Plain and Short Sleeve T-Shirts", 4: "Other T-Shirts",
            5: "Regular, Plain, Short Sleeve Polo Shirt", 6: "All other Polo Shirts", 7: "Chino Shorts", 8: "All other shorts", 9: "Basic Tank Tops or Vest Tops", 10: "Other Tank Tops or Vest Tops"}

df = pd.read_csv('Datasets/uniqlo_dataset.csv', encoding='latin1')
df['category'] = df['classification'].map(id2label)
df.to_csv('Datasets/uniqlo_map_dataset.csv', index=False)
