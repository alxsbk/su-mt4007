```python
import pandas as pd

# Data for apartments sold in lappis

# Load data from CSV
data = pd.read_csv("Booli_sold.csv")

# Add a new column for price per square meter
data['ppsqm'] = data['soldPrice'] / data['livingArea']

# Sort data by ppsqm in descending order and select the top 5 rows
top_5_expensive = data.sort_values(by='ppsqm', ascending=False).head(5)
top_5_expensive = top_5_expensive[['booliId', 'listPrice', 'soldPrice', 'livingArea', 'ppsqm', 'location.address.streetAddress', 'location.region.municipalityName']]

# Filter data for apartments in Ekhagen
ekhagen_data = data[data['location.region.municipalityName'] == 'Ekhagen']

# Calculate the average ppsqm
average_ppsqm_ekhagen = ekhagen_data['ppsqm'].mean()
```

