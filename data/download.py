import json
from google.cloud import bigquery

# Initialize a BigQuery client
client = bigquery.Client()

# Define the TEST_ITEM_COMMUNICATIONKEY
test_item_communication_key = 1611665929

# Define the SQL query
query = f"""
SELECT
  MIN(CALENDAR_DATE) AS Start_Date,
  MAX(CALENDAR_DATE) AS End_Date,
  STRUCT(ITEM_COMMUNICATIONKEY AS Key, ARRAY_AGG(ANSPRACHE_ITEM) AS Values) AS Item_Data
FROM `brain-flash-dev.dagster_common.CN_twins_dynamic`
WHERE CALENDAR_DATE BETWEEN "2021-12-01" AND "2022-01-01" AND TEST_ITEM_COMMUNICATIONKEY = {test_item_communication_key}
GROUP BY ITEM_COMMUNICATIONKEY
"""

# Run the query
query_job = client.query(query)

# Fetch the results
results = query_job.result()

# Prepare the data for JSON output
output_data = []
for row in results:
    output_data.append({
        "Start_Date": row.Start_Date.isoformat() if row.Start_Date else None,
        "End_Date": row.End_Date.isoformat() if row.End_Date else None,
        "Item_Data": {
            "Key": row.Item_Data.Key,
            "Values": row.Item_Data.Values
        }
    })

# Define the output filename
output_filename = f"{test_item_communication_key}.json"

# Write the results to a JSON file
with open(output_filename, 'w') as json_file:
    json.dump(output_data, json_file, indent=4)

print(f"Results saved to {output_filename}")