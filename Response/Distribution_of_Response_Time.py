!pip install pymongo
import pymongo
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt


mgdb_host = "HOST"
mgdb_port = 27017
mgdb_username = 'USERNAME'
mgdb_password = 'PASSWORD'
mgdb_database = "DATABASE"
mgdb_collection = "COLLECTION"
client = MongoClient('mongodb+srv://%s:%s@%s/%s' % (mgdb_username, mgdb_password, mgdb_host, mgdb_database))
database = client[mgdb_database]
collection = database[mgdb_collection]


# Find all documents in the collection using an empty query object
all_documents = collection.find({})

# Convert the documents to a pandas DataFrame
df = pd.DataFrame(list(all_documents))
print(df['response_time'])


# Calculate the mean response time
mean_response_time = df['response_time'].mean()

# Plot the line for the mean response time
plt.axhline(mean_response_time, color='red', linestyle='dashed', linewidth=2, label=f'Mean Response Time: {mean_response_time:.2f} seconds')

# Convert ObjectId to string for the '_id' column
df['_id'] = df['_id'].astype(str)

# Plot the response time data as a line plot
plt.plot(df['_id'], df['response_time'], marker='o', linestyle='-', label='Response Time')

# Set the labels and title
plt.xlabel('Document ID')
plt.ylabel('Response Time (seconds)')
plt.title('Distribution of Responses Time')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=90)

# Display the plot
plt.show()
