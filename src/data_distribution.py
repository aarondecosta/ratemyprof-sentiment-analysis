import json
import matplotlib.pyplot as plt

with open("reviews.json", "r") as f:
    datastore = json.load(f)

# Initialize counts for each category
ratings_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

# Iterate through the data and count ratings
for entry in datastore:
    rating = entry["rating"] + 1
    ratings_count[rating] += 1

# Print the counts for each category
for rating, count in ratings_count.items():
    print(f"Rating {rating}: {count} occurrences")

# Plotting the bar graph
plt.bar(ratings_count.keys(), ratings_count.values())

# Adding labels and title
plt.xlabel("Categories")
plt.ylabel("Values")
plt.title("Rating Distribution")

# Optionally, rotate x-axis labels if they're too long
plt.xticks(rotation=45)

# Optionally, add gridlines
plt.grid(True)

# Display the graph
plt.show()
