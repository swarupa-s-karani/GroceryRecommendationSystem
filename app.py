from flask import Flask, request, render_template
import random
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, count, desc, sum as _sum

app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("GroceryRecommendationSystem") \
    .getOrCreate()

# Load the dataset
file_path = "./grocery-1.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# List of item columns
items_columns = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"]

# Explode the items columns to create a user-item interaction dataset
exploded_df = None
for col_name in items_columns:
    temp_df = df.select(col_name).withColumnRenamed(col_name, "item").withColumn("transactionId", lit(col_name))
    if exploded_df is None:
        exploded_df = temp_df
    else:
        exploded_df = exploded_df.union(temp_df)
# Remove null items
exploded_df = exploded_df.filter(col("item").isNotNull())

# Indexing string columns to numeric for user and item IDs
indexer_item = StringIndexer(inputCol="item", outputCol="itemIndex")
indexer_transaction = StringIndexer(inputCol="transactionId", outputCol="transactionIndex")
df_indexed = indexer_item.fit(exploded_df).transform(exploded_df)
df_indexed = indexer_transaction.fit(df_indexed).transform(df_indexed)

# Create a column for implicit feedback (binary interaction)
df_indexed = df_indexed.withColumn("rating", lit(1))

# Prepare training and test data
(training, test) = df_indexed.randomSplit([0.8, 0.2])
# Build the ALS model
als = ALS(
    maxIter=10,
    regParam=0.1,
    userCol="transactionIndex",
    itemCol="itemIndex",
    ratingCol="rating",
    coldStartStrategy="drop"
)
model = als.fit(training)

# Evaluate the model
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)
predictions = model.transform(test)
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error = {rmse}")

# Generate top N item recommendations for a given item name
def get_recommendations(item_name, top_n=5):
    item_index = df_indexed.filter(col("item") == item_name).select("itemIndex").distinct().collect()
    if not item_index:
        print(f"No data found for item: {item_name}")
        return
    item_index = item_index[0][0]

    # Get user IDs that interacted with the given item
    user_ids = df_indexed.filter(col("itemIndex") == item_index).select("transactionIndex").distinct()

    # Generate recommendations for these users
    user_recommendations = model.recommendForUserSubset(user_ids, top_n)

    # Collect recommendations for the given item
    rec_item_ids = set()
    for user_rec in user_recommendations.collect():
        recs = user_rec.recommendations
        for rec in recs:
            rec_item_ids.add(rec.itemIndex)

    # Convert item indices back to item names
    rec_item_names = [row['item'] for row in df_indexed.filter(col("itemIndex").isin(rec_item_ids)).select("item").distinct().collect()]

    if rec_item_names:
        random_recommendations = random.sample(rec_item_names, min(len(rec_item_names), top_n))
        return random_recommendations
    else:
        return []

# Get the most bought items
def get_most_bought_items(top_n=15):
    most_bought_items = None
    for col_name in items_columns:
        if most_bought_items is None:
            most_bought_items = df.groupBy(col(col_name).alias("item")).agg(count("*").alias("count"))
        else:
            most_bought_items = most_bought_items.union(df.groupBy(col(col_name).alias("item")).agg(count("*").alias("count")))
    
    # Filter out rows where item is None
    most_bought_items = most_bought_items.filter(col("item").isNotNull())
    
    # Convert count column to integer
    most_bought_items = most_bought_items.withColumn("count", col("count").cast("int"))
    
    # Aggregate counts and remove duplicates
    most_bought_items = most_bought_items.groupBy("item").agg(_sum("count").alias("total_count"))
    
    # Order by total count in descending order
    most_bought_items = most_bought_items.orderBy(desc("total_count")).limit(top_n).collect()
    
    # Prepare the results
    results = []
    for idx, item in enumerate(most_bought_items, start=1):
        results.append({"rank": idx, "item": item["item"], "total_count": item["total_count"]})
    
    return results

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    if request.method == 'POST':
        item_name = request.form['item_name']
        recommendations = get_recommendations(item_name, top_n=5)
    most_bought_items = get_most_bought_items(top_n=15)
    return render_template('index.html', recommendations=recommendations, most_bought_items=most_bought_items)

if __name__ == '__main__':
    app.run(debug=True)
