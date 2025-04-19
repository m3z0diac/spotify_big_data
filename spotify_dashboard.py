from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, col
from pyspark.sql.types import DoubleType, IntegerType
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Start Spark session with adjusted log level
spark = SparkSession.builder \
    .appName("SpotifyCharts") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
    .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017") \
    .getOrCreate()

# Set log level to WARN to reduce console output
spark.sparkContext.setLogLevel("WARN")

# Read from MongoDB
df = spark.read \
    .format("mongodb") \
    .option("uri", "mongodb://localhost:27017") \
    .option("database", "spotify") \
    .option("collection", "songs") \
    .load()

# Clean and prep - ensure we only select and cast numeric columns
numeric_cols = ["year", "tempo", "duration_min", "danceability", "energy"]
df = df.select([col(c).cast(DoubleType()).alias(c) for c in numeric_cols]).dropna()

# === 1. Yearly Music Trends ===
yearly_avg = df.groupBy("year").agg(
    count("*").alias("num_songs"),
    avg("danceability").alias("avg_danceability"),
    avg("energy").alias("avg_energy"),
    avg("tempo").alias("avg_tempo"),
    avg("duration_min").alias("avg_duration_min")
).orderBy("year")

# === 2. Feature Correlations ===
try:
    # Convert to Pandas and ensure all values are numeric
    correlation_df = df.select(numeric_cols).toPandas()
    # Convert all columns to numeric, coercing errors to NaN
    correlation_df = correlation_df.apply(pd.to_numeric, errors='coerce').dropna()
    corr_matrix = correlation_df.corr()
except Exception as e:
    print(f"Error calculating correlations: {e}")
    corr_matrix = pd.DataFrame()

# === 3. Energy vs. Danceability Analysis ===
energy_dance = df.sample(fraction=0.1).select("energy", "danceability").toPandas()

# === 4. Duration Distribution ===
duration_dist = df.select("duration_min").toPandas()

# Convert to Pandas
yearly_df = yearly_avg.toPandas()

# === CHARTS ===
plt.figure(figsize=(15, 10))

# Chart 1: Yearly Trends - Danceability and Energy
plt.subplot(2, 2, 1)
plt.plot(yearly_df['year'], yearly_df['avg_danceability'], label='Danceability')
plt.plot(yearly_df['year'], yearly_df['avg_energy'], label='Energy')
plt.title('Yearly Trends: Danceability vs Energy')
plt.xlabel('Year')
plt.ylabel('Average Score')
plt.legend()
plt.grid(True)

# Chart 2: Feature Correlation Heatmap (only if we have data)
if not corr_matrix.empty:
    plt.subplot(2, 2, 2)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
else:
    plt.subplot(2, 2, 2)
    plt.text(0.5, 0.5, 'Correlation data not available', ha='center', va='center')
    plt.title('Feature Correlation Matrix (Data Unavailable)')

# Chart 3: Energy vs Danceability Scatter Plot
plt.subplot(2, 2, 3)
sns.scatterplot(x='danceability', y='energy', data=energy_dance, alpha=0.5)
plt.title('Energy vs Danceability Relationship')
plt.xlabel('Danceability')
plt.ylabel('Energy')

# Chart 4: Duration Distribution
plt.subplot(2, 2, 4)
sns.histplot(duration_dist['duration_min'], bins=30, kde=True)
plt.title('Song Duration Distribution')
plt.xlabel('Duration (minutes)')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Additional Visualization: Yearly Song Count
plt.figure(figsize=(8, 4))
sns.lineplot(x='year', y='num_songs', data=yearly_df)
plt.title('Number of Songs per Year')
plt.xlabel('Year')
plt.ylabel('Number of Songs')
plt.grid(True)
plt.tight_layout()
plt.show()

# Stop Spark
spark.stop()
