from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, udf
from pyspark.sql.types import DoubleType, ArrayType, StringType
from pyspark.ml.feature import CountVectorizer, IDF, Normalizer
from pyspark.ml.recommendation import ALS
import numpy as np

# Start Spark Session
spark = (SparkSession.builder
    .appName("HybridMovieRecommender")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.executor.heartbeatInterval", "60s")
    .config("spark.network.timeout", "800s")
    .getOrCreate())

# Load datasets
ratings = spark.read.option("header", True).option("inferSchema", True).csv("hdfs://localhost:9000/user/prakash/moviedata/ratings.csv")
movies = spark.read.option("header", True).option("inferSchema", True).csv("hdfs://localhost:9000/user/prakash/moviedata/movies.csv")

# Clean genres
valid_genres = {"Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
                "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery",
                "Romance", "Sci-Fi", "Thriller", "War", "Western"}

def clean_genre_list(genres):
    return [g for g in genres if g in valid_genres]

clean_genres_udf = udf(clean_genre_list, ArrayType(StringType()))
movies = movies.withColumn("genre_list", split(col("genres"), "\\|"))
movies = movies.withColumn("genre_list", clean_genres_udf(col("genre_list")))

# Content-Based Feature Extraction
cv = CountVectorizer(inputCol="genre_list", outputCol="rawFeatures")
cv_model = cv.fit(movies)
vectorized = cv_model.transform(movies)

idf = IDF(inputCol="rawFeatures", outputCol="contentFeatures")
idf_model = idf.fit(vectorized)
content_features = idf_model.transform(vectorized)

normalizer = Normalizer(inputCol="contentFeatures", outputCol="normFeatures")
movies_content = normalizer.transform(content_features)

# ALS Collaborative Filtering
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative=True, coldStartStrategy="drop")
als_model = als.fit(ratings)
als_recommendations = als_model.recommendForAllUsers(10)

# Collaborative Filtering Function
def recommend_collaborative(user_id, top_n=10):
    als_flat = als_recommendations.selectExpr("userId", "explode(recommendations) as rec") \
                                  .select("userId", col("rec.movieId"), col("rec.rating").alias("als_score")) \
                                  .filter(col("userId") == user_id)
    result = als_flat.join(movies, on="movieId", how="inner") \
                     .select("title", "als_score") \
                     .orderBy(col("als_score").desc())
    print(f"\n🎬 Collaborative Filtering Recommendations for user {user_id}:")
    result.show(top_n, truncate=False)

# Content-Based Filtering Function
def recommend_by_movie_title(movie_title, top_n=15):
    target_row = movies_content.filter(col("title") == movie_title).select("normFeatures").first()
    if not target_row:
        print(f"Movie '{movie_title}' not found.")
        return

    target_vector = np.array(target_row["normFeatures"].toArray())

    def cosine_sim(v):
        v = np.array(v.toArray())
        return float(np.dot(target_vector, v) / (np.linalg.norm(target_vector) * np.linalg.norm(v))) \
            if np.linalg.norm(v) > 0 else 0.0

    cosine_udf = udf(cosine_sim, DoubleType())

    content_scores = movies_content.withColumn("content_score", cosine_udf(col("normFeatures"))) \
                                   .select("movieId", "title", "content_score") \
                                   .orderBy(col("content_score").desc())

    print(f"\n🎬 Content-Based Recommendations for '{movie_title}':")
    content_scores.filter(col("title") != movie_title).show(top_n, truncate=False)

# Hybrid Filtering Function
def recommend_hybrid(user_id, top_n=10):
    user_recs = als_recommendations.filter(col("userId") == user_id).select(explode("recommendations").alias("rec"))
    top_5_movie_df = user_recs.orderBy(col("rec.rating").desc()).select(col("rec.movieId")).limit(5)
    top_5_movie_ids = [row["movieId"] for row in top_5_movie_df.collect()]

    if not top_5_movie_ids:
        print(f"No top 5 movies found in ALS recommendation for user {user_id}.")
        return

    top_vectors = movies_content.filter(col("movieId").isin(top_5_movie_ids)) \
                                .select("movieId", "normFeatures") \
                                .collect()
    vectors_np = [np.array(row["normFeatures"].toArray()) for row in top_vectors]
    avg_vector = np.mean(vectors_np, axis=0)

    def cosine_similarity(v1):
        v1 = np.array(v1.toArray())
        return float(np.dot(avg_vector, v1) / (np.linalg.norm(avg_vector) * np.linalg.norm(v1))) \
            if np.linalg.norm(v1) > 0 else 0.0

    cosine_udf = udf(cosine_similarity, DoubleType())

    content_scores = movies_content.withColumn("content_score", cosine_udf(col("normFeatures"))) \
                                   .select("movieId", "title", "content_score")

    als_flat = als_recommendations.selectExpr("userId", "explode(recommendations) as rec") \
                                  .select("userId", col("rec.movieId"), col("rec.rating").alias("als_score"))

    hybrid = als_flat.join(content_scores, on="movieId", how="inner") \
                     .filter(col("userId") == user_id) \
                     .withColumn("hybrid_score", (col("als_score") + col("content_score")) / 2.0) \
                     .orderBy(col("hybrid_score").desc())

    print(f"\n🎬 Hybrid Recommendations for user {user_id}:")
    hybrid.select("title", "als_score", "content_score", "hybrid_score").show(top_n, truncate=False)

# User Menu
print("\nChoose recommendation type:")
print("1. Collaborative Filtering")
print("2. Content-Based Filtering")
print("3. Hybrid Filtering")
choice = input("Enter 1 / 2 / 3: ").strip()

if choice == "1":
    uid = int(input("Enter user ID: "))
    recommend_collaborative(uid)

elif choice == "2":
    title = input("Enter movie title: ")
    recommend_by_movie_title(title)

elif choice == "3":
    uid = int(input("Enter user ID: "))
    recommend_hybrid(uid)

else:
    print("Invalid choice.")
