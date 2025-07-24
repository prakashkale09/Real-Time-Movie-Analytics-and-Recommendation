from pyspark.sql import SparkSession
from pyspark.sql.functions import col,split,explode,udf
from pyspark.sql.types import DoubleType, ArrayType, StringType
from pyspark.ml.feature import CountVectorizer, IDF, Normalizer
from pyspark.ml.recommendation import ALS
import numpy as np
spark = (SparkSession.builder\
    .appName("HybridMovieRecommender")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")     
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.executor.heartbeatInterval", "60s")
    .config("spark.network.timeout", "800s")
    .getOrCreate())
#step1: Load the dataset

ratings=spark.read.option("header",True).option("inferSchema",True).csv("hdfs://localhost:9000/user/prakash/moviedata/ratings.csv")
movies=spark.read.option("header",True).option("inferSchema",True).csv("hdfs://localhost:9000/user/prakash/moviedata/movies.csv")

#EDA
#ratings.show(5)
#movies.show(5)
#movies.printSchema()
#ratings.printSchema()

# Find all genres from genre column
movies = movies.withColumn("genre_list", split(col("genres"), "\\|"))
all_genres = movies.select(explode(col("genre_list")).alias("genre"))
unique_genres=all_genres.distinct()
unique_genres.show(truncate=False)
#print(unique_genres)

valid_genres = {"Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
                "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery",
                "Romance", "Sci-Fi", "Thriller", "War", "Western"}

#Remove Generes which are not valid or noisy
def clean_genre_list(genres):
    return [g for g in genres if g in valid_genres]

clean_genres_udf = udf(clean_genre_list, ArrayType(StringType()))
movies = movies.withColumn("genre_list", split(col("genres"), "\\|"))
movies = movies.withColumn("genre_list", clean_genres_udf(col("genre_list")))

#movies.show(5)

#step 2:Content-based filtering
#Convert the genre_list into numerical vectors
cv=CountVectorizer(inputCol="genre_list",outputCol="rawFeatures")
cv_model=cv.fit(movies)
vectorized=cv_model.transform(movies)

idf=IDF(inputCol="rawFeatures",outputCol="contentFeatures")
idf_model=idf.fit(vectorized)
content_Features=idf_model.transform(vectorized)

#content_Features.show()

normalizer = Normalizer(inputCol="contentFeatures", outputCol="normFeatures")
movies_content = normalizer.transform(content_Features)

#Step 3:Collaborative Filtering
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative=True, coldStartStrategy="drop")
als_model = als.fit(ratings)
als_recommendations = als_model.recommendForAllUsers(10)


#Step 4: Hybrid Recommender for a user
from pyspark.sql.functions import explode
target_user_id = 3

user_recs = als_recommendations.filter(col("userId") == target_user_id).select(explode("recommendations").alias("rec"))
top_5_movie_df = user_recs.orderBy(col("rec.rating").desc()).select(col("rec.movieId")).limit(5)
top_5_movie_ids=[row["movieId"] for row in top_5_movie_df.collect()]

if top_5_movie_ids:
    # Get content vectors of those top 5 movies
    top_vectors = movies_content.filter(col("movieId").isin(top_5_movie_ids)) \
                                .select("movieId", "normFeatures") \
                                .collect()
     # Compute the average vector
    vectors_np = [np.array(row["normFeatures"].toArray()) for row in top_vectors]
    avg_vector = np.mean(vectors_np, axis=0)

    # Define cosine similarity UDF
    def cosine_similarity(v1):
        v1 = np.array(v1.toArray())
        return float(np.dot(avg_vector, v1) / (np.linalg.norm(avg_vector) * np.linalg.norm(v1))) \
            if np.linalg.norm(v1) > 0 else 0.0
    
    cosine_udf = udf(cosine_similarity, DoubleType())

    # Compute content similarity scores
    content_scores = movies_content.withColumn("content_score", cosine_udf(col("normFeatures"))) \
                                   .select("movieId", "title", "content_score")
    # Flatten ALS predictions
    als_flat = als_recommendations.selectExpr("userId", "explode(recommendations) as rec") \
                                  .select("userId", col("rec.movieId"), col("rec.rating").alias("als_score"))

    # Join ALS + Content and calculate hybrid score
    hybrid = als_flat.join(content_scores, on="movieId", how="inner") \
                     .filter(col("userId") == target_user_id) \
                     .withColumn("hybrid_score", (col("als_score") + col("content_score")) / 2.0) \
                     .orderBy(col("hybrid_score").desc())

    # Show top 10 recommendations
    hybrid.select("title", "als_score", "content_score", "hybrid_score").show(5, truncate=False)

else:
    print(f"No top 5 movies found in ALS recommendation for user {target_user_id}.")
    


