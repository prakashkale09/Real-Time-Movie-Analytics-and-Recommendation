from pyspark.sql import SparkSession
from pyspark.sql.functions import col,split,explode,udf
from pyspark.sql.types import DoubleType, ArrayType, StringType
from pyspark.ml.feature import CountVectorizer, IDF, Normalizer
spark=SparkSession.builder\
      .appName("HybridMovieRecommender")\
      .getOrCreate()
#step1: Load the dataset

ratings=spark.read.option("header",True).option("inferSchema",True).csv("hdfs://localhost:9000/user/prakash/moviedata/ratings.csv")
movies=spark.read.option("header",True).option("inferSchema",True).csv("hdfs://localhost:9000/user/prakash/moviedata/movies.csv")

#EDA
ratings.show(5)
movies.show(5)
movies.printSchema()
ratings.printSchema()

# Find all genres from genre column
movies = movies.withColumn("genre_list", split(col("genres"), "\\|"))
all_genres = movies.select(explode(col("genre_list")).alias("genre"))
unique_genres=all_genres.distinct()
unique_genres.show(truncate=False)
print(unique_genres)

valid_genres = {"Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
                "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery",
                "Romance", "Sci-Fi", "Thriller", "War", "Western"}

#Remove Generes which are not valid or noisy
def clean_genre_list(genres):
    return [g for g in genres if g in valid_genres]

clean_genres_udf = udf(clean_genre_list, ArrayType(StringType()))
movies = movies.withColumn("genre_list", split(col("genres"), "\\|"))
movies = movies.withColumn("genre_list", clean_genres_udf(col("genre_list")))

movies.show(5)

#step 2:Content-based filtering
#Convert the genre_list into numerical vectors
cv=CountVectorizer(inputCol="genre_list",outputCol="rawFeatures")
cv_model=cv.fit(movies)
vectorized=cv_model.transform(movies)

idf=IDF(inputCol="rawFeatures",outputCol="contentFetures")
idf_model=idf.fit(vectorized)
content_Features=idf_model.transform(vectorized)

content_Features.show()

normalizer = Normalizer(inputCol="contentFeatures", outputCol="normFeatures")
movies_content = normalizer.transform(content_Features)
