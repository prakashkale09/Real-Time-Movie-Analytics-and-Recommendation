from pyspark.sql import SparkSession
from pyspark.sql.functions import col,split,explode,udf
spark=SparkSession.builder\
      .appName("HybridMovieRecommender")\
      .getOrCreate()
#step1: Load the dataset

ratings=spark.read.option("header",True).option("inferSchema",True).csv("hdfs://localhost:9000/user/prakash/moviedata/ratings.csv")
movies=spark.read.option("header",True).option("inferSchema",True).csv("hdfs://localhost:9000/user/prakash/moviedata/movies.csv")

ratings.show(5)
movies.show(5)
movies.printSchema()
ratings.printSchema()

#step2: Merge the both dataset

data=ratings.join(movies,on="movieId",how="inner")

data.show(5)

#step3: Content-based filtering

movies=movies.withColumn("genre_list", split(col("genres"),"\\|"))
movies.show(5)
movies.printSchema()

# Find all genres from genre column
all_genres = movies.select(explode(col("genre_list")).alias("genre"))
unique_genres=all_genres.distinct()
unique_genres.show(truncate=False)
print(unique_genres)