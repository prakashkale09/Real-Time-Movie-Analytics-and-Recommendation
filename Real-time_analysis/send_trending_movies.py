import requests
import json
from datetime import datetime, timedelta
import os
import subprocess

API_KEY = "5662ff10308be2f922c3b6d7a989232c"

def fetch_genre_map():
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        genres = response.json().get("genres", [])
        return {genre["id"]: genre["name"] for genre in genres}
    return {}

def fetch_movies(start_date, end_date, language_code=None):
    movies = []
    page = 1
    while page <= 2:
        url = (
            f"https://api.themoviedb.org/3/discover/movie"
            f"?api_key={API_KEY}"
            f"&sort_by=popularity.desc"
            f"&primary_release_date.gte={start_date}"
            f"&primary_release_date.lte={end_date}"
            f"&page={page}"
        )
        if language_code:
            url += f"&with_original_language={language_code}"

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get("results", [])
            if not data:
                break
            movies.extend(data)
            page += 1
        else:
            break
    return movies

def main():
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=10 * 365)
    genre_map = fetch_genre_map()

    movies = fetch_movies(start_date, end_date) + fetch_movies(start_date, end_date, language_code="hi")
    seen_ids = set()
    today = datetime.today()
    local_file_path = f"trending_movies_{today.strftime('%Y_%m_%d')}.json"

    # Open file for line-delimited JSON
    with open(local_file_path, 'w') as f:
        for movie in movies:
            movie_id = movie.get('id')
            release_date = movie.get('release_date', '')

            if movie_id and movie_id not in seen_ids and release_date:
                genre_ids = movie.get('genre_ids', [])
                genres = [genre_map.get(gid, "Unknown") for gid in genre_ids]

                movie_data = {
                    'id': movie_id,
                    'title': movie.get('title', 'N/A'),
                    'original_language': movie.get('original_language', 'N/A'),
                    'vote_average': movie.get('vote_average', 0.0),
                    'release_date': release_date,
                    'popularity': movie.get('popularity', 0.0),
                    'genres': genres,
                    'year': today.year,
                    'month': today.month,
                    'day': today.day
                }

                json.dump(movie_data, f)
                f.write('\n')
                seen_ids.add(movie_id)

    print(f"[INFO] Saved {len(seen_ids)} records to {local_file_path}")

    # HDFS target path
    hdfs_dir = f"/user/movies/trending/year={today.year}/month={today.month:02d}/day={today.day:02d}/"
    subprocess.run(["hdfs", "dfs", "-mkdir", "-p", hdfs_dir])
    subprocess.run(["hdfs", "dfs", "-put", "-f", local_file_path, hdfs_dir])
    print(f"[INFO] Uploaded to HDFS at {hdfs_dir}")

if __name__ == "__main__":
    main()
