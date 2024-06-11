## Table of Contents

- [Overview](#Overview)
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
  - [Filling Null Values](#filling-null-values)
  - [Merging Datasets](#merging-datasets)
  - [Cleaning Genres Column](#cleaning-genres-column)
- [Visualization](#visualization)
- [Model Building](#model-building)
  - [Feature Selection](#feature-selection)
  - [Vectorization and Similarity Calculation](#vectorization-and-similarity-calculation)
  - [Recommendation Function](#recommendation-function)
- [Model Deployment](#model-deployment)
- [Example](#example)
  
# Overview

This project is a movie recommendation system built using Count Vectorizer and Cosine Similarity from the Scikit-Learn library. It was developed during a 4-week internship at Delta Sigma Technologies in May-June 2022.

The main objective is to suggest movies that a user might like based on the movie input they provide.

## Data

The project uses two datasets:
- `titles.csv`: Contains data about movies with 5806 entries. Columns: `['id', 'title', 'type', 'description', 'release_year', 'age_certification', 'runtime', 'genres', 'production_countries', 'seasons', 'imdb_id', 'imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']`
- `credits.csv`: Contains data about actors and directors with 77213 entries. Columns: `['person_id', 'id', 'name', 'character', 'role']`

## Data Preprocessing

1. **Filling Null Values**:
   - `credits.csv`: Replaced null values in `character` with the mode.
   - `titles.csv`: Replaced null values in `imdb_score`, `imdb_votes`, `tmdb_score`, `tmdb_popularity` with column means; `age_certification` with the mode; dropped rows with null `title` and `description`.

2. **Merging Datasets**:
   - Dropped unnecessary columns from both datasets.
   - Grouped and combined credits data by `id`.
   - Merged `credits` and `titles` datasets on the `id` column.

3. **Cleaning Genres Column**:
   - Converted genre strings to lists and then to cleaned strings.

## Visualization

- Created visualizations including pie charts, bar graphs, crosstabs, and histograms to explore the data.

![Pie Chart](img/01.jpg?raw=true "Pie Chart")

![Bar graph](img/02.jpg?raw=true "Bar graph") 

![Latest Movies](img/03.jpg?raw=true "Latest Movies") 

## Model Building

![Model Building](img/04.jpg?raw=true "Model Buildings") 

1. **Feature Selection**:
   - Created a new dataframe `all_movies` for movies and `tv_shows` for TV shows.
   - Selected columns: `['description', 'cast', 'genres', 'age_certification']` and combined them into a `combine_features` column.

2. **Vectorization and Similarity Calculation**:
   - Used `CountVectorizer` to transform the `combine_features` column into vectors.
   - Calculated cosine similarity between movie vectors.

3. **Recommendation Function**:
   - Defined functions to get movie titles from indices and indices from titles.
   - Created the `selectmovie` function to get top 10 similar movie recommendations based on a given movie title.

## Model Deployment

![Flask](img/06.jpg?raw=true "Flask") 

- Used Flask to create an API for the recommendation system.
- Developed the `Netflix_app.py` file to handle API requests and return recommendations in JSON format.
- Tested the API using Postman.

![Postman](img/07.jpg?raw=true "Postman") 

## Example

If the input to the `selectmovie` function is the movie "Blade", the output will be a list of 10 similar movie recommendations.

![Example](img/05.jpg?raw=true "Example") 
