#%%
import pandas as pd
import warnings
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

titles = pd.read_csv("titles.csv")

# titles.columns
# titles.head()
# titles.info()

credits = pd.read_csv("credits.csv")

credits.columns
credits.head()
credits.tail(10)

# titles.shape
# credits.shape


#Filling Null Values
# titles.isnull().sum()
# credits.isnull().sum()

x1 = titles['age_certification'].mode()
# x1
titles['age_certification'] = titles['age_certification'].fillna('TV-MA')

x2 = titles['imdb_score'].mean()
# x2
titles['imdb_score'] = titles['imdb_score'].fillna(7)

x3 = titles['imdb_votes'].mean()
# x3
titles['imdb_votes'] = titles['imdb_votes'].fillna(23407)

x4 = titles['tmdb_popularity'].mean()
# x4
titles['tmdb_popularity'] = titles['tmdb_popularity'].fillna(23)

x5 = titles['tmdb_score'].mean()
# x5
titles['tmdb_score'] = titles['tmdb_score'].fillna(7)

x6 = credits['character'].mode()
# x6
credits['character'] = credits['character'].fillna('Self')

# credits.isnull().sum()

#making a copy of titles data
titles1 = titles.dropna(subset=['title'])

# titles1.isnull().sum()

#%%
#Deleting columns irrelevant to model building
del titles1['seasons']
del titles1['imdb_id']
del titles1['production_countries']

# titles1.isnull().sum()
# titles1.head()
# credits.head()


#credits data is not grouped, so I am grouping the data based on movie ID, credits1 is the dataframe when this data will be stored
credits1 = credits.groupby('id').agg(lambda x: x.tolist())

# credits1.info()
# credits1.head()


#Transforming name and character list columns to string
mask = credits1['name'].notnull()
credits1.loc[mask, 'name'] = [', '.join(map(str, x)) for x in credits1.loc[mask, 'name']]
credits1.loc[mask, 'character'] = [', '.join(map(str, x)) for x in credits1.loc[mask, 'character']]


# credits1.head()
# titles1.shape
# credits1.shape


#merging title1 and credits1 data
movies = pd.merge(titles1, credits1,on='id')
# movies.head(10)
# movies.info()


movies=movies.drop(['person_id','role'], axis = 1) #dropping irrelevant columns after merge

#renaming 'name' column to 'cast'
movies.rename(
    columns={"name":"cast"}
          ,inplace=True)

# movies.head()
# movies['type'].value_counts() #number of movies and shows


#The genres column looks like a list but is actually a string column.
#Below code converts it to a list, then makes a string out of it and
#then removes unnecessary quotes
movies['genres'] = movies.genres.apply(lambda x: x[1:-1].split(',')) #convert to list

# movies.genres[0]

#%%
#convert to string
j=0
for i in movies.genres:
    i = ",".join(i)
    movies.genres[j] = i
    j+=1

# movies.genres[0]

#remove single quotes over each genre
j=0
for i in movies.genres:
    i = i.replace("\'","")
    movies.genres[j] = i
    j+=1

# movies.genres[0]
# movies.head()

#%%
#Model Building
all_movies = movies.loc[movies.type=='MOVIE',:].reset_index()
all_movies.title = all_movies.title.str.lower()
all_movies['index'] = all_movies.index
# all_movies.head()

# all_movies['type'].value_counts()

tv_shows = movies.loc[movies.type=='SHOW',:].reset_index()
tv_shows.title = tv_shows.title.str.lower()
tv_shows['index'] = tv_shows.index
# tv_shows.head()

all_movies.duplicated().sum()
tv_shows.duplicated().sum()

#getting index of tv_shows
index = tv_shows.index
number_of_rows_tv = len(index)

#getting index of all_movies
index = all_movies.index
number_of_rows_movies = len(index)

#%%
# # List of Latest 15 movies
latest_15 = all_movies.sort_values(by = 'release_year', ascending = False).head(15)
# print(latest_15[["title", "release_year"]])

#%%
features = ['description', 'cast', 'genres', 'age_certification']

for feature in features:
    all_movies[feature] = all_movies[feature].fillna('')
def combine_features(row):
    return row['description']+" "+row['cast']+" "+row['genres']+" "+row['age_certification']
all_movies["combine_features"] = all_movies.apply(combine_features,axis=1)
# print("Combined Features:", all_movies["combine_features"])

cv = CountVectorizer()
count_matrix = cv.fit_transform(all_movies["combine_features"])
cosine_sim = cosine_similarity(count_matrix)

#get index of movie from title
def title_from_index(index):
    return all_movies[all_movies.index == index]["title"].values[0]

def title_from_index(df, index):
    return df[df.index == index]["title"].values[0]

def index_from_title(df, title):
    return df[df.title == title]["index"].values[0]

def selectmovie(movie_user_likes):
    try:
        movie_user_likes = movie_user_likes.lower()
        movie_index = index_from_title(all_movies, movie_user_likes)
        similar_movies = list(enumerate(cosine_sim[movie_index]))
        sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
        i=0
        dicts = {}
        for element in sorted_similar_movies:
            dicts[i] = title_from_index(all_movies, element[0])
            i=i+1
            if i>=10:
                return(dicts)
                break
    except:
        d = {0:'Movie not found on Netflix'}
        return(d)


#OUTPUT
#movies_dict = selectmovie("blade ii")
#print("Top 10 similar movies are:\n")
#print(movies_dict)


