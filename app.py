from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import difflib

app = Flask(__name__)

# Dil sözlüğü
translations = {
    "en": {
        "title": "Movie Recommendation System",
        "select_option": "Choose a recommendation method:",
        "option1": "Similar movies by movie name",
        "option2": "Movies with this actor",
        "option3": "Movies in this genre",
        "option4": "Movies by this director",
        "option5": "Popular movies in this genre",
        "placeholder": "Enter movie, actor, genre or director...",
        "show": "Show Recommendations",
        "not_found": "No results found for your input."
    },
    "tr": {
        "title": "Film Öneri Sistemi",
        "select_option": "Bir öneri yöntemi seçin:",
        "option1": "Film adına göre benzer filmler",
        "option2": "Oyuncunun oynadığı filmler",
        "option3": "Bu türdeki filmler",
        "option4": "Bu yönetmenin filmleri",
        "option5": "Bu türde en popüler filmler",
        "placeholder": "Film, oyuncu, tür ya da yönetmen giriniz...",
        "show": "Filmleri Göster",
        "not_found": "Girdiğiniz değer için sonuç bulunamadı."
    }
}

# Veri setlerini yükle
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# TF-IDF vektörleştirme
movies['overview'] = movies['overview'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Dil seçimi
def get_lang():
    lang = request.args.get("lang", "en")
    return lang if lang in translations else "en"

@app.route("/", methods=["GET"])
def index():
    lang_code = get_lang()
    lang = translations[lang_code]
    return render_template("index.html", lang=lang, lang_code=lang_code)

@app.route("/result", methods=["POST"])
def result():
    lang_code = request.form.get("lang_code", "en")
    lang = translations[lang_code]
    selection = request.form["secim"]
    user_input = request.form["girdi"].strip()
    result = []

    if selection == 'film_adi':
        matches = difflib.get_close_matches(user_input, movies['title'], n=1, cutoff=0.5)
        if not matches:
            return render_template('sonuc.html', baslik=lang["not_found"], sonuc=[], lang=lang, lang_code=lang_code)
        closest_title = matches[0]
        idx = indices[closest_title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]  # 10 film
        movie_indices = [i[0] for i in sim_scores]
        for i in movie_indices:
            result.append({
                "title": movies.iloc[i]["title"],
                "overview": movies.iloc[i]["overview"]
            })
        baslik = f"{closest_title} →"

    elif selection == 'oyuncu':
        credits['cast'] = credits['cast'].fillna('')
        for i, row in credits.iterrows():
            if user_input.lower() in row['cast'].lower():
                result.append({
                    "title": movies.loc[i, 'title'],
                    "overview": movies.loc[i, 'overview']
                })
        result = result[:10]
        baslik = f"{user_input} →"

    elif selection == 'tur':
        for i, row in movies.iterrows():
            if user_input.lower() in row['genres'].lower():
                result.append({
                    "title": row['title'],
                    "overview": row['overview']
                })
        result = result[:10]
        baslik = f"{user_input.title()} →"

    elif selection == 'yonetmen':
        credits['crew'] = credits['crew'].fillna('')
        for i, row in credits.iterrows():
            if user_input.lower() in row['crew'].lower():
                result.append({
                    "title": movies.loc[i, 'title'],
                    "overview": movies.loc[i, 'overview']
                })
        result = result[:10]
        baslik = f"{user_input} →"

    elif selection == 'populer_tur':
        temp = movies[movies['genres'].str.lower().str.contains(user_input.lower())]
        temp = temp.sort_values(by='popularity', ascending=False).head(10)
        for _, row in temp.iterrows():
            result.append({
                "title": row['title'],
                "overview": row['overview']
            })
        baslik = f"Popular {user_input.title()} movies →"

    else:
        baslik = lang["not_found"]

    return render_template('sonuc.html', baslik=baslik, sonuc=result, lang=lang, lang_code=lang_code)

if __name__ == '__main__':
    app.run(debug=True)
