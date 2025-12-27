import os
import sys
import gc
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.datasets import fetch_openml
import kagglehub
import openml
from urllib.error import HTTPError

# Add the project root to sys.path to allow importing nanotabstar
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nanotabstar.preparation import TabSTARPreprocessor

# --- Test Dataset Catalog (from TabSTAR Paper Tables 12 & 13) ---
TEST_DATASETS = {
    # Classification (C01-C14)
    "MUL_CONSUMER_WOMEN_ECOMMERCE_CLOTHING_REVIEW": {"type": "openml", "id": 46659, "target": "Rating"},
    #"MUL_TRANSPORTATION_US_ACCIDENTS_MARCH23": {"type": "kaggle", "path": "sobhanmoosavi/us-accidents/US_Accidents_March23.csv", "target": "Severity"},
    "MUL_PROFESSIONAL_DATA_SCIENTIST_SALARY": {"type": "openml", "id": 46664, "target": "Salary_Range"},
    "BIN_SOCIAL_IMDB_GENRE_PREDICTION": {"type": "openml", "id": 46667, "target": "Genre"},
    "MUL_CONSUMER_PRODUCT_SENTIMENT": {"type": "openml", "id": 46651, "target": "Sentiment"},
    "MUL_SOCIAL_GOOGLE_QA_TYPE_REASON": {"type": "openml", "id": 46658, "target": "Reason"},
    "MUL_FOOD_MICHELIN_GUIDE_RESTAURANTS": {"type": "kaggle", "path": "ngshiheng/michelin-guide-restaurants-2021/michelin_my_maps.csv", "target": "Award"},
    "BIN_PROFESSIONAL_FAKE_JOB_POSTING": {"type": "openml", "id": 46655, "target": "fraudulent"},
    "BIN_SOCIAL_JIGSAW_TOXICITY": {"type": "openml", "id": 46654, "target": "toxic"},
    "MUL_FOOD_YELP_REVIEWS": {"type": "kaggle", "path": "omkarsabnis/yelp-reviews-dataset/yelp.csv", "target": "stars"},
    "MUL_SOCIAL_NEWS_CHANNEL_CATEGORY": {"type": "openml", "id": 46652, "target": "Category"},
    "MUL_FOOD_WINE_REVIEW": {"type": "openml", "id": 46653, "target": "Variety"},
    "BIN_PROFESSIONAL_KICKSTARTER_FUNDING": {"type": "openml", "id": 46668, "target": "State"},
    "MUL_HOUSES_MELBOURNE_AIRBNB": {"type": "openml", "id": 46665, "target": "Price_Range"},

    # Regression (R01-R36)
    "REG_CONSUMER_CAR_PRICE_CARDEKHO": {"type": "kaggle", "path": "sukritchatterjee/used-cars-dataset-cardekho/cars_details_merges.csv", "target": "selling_price"},
    "REG_TRANSPORTATION_USED_CAR_MERCEDES_BENZ_ITALY": {"type": "kaggle", "path": "bogdansorin/second-hand-mercedes-benz-registered-2000-2023-ita/mercedes-benz.csv", "sep": ";", "target": "price"},
    "REG_SOCIAL_ANIME_PLANET_RATING": {"type": "kaggle", "path": "hernan4444/animeplanet-recommendation-database-2020/anime.csv", "target": "rating"},
    "REG_PROFESSIONAL_ML_DS_AI_JOBS_SALARIES": {"type": "url", "url": "https://ai-jobs.net/salaries/download/salaries.csv", "target": "salary_in_usd"},
    "REG_CONSUMER_BABIES_R_US_PRICES": {"type": "url", "url": "http://pages.cs.wisc.edu/~anhai/data/784_data/baby_products/csv_files/babies_r_us.csv", "target": "price"},
    "REG_PROFESSIONAL_EMPLOYEE_SALARY_MONTGOMERY": {"type": "openml", "id": 42125, "target": "Current_Annual_Salary"},
    "REG_SOCIAL_SPOTIFY_POPULARITY": {"type": "kaggle", "path": "maharshipandya/-spotify-tracks-dataset/dataset.csv", "target": "popularity"},
    "REG_HOUSES_CALIFORNIA_PRICES_2020": {"type": "openml", "id": 46669, "target": "Median_House_Value"},
    "REG_SPORTS_FIFA22_WAGES": {"type": "openml", "id": 45012, "target": "Wage_EUR"},
    "REG_FOOD_COFFEE_REVIEW": {"type": "kaggle", "path": "hanifalirsyad/coffee-scrap-coffeereview/coffee_clean.csv", "target": "rating"},
    "REG_CONSUMER_BIKE_PRICE_BIKEWALE": {"type": "url", "url": "http://pages.cs.wisc.edu/~anhai/data/784_data/bikes/csv_files/bikewale.csv", "target": "price"},
    "REG_TRANSPORTATION_USED_CAR_PAKISTAN": {"type": "kaggle", "path": "mustafaimam/used-car-prices-in-pakistan-2021/Used_car_prices_in_Pakistan_cleaned.csv", "target": "Price"},
    "REG_CONSUMER_BOOK_PRICE_PREDICTION": {"type": "openml", "id": 46663, "target": "Price"},
    "REG_CONSUMER_AMERICAN_EAGLE_PRICES": {"type": "openml", "id": 46656, "target": "Price"},
    "REG_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER": {"type": "url", "url": "https://opendata.vancouver.ca/api/records/1.0/download/?dataset=employee-remuneration-and-expenses-earning-over-75000&format=csv", "sep": ";", "target": "total_remuneration_and_expenses"},
    "REG_SOCIAL_FILMTV_MOVIE_RATING_ITALY": {"type": "kaggle", "path": "stefanoleone992/filmtv-movies-dataset/filmtv_movies.csv", "target": "avg_vote"},
    #"REG_PROFESSIONAL_COMPANY_EMPLOYEES_SIZE": {"type": "kaggle", "path": "peopledatalabssf/free-7-million-company-dataset/companies_sorted.csv", "target": "size_range"},
    "REG_SOCIAL_MUSEUMS_US_REVENUES": {"type": "kaggle", "path": "markusschmitz/museums/museums_prep.csv", "target": "Revenue"},
    "REG_FOOD_WINE_VIVINO_SPAIN": {"type": "kaggle", "path": "joshuakalobbowles/vivino-wine-data/vivino.csv", "target": "price"},
    "REG_FOOD_ALCOHOL_WIKILIQ_PRICES": {"type": "kaggle", "path": "limtis/wikiliq-dataset/spirits_data.csv", "target": "price"},
    "REG_FOOD_BEER_RATINGS": {"type": "kaggle", "path": "ruthgn/beer-profile-and-ratings-data-set/beer_profile_and_ratings.csv", "target": "review_overall"},
    "REG_SOCIAL_KOREAN_DRAMA": {"type": "kaggle", "path": "noorrizki/top-korean-drama-list-1500/kdrama_list.csv", "target": "Score"},
    "REG_SOCIAL_VIDEO_GAMES_SALES": {"type": "kaggle", "path": "gregorut/videogamesales/vgsales.csv", "target": "Global_Sales"},
    "REG_FOOD_ZOMATO_RESTAURANTS": {"type": "kaggle", "path": "himanshupoddar/zomato-bangalore-restaurants/zomato.csv", "target": "rate"},
    "REG_SOCIAL_MOVIES_DATASET_REVENUE": {"type": "kaggle", "path": "rounakbanik/the-movies-dataset/movies_metadata.csv", "target": "revenue"},
    "REG_SPORTS_NBA_DRAFT_VALUE_OVER_REPLACEMENT": {"type": "kaggle", "path": "mattop/nba-draft-basketball-player-data-19892021/nbaplayersdraft.csv", "target": "vorp"},
    "REG_SOCIAL_BOOKS_GOODREADS": {"type": "url", "url": "http://pages.cs.wisc.edu/~anhai/data/784_data/books2/csv_files/goodreads.csv", "target": "rating"},
    "REG_SOCIAL_MOVIES_ROTTEN_TOMATOES": {"type": "url", "url": "http://pages.cs.wisc.edu/~anhai/data/784_data/movies1/csv_files/rotten_tomatoes.csv", "target": "rating"},
    "REG_TRANSPORTATION_USED_CAR_SAUDI_ARABIA": {"type": "kaggle", "path": "turkibintalib/saudi-arabia-used-cars-dataset/UsedCarsSA_Clean_EN.csv", "target": "Price"},
    "REG_FOOD_RAMEN_RATINGS_2022": {"type": "kaggle", "path": "ankanhore545/top-ramen-ratings-2022/Top Ramen Ratings .csv", "target": "Stars"},
    "REG_PROFESSIONAL_SCIMAGOJR_ACADEMIC_IMPACT": {"type": "url", "url": "https://www.scimagojr.com/journalrank.php?out=xls", "sep": ";", "target": "SJR"},
    "REG_FOOD_CHOCOLATE_BAR_RATINGS": {"type": "kaggle", "path": "rtatman/chocolate-bar-ratings/flavors_of_cacao.csv", "target": "Rating"},
    "REG_CONSUMER_MERCARI_ONLINE_MARKETPLACE": {"type": "openml", "id": 46660, "target": "price"},
    "REG_FOOD_WINE_POLISH_MARKET_PRICES": {"type": "kaggle", "path": "skamlo/wine-price-on-polish-market/wine.csv", "target": "price"},
    "REG_SOCIAL_BOOK_READABILITY_CLEAR": {"type": "kaggle", "path": "verracodeguacas/clear-corpus/CLEAR.csv", "target": "BT_easiness"},
    "REG_CONSUMER_JC_PENNEY_PRODUCT_PRICE": {"type": "openml", "id": 46661, "target": "sale_price"},
}

MAX_SAMPLES_TEST = 10000 # Limit test samples for speed

def download_dataset(name, config):
    print(f"Downloading {name}...")
    try:
        if config["type"] == "openml":
            dataset = openml.datasets.get_dataset(config["id"], download_data=True)
            # Use specified target if available, else default
            target_col = config.get("target", dataset.default_target_attribute)
            X, y, _, _ = dataset.get_data(target=target_col)
            return X, y
        elif config["type"] == "kaggle":
            dataset_name, file = config["path"].rsplit('/', 1)
            dir_path = kagglehub.dataset_download(dataset_name)
            file_path = os.path.join(dir_path, file)
            sep = config.get("sep", ",")
            df = pd.read_csv(file_path, sep=sep)
            
            target_col = config.get("target")
            if target_col not in df.columns:
                # Try to find it case-insensitively
                for col in df.columns:
                    if col.lower() == target_col.lower():
                        target_col = col
                        break
            
            if target_col not in df.columns:
                print(f"Warning: Target {target_col} not found in {name}. Using last column.")
                target_col = df.columns[-1]
            
            # Drop columns if specified
            drop_cols = config.get("drop", [])
            if name == "MUL_TRANSPORTATION_US_ACCIDENTS_MARCH23":
                drop_cols.append("ID")
            
            X = df.drop(columns=[target_col] + [c for c in drop_cols if c in df.columns])
            y = df[target_col]
            return X, y
        elif config["type"] == "url":
            sep = config.get("sep", ",")
            df = pd.read_csv(config["url"], sep=sep)
            target_col = config.get("target")
            
            if target_col not in df.columns:
                print(f"Warning: Target {target_col} not found in {name}. Using last column.")
                target_col = df.columns[-1]
                
            X = df.drop(columns=[target_col])
            y = df[target_col]
            return X, y
    except Exception as e:
        print(f"Error downloading {name}: {e}")
        return None, None

def create_test_corpus(output_path="data/test_corpus_tabstar.h5"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, "w") as h5f:
        for ds_name, config in tqdm(TEST_DATASETS.items(), desc="Processing Test Datasets"):
            X, y = download_dataset(ds_name, config)
            if X is None or y is None:
                continue
            
            # Basic cleaning: drop rows with NaN in target
            mask = y.notna()
            X = X[mask]
            y = y[mask]

            if len(X) > MAX_SAMPLES_TEST:
                X = X.sample(MAX_SAMPLES_TEST, random_state=42)
                y = y.loc[X.index]

            is_cls = ds_name.startswith("BIN") or ds_name.startswith("MUL")
            preprocessor = TabSTARPreprocessor(is_cls=is_cls)
            
            try:
                preprocessor.fit(X, y)
                data = preprocessor.transform(X, y)
                
                grp = h5f.create_group(ds_name)
                grp.create_dataset("feature_texts", data=data.x_txt.astype(h5py.string_dtype()), compression="gzip")
                grp.create_dataset("feature_num_values", data=data.x_num, compression="gzip")
                grp.create_dataset("labels", data=data.y.values if hasattr(data.y, 'values') else data.y, compression="gzip")
                
                # Target descriptions
                target_texts = []
                if is_cls:
                    for val in preprocessor.y_values:
                        target_texts.append(f"Target Feature: {preprocessor.y_name}\nFeature Value: {val}")
                else:
                    target_texts.append(f"Numerical Target Feature: {preprocessor.y_name}")
                
                grp.create_dataset("target_texts", data=np.array(target_texts, dtype=h5py.string_dtype()))
                
                grp.attrs["task_type"] = "classification" if is_cls else "regression"
                grp.attrs["d_output"] = data.d_output
                grp.attrs["n_features"] = data.x_txt.shape[1]
                
                print(f"  -> Saved {ds_name} ({len(X)} samples, {data.x_txt.shape[1]} features)")
                
            except Exception as e:
                print(f"Error processing {ds_name}: {e}")
            
            del X, y
            gc.collect()

if __name__ == "__main__":
    create_test_corpus()
