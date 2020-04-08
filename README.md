# Good Film // Bad Film

A regression project in Python.

Predicts IMdB movie ratings (from 0 to 10) using pre-release movie metadata.

<img src="readme_images/imdb_logo.png" width="150">&#160;&#160;&#160;
<img src="readme_images/kaggle_logo.png" width="200">&#160;&#160;&#160;
<img src="readme_images/sklearn_logo.png" width="200">

***

## Best Model's Coefficients
Here are some of the strongest coefficients that emerged from my __lasso__ estimator, which performed better than a linear SVR, random forest regressor, or neural net

(Interaction terms are formatted as "variable_x*variable_y", e.g. "durations * actor_1_facebook_likes_box" is the interaction term composed of film runtimes and the facebook popularity of the leading actor.)

![Model Coefficients](readme_images/lasso_coefs.png)

How to interpret the above:

In my lasso regresion, 
* __Longer movies__ are predicted to get higher scores. 
* Longer movies with __popular leading actors__ get higher score predictions. 
* Documentaries in color get higher scores. 
* English language films in color tend to do worse. 
* Etc.

Roger Ebert is quoted as saying "No good film is too long and no bad movie is short enough."

## Best Model's Loss
![Lasso Loss](readme_images/lasso_loss.png)
These are the errors from my best model, the LassoCV estimator.

RMSE and Median Absolute Error are in terms of the ten-point IMdB rating.

__Best model's RMSE, compared to dummy regressor:__

![Lasso v dummy](readme_images/lasso_vs_dummy.png)

In plain English: If you were to naïvely guess the mean IMdB score for every sample in the test set, your average error in terms of IMdB critic score points would be around 1.1. My best model did somewhat better, with a root-mean-squared error of 0.92. 

# Process
## 1. Data Collection
* Grab this [kaggle imdb dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata):

<img src="readme_images/kaggle_tail.png" width="900">
<br><br>

* Extract IMdB ID's from each row, spam [themoviedb API](https://www.themoviedb.org/documentation/api) for more metadata, join it in:

<img src="readme_images/moviedb_head.png" width="900">

## 2. Data Cleaning
__Structured Data:__
* Keep only seemingly predictive columns (budget, actor popularities, genre, etc.)
* Drop duplicates
* Manage obvious null values (selective imputation, dropping rows)

__Null Value Counts by Column:__

<img src="readme_images/nulls.png" width="800">

* Manage non-obvious null values (e.g. zero when there is actually no info)
* Bin categorical columns into somewhat balanced metacategories

__E.g.: "Aspect Ratio" Value Counts Before Binning:__

<img src="readme_images/aspects_before.png">
<br><br>

__After Binning:__

<img src="readme_images/aspects_after.png">

(Here I used my familiarity with aspect ratios in film to condense this column.)

* Box-Cox transform non-normally distributed continuous value columns

__E.g. Film Budgets: Vanilla__

<img src="readme_images/budgets_vanilla.png">
<br>

__Film Budgets: Box-Cox Transformed__

<img src="readme_images/budgets_boxcox.png">
<br><br>

__Unstructured Data (Plot Synopses):__
* Tokenize corpus. Remove stopwords, punctuation
* Part-of-speech tag each token
* Lemmatize words by POS
* Vectorize plot texts with n-gram CountVectorizer
* TF-IDF normalize vectors

## 3. Feature Engineering
* One-hot encode categorical variables

E.g. "genre" column gets split into many binary variables:
<img src="readme_images/genre_encoding.png">
<br><br>

* __Train/Test split happens here__
* Drop multicollinear features

Multicollinearity of features as a heatmap:
<img src="readme_images/corr_matrix.png">

* Generate interaction terms
* MinMax scale features

## 4. Model Building / Benchmarking
__Models evaluated:__
* Baseline: Dummy regressor
* Linear: Lasso
* Non-Parametric: Linear SVR
* Tried incorporating vectorized text features (with PCA) — Not great
* Tree-based: Random forest regressor
* Neural Net

Used Pipeline + GridSearchCV to search for better hyperparameters on text-based regressor and random forest.

__Model Loss Metrics:__
<img src="readme_images/losses_allmodels.png">