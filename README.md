# Good Film // Bad Film

Constructs a classifier to predict whether or not a film will be received well — based on the plot synopsis, the social media popularity of its leading actors, its genre, and some other pre-release indicators.

___
## Process


We passed 5k+ movies from a Kaggle dataset into the OMdB API to get even more metadata. Here's a tiny peek at what we started with:

![](/readme_images/data.png)

We checked for redundant features and inspected correlation coefficients on a baseline logistic regression model:

![](/readme_images/correl.png)

![](/readme_images/coefs.png)


We even topic-modeled the plot synopses in order to extract latent unstructured features:

![](/readme_images/topics.png)

Our ultimate XGBoost classifier significantly outperformed a scrappy naive bayes classifier, and that same naive bayes classifier thankfully outperformed a baseline/dummy classifier.

![](/readme_images/ROC.png)