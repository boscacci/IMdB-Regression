# Good Film // Bad Film

Constructs a classifier to predict whether or not a film will be received well — based on the plot synopsis, the social media popularity of its leading actors, its genre, and some other pre-release indicators.

___
## Process


We passed 5k+ movies from a Kaggle dataset into the OMdB API to get even more metadata. Here's a tiny peek at what we started with:

![](/media/data.png)

We checked for redundant features and inspected correlation coefficients on a baseline logistic regression model:

![](/media/correl.png)

![](/media/coefs.png)


We even topic-modeled the plot synopses in order to extract latent unstructured features:

![](/media/topics.png)

Ultimately we ended up running with an XGBoost classifier that performed respectably well:

![](/media/ROC_curve.png)