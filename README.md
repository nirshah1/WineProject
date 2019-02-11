# Classifying Wine Variety by Review & Price

BACKGROUND:

In order to become a certified sommelier (wine expert), The Court of Master Sommeliers administers a Tasting Examination that requires candidates to describe and identify the grape varieties of six different wines in 25 minutes. Often times, in preparation for the exam candidates will associate a multitude of smells and tastes with certain wine varieties. As a study mechanism, they will usually recite these sensory descriptors out loud as they taste each of the wines; doing this helps the candidates map the descriptors to a particular variety.

After watching the documentary Somm on Netflix, which documents the struggles of four candidates attempting to pass the Sommelier exam, I became intrigued by this subject matter. I wondered if I could use machine learning methods to train a model that would identify wines through blind tasting like a master sommelier would. Since at least the computers that I own cannot taste the wines themselves, I figured that I could use the text from professional wine reviews as a feature from which the machine would use to predict the grape variety. Wine reviews typically include many sensory descriptors to describe a particular wine variety, so in a way they imitate the recitation that sommelier candidates do to prepare for the exam.

Fortunately, I discovered a Kaggle dataset of more than 150,000 wine reviews, scraped by the Kaggle user Zack Thoutt from the WineEnthusiast website. The dataset is a CSV file with 10 columns and 150k rows of wine reviews. Along with the wine review itself, the dataset includes the country of origin, designation, points (as scored by the reviewer), price, province, region(s), and winery for each wine variety. There are 623 unique wine varieties from 49 different countries.

The dataset can be seen here: https://www.kaggle.com/zynicide/wine-reviews

GOAL:

My goal was to predict as accurately as possible the variety of a wine using the words from its review and its price. My plan was to generate a predictive model on top of a kernel that was previously made for this dataset on Kaggle. In this kernel, the Kaggle user “CarKar” generated a logistic regression model in Python to classify the wine varieties in this dataset, using the review and price as predictors. He then tested his model on some test data from the dataset and was able to produce an accuracy score of about 53%. I used several other classification methods to see if I could produce a better accuracy score. My plan was to use the same two predictors (review and price) while also utilizing a variable selection method to see if removing or adding predictors from the dataset will have any effect on the accuracy score.

The accuracy score was based on the correct classification rate for the model’s performance on the test data. I generated and assessed these models in Python, using the PyCharm IDE. As a sampling method, I used 10-fold cross-validation for model assessment and comparison.

RESULTS:
