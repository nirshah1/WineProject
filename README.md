# WineProject
Determining the wine variety based on its review and price


In order to become a certified sommelier (wine expert), The Court of Master Sommeliers administers a Tasting Examination that requires candidates to describe and identify the grape varieties of six different wines in 25 minutes. Often times, in preparation for the exam candidates will associate a multitude of smells and tastes with certain wine varieties. As a study mechanism, they will usually recite these sensory descriptors out loud as they taste each of the wines; doing this helps the candidates map the descriptors to a particular variety.

After watching the documentary Somm on Netflix, which documents the struggles of four candidates attempting to pass the Sommelier exam, I became intrigued by this subject matter. I wondered if I could use machine learning methods to train a model that would identify wines through blind tasting like a master sommelier would. Since at least the computers that I own cannot taste the wines themselves, I figured that I could use the text from professional wine reviews as a feature from which the machine would use to predict the grape variety. Wine reviews typically include many sensory descriptors to describe a particular wine variety, so in a way they imitate the recitation that sommelier candidates do to prepare for the exam.

Fortunately, I discovered a Kaggle dataset of more than 150,000 wine reviews, scraped by the Kaggle user Zack Thoutt from the WineEnthusiast website. The dataset is a CSV file with 10 columns and 150k rows of wine reviews. Along with the wine review itself, the dataset includes the country of origin, designation, points (as scored by the reviewer), price, province, region(s), and winery for each wine variety. There are 623 unique wine varieties from 49 different countries.


