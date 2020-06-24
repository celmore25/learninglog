# Clay's Personal Learning Log

For a few days I have been thinking about how I should go about my personal learning. While listening to a podcast from TDS, I heard about making an effort every day to "get in the gym" even if you don't work out as hard as possible. So, my goal will now be to get into the data science gym everyday starting today. This can be as much as working on a project, or as simple as reading a DS blog post. Please see my findings below!

## Continuous Learning Table with Links

| Date | Title | Content Type | Link | Personal Rating |
|---------|-------|------|------|------|
| 6/20/2020 | Deploying a Streamlit App on Heroku | Tutorial | [link](https://gilberttanner.com/blog/deploying-your-streamlit-dashboard-with-heroku) | 8 |
| 6/21/2020 | Data Science Infrastructure and MLops | Podcast | [link](https://towardsdatascience.com/data-science-infrastructure-and-mlops-ba0da1c4d8b) | 6 |
| 6/22/2020 | Machine Learning Engineering: Introduction, Before the Project Starts | Book | [link](http://www.mlebook.com/wiki/doku.php ) | 7 |
| 6/22/2020 | Machine Learning Engineering: Data Prep, Feature Engineering | Book | [link](http://www.mlebook.com/wiki/doku.php ) | 9 |


## Continuous Learning Notes


* **Deploying a Streamlit App with Heroku Tutorial**
  * Date: 6/20/2020
  * Summary: This tutorial walked through how to deploy a streamlit app on a heroku server
  * Key Learnings:
    * How to build a streamlit app
    * How to deploy to heroku
    * Using github to sync with a Heroku Application
    * Final output (for father's day)
      * [my web app](https://enigmatic-springs-10364.herokuapp.com/)
* **Data Science Infrastructure Podcast**
  * Date: 6/21/2020
  * Summary: An interview with the founder of dotscience who goes over the challenges of ML Dev Ops and how it is fundamentally different that just software engineering.
  * Key Learnings:
    * Learning to deploy models in a sustainable manner (Docker, Kubes, CICD) is going to be a huge plus for any data scientist
    * Monitoring models should be done statistically, not via route logic
    * Think about "runs" instead of "code" when it comes to git for ML. This will incorporate data versioning as well as code versioning
* **Machine Learning Engineering: Chapter 1: Introduction**
  * Date: 6/22/2020
  * Summary: This is an introductory chapter to a new machine learning engineering book which covers some mathematics definitions, basic ML terminology, when to (and not to) apply ML, and basics of ML Engineering.
  * Key Learnings:
    * Tuning an *entire pipeline* is a critical machine learning task as opposed to the general belief that just hyperparameters need to be tuned
    * Contrary to popular belief, the best time to apply ML to a problem is has a very simple objective (dog or not dog, 0 - 1, buy or sell, ...)
    * Make sure to distinguish between the business goal and the machine learning goal. They are rarely the same.
* **Machine Learning Engineering: Chapter 2: Before the Project Starts**
  * Date: 6/22/2020
  * Summary: This chapter goes through challenges in prioritizing, sizing difficulty, and building a team when it comes to machine learning projects all through the eyes of a business decision.
  * Key Learnings:
    * There are two main things to consider when prioritizing an ML project
      * Impact: how is the ML implementation going to easy a currently problem
      * Cost: things that impact cost are difficulty, data, and accuracy
    * Assessing the difficulty of an ML project is very difficult. There is no substitute for experience here which is why logging projects can be so helpful in future planning. However, POC, smaller problems are a good place to start.
    * The two basic skills needed in a successful ML team are 1) ML foundations and 2) software engineering skills.
* **Machine Learning Engineering: Chapter 3: Data Collection and Prep**
  * Date: 6/23/2020
  * Summary: This chapter goes over the process of curating raw data for a ML project. Topics covered include determining data sizing, data quality, bias, leakage, missing data, data augmentation, sampling, and best practices for data versioning and documentation.
  * Key Learnings:
    * Plotting learning curves is not just a good method for model exploration but also a good method for determining data sizing needs.
    * Context of where/when/how data is obtained can be just as valuable as the raw data in a dataset.
    * Bias in datasets can be both engrained in the raw data and a function of the data scientist who is gathering the data. Both are detrimental to algorithm performance.
    * One of the best ways to avoid bias is to "peer review" data collection similar to code reviews.
    * Data leakage that is sometimes hard to find is when future data is buried in what is considered past data. The best way to avoid this is by having a good understanding of the data being input to a model.
    * An advanced data imputation technique is to build a regression/classification problem to solve for the missing data then input that data into an ML model.
    * Image and text augmentation can greatly improve generalization for deep learning frameworks in these fields.
      * Common techniques include: changing an image (flip, contrast, loss, noise, etc..), image addition, text hypernyms, text back translation, and BERT text changing
    * Sampling data correctly (especially in skewed distributions) can be the key to breaking through performance barriers without aquiring new data.
    * Great quote: "data first, algorithm second"
    * I should pay more attention to reproducability
* **Machine Learning Engineering: Chapter 4: Feature Engineering**
  * Date: 6/23/2020
  * Summary: 
  * Key Learnings:
    * Mean encoding is a great alternative to OHE which is shown to not always be the best option for creating a numerical feature
    * Sin-cos tranformation should be used for cyclical features (days of week, season, etc.)
    * Speed of feature calculation is important. You can't be waiting on slow API's in production even if predictive power is high (@wikipedia)
    * Need to check out Boruta in the future for feature selection
    * When data clearly fits into buckets, clustering features to create new ones can be a good idea
    * Standardize with outlier, bell curves, and clustering; normalize for most other cases
    * Consider feature removal only is absolutely necessay


## Current Favorites

## Future Learnings

### Modeling / Algorithms
* https://podcasts.apple.com/us/podcast/machine-learning-guide/id1204521130
* http://themlbook.com/wiki/doku.php

### ML Engineering
* http://www.mlebook.com/wiki/doku.php 

### Infrastructure