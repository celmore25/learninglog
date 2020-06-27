# Clay's Personal Learning Log

For a few days I have been thinking about how I should go about my personal learning. While listening to a podcast from TDS, I heard about making an effort every day to "get in the gym" even if you don't work out as hard as possible. So, my goal will now be to get into the data science gym everyday starting today. This can be as much as working on a project, or as simple as reading a DS blog post. Please see my findings below!

## Continuous Learning Table with Links

| Date | Title | Content Type | Link | Personal Rating |
|---------|-------|------|------|------|
| 6/20/2020 | Deploying a Streamlit App on Heroku | Tutorial | [link](https://gilberttanner.com/blog/deploying-your-streamlit-dashboard-with-heroku) | 8 |
| 6/21/2020 | Data Science Infrastructure and MLops | Podcast | [link](https://towardsdatascience.com/data-science-infrastructure-and-mlops-ba0da1c4d8b) | 6 |
| 6/22/2020 | Machine Learning Engineering: Introduction, Before the Project Starts | Book | [link](http://www.mlebook.com/wiki/doku.php ) | 7 |
| 6/23/2020 | Machine Learning Engineering: Data Prep, Feature Engineering | Book | [link](http://www.mlebook.com/wiki/doku.php ) | 9 |
| 6/24/2020 | Machine Learning Engineering: Supervised Learning | Book | [link](http://www.mlebook.com/wiki/doku.php ) | 6 |
| 6/25/2020 | Machine Learning Engineering: Model Evaluation, Deployment, and Monitoring | Book | [link](http://www.mlebook.com/wiki/doku.php ) | 4 |
| 6/26/2020 | An Introduction to Graph Theory | Webpage | [link](https://www.tutorialspoint.com/graph_theory/graph_theory_introduction) | 2 |
| 6/27/2020 | Agorithm Intuition and Linear Regression | Podcast | [link](https://www.tutorialspoint.com/graph_theory/graph_theory_introduction) | 6 |


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
  * Summary: This chapter goes over the fundamentals of feature engineering cover what is a good feature, how to deal with different irregularities in raw data, how to expand feature sets, and best computational practice.
  * Key Learnings:
    * Mean encoding is a great alternative to OHE which is shown to not always be the best option for creating a numerical feature
    * Sin-cos tranformation should be used for cyclical features (days of week, season, etc.)
    * Speed of feature calculation is important. You can't be waiting on slow API's in production even if predictive power is high (@wikipedia)
    * Need to check out Boruta in the future for feature selection
    * When data clearly fits into buckets, clustering features to create new ones can be a good idea
    * Standardize with outlier, bell curves, and clustering; normalize for most other cases
    * Consider feature removal only is absolutely necessay
* **Machine Learning Engineering: Chapter 5 (section 1): Supervised Learning**
  * Date: 6/24/2020
  * Summary: This chapter covers how to train a model once you have collected data and performed feature engineering. The chapter stresses how this is probably the least important part of the entire ML pipeline even though it gets a lot of attention. Topics covered are evaluation, metrics, algorithm selection, hypertuning, calibration, and pipelines.
  * Key Learnings:
    * Before the modeling process make sure to pick a metric, baseline, 3 datasets (train, validate, test), and define the necessary performance (usually a business decision)
    * *Page 12 (5.1) has an awesome algorithm selection flowchart from sci-kit*
    * Understanding the production requirements is most important when selecting an algorithm
    * ACPER is a good regression metric when you have a good idea of what (%) error is "good" for a given problem
    * Variance-bias tradeoff does not have a set answer, it has a range of answers which are problem dependent
    * A good research/learning software task would be to create a pipeline that tunes feature engineering hyperparameters as well as model parameters
    * Plot your calibration curves!
* **Machine Learning Engineering: Chapter 5 (section 2): Supervised Learning**
  * Date: 6/24/2020
  * Summary: This chapter goes over deep learning training techniques, imbalanced datasets, and troubleshooting
  * Key Learnings:
    * Dropout is meant to perform auto-regulation
    * Learning rate hyperparameters and nodes/layer are usually the most important values to hypertune in deep learning
    * When combining different types of data feeds (ex: image w text), train 2 networks then have a final embedding layer that feeds a sigmoid, softmax, etc.
    * Softmx is well suited for probability learning
    * Use an adapted metric for imbalance datasets (Cohen statistic)
    * When your model can't handle certain examples, it's okay to remove them from training sets (if they are minimal)
    * Error propogation is hugely important in evaluating chained models
    * Retrain from scratch when time permits
    * I need to write more C++ or move to scala...
* **Machine Learning Engineering: Chapter 6: Model Evaluation**
  * Date: 6/25/2020
  * Summary: This chapter goes over how to evaluate a model once it is in production. Method's reviewed include A/B testing, and MAB. Model robustness and testing sets are also covered.
  * Key Learnings:
    * Online evalution should not just be used for monitoring but should be actively used in incremental model improvements
    * The two basic "denominations" of A/B testing are "yes/no" or "how many". This is analagous to classification vs regression and a different statistical toolkit is needed for each problem.
    * The easiest way to mess up an A/B test is usually not methdology but code bugs
    * UCB1 has convergence properties (who knew...)
    * Neuron testing is a way to determine good testing sets for neural networks. I wonder if there is an anlagous moethod for tree algorithms
* **Machine Learning Engineering: Chapter 7: Model Deployment**
  * Date: 6/25/2020
  * Summary: This chapter goes over how models can be deployed in the real world including static, dynamic, cloud, and hybrid techniques. Model serialization, containers, and versioning are also touched on.
  * Key Learnings:
    * Silent deployment is a good way to potentially test model's as well as deploy them at the same time.
    * Model serialization can be the difference between production and never getting there.
* **Machine Learning Engineering: Chapter 8: Model Serving and Monitoring**
  * Date: 6/25/2020
  * Summary: This chapter covers the various ways that models can be served to either a machine or human, how this can go wrong, and how to monitor algorithms
  * Key Learnings:
    * Small erros in the model can cause huge errors when they are amplified to the population ingesting the algorithm
    * Prediction bias is a great thing to monitor in production and can be paired with a confidence test.
* **An Introduction to Graph Theory**
  * Date: 6/26/2020
  * Summary: A recent problem came up in work of "how to identify skiers in a group", and I thought this would be a good application of graph theory to some of our databases, so I set out to explore more about this emerging field of ML in which I know extremely little. This website provided a basic overview of the concept of a graph and
  * Key Learnings:
    * Degree of a vertix is the number of edges meeting at a vertex
      * "Pendant" means a single connection
      * "Isolated" is a node with no connections
      * For directed graphs there is "indegree" and "outdegree"
    * Eccentricity is the max distance of a single node to all other (connected) nodes in the graph
      * Minimum of this is considered the "Radius"
        * This node is the "Central Point"
      * Maximum of this is considered the "Diameter"
    * Various types of graphs exist and each one has specific properties. These include regular, complete, cycle, and many more
    * There is a ton of jargon in graphs (similar to a lot of other mathematical fields) and I would need to spend more time understanding it if I want to go further in this field
* **Algorithm Intuition and Linear Regression Podcast**
  * Date: 6/27/2020
  * Summary: These two podcast episodes introducted basic concepts in ML such as supervision and reinforcement then went into basics of Linear Regression.
  * Key Learnings:
    * A review of the gradient decent algorithm
    * This topic always makes me wonder about Newton implementation in commercial/standard ML libraries. Is there a reason that so many of them do not use a second derivative based approach?


## Current Favorites
* **Machine Learning Engineering: Chapter 4: Feature Engineering**
  * This section of the machine learning book I recently read was OUTSTANDING. Gave me a ton of ideas about how to handle various situations that arise in feature engineering as well as how to combat the not so favorable situations as well.

## Future Learnings

### Modeling / Algorithms
* https://podcasts.apple.com/us/podcast/machine-learning-guide/id1204521130
* http://themlbook.com/wiki/doku.php
* https://towardsdatascience.com/a-bayesian-approach-to-linear-mixed-models-lmm-in-r-python-b2f1378c3ac8

### ML Engineering

### Infrastructure
