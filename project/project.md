---
date: 2021-03-15
title: NBA Performance and Injury
linkTitle: NBA
tags: ["project", "ai", "sports", "health"]
description: NBA Performance and Injury
author: Gavin Hemmerlein
resources:
- src: "**.{png,jpg}"
  title: "Image #:counter"
---

[![Check Report](https://github.com/cybertraining-dsc/fa20-523-301/workflows/Check%20Report/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-301/actions)
[![Status](https://github.com/cybertraining-dsc/fa20-523-301/workflows/Status/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-301/actions)
Status: final, Type: Project

* Gavin Hemmerlein, fa20-523-301 
* Chelsea Gorius, fa20-523-344
* [Edit](https://github.com/cybertraining-dsc/fa20-523-301/blob/main/project/project.md)

{{% pageinfo %}}

## Abstract

Sports Medicine will be a $7.2 billion dollar industry by 2025. The NBA has a vested interest in predicting performance of players as they return from injury. The authors evaluated datasets available to the public within the 2010 decade to build machine and deep learning models to expect results. The team utilized Gradient Based Regressor, Light GBM, and Keras Deep Learning models. The results showed that the coefficient of determination for the deep learning model was approximately 98.5%. The team recommends future work to predicting individual player performance utilizing the Keras model.

Contents

{{< table_of_contents >}}

{{% /pageinfo %}}

**Keywords:** basketball, NBA, injury, performance, salary, rehabilitation, artificial intelligence, convolutional neural network, lightGBM, deep learning, gradient based regressor.

## 1. Introduction

The topic to be investigated is basketball player performance as it relates to injury. The topic of injury and recovery is a multi-billion dollar industry.  The Sports Medicine field is expected to reach $7.2 billion dollars by 2025 [^1].  The scope of this effort is to explore National Basketball Association(NBA) teams, but the additional uses of a topic such as this could expand into other realms such as the National Football League, Major League Baseball, the Olympic Committees, and many other avenues.  For leagues with salaries, projecting an expected return on the investment can assist in contract negotiations and cater expectations.  Competing at such a high level of intensity puts these players at a greater risk to injury than the average athlete because of the intense and constant strain on their bodies.  The overall valuation of the NBA in recent years is over 2 billion dollars, meaning each team is spending millions of dollars in the pursuit of a championship every season.  Injuries to players can cost teams not only wins but also significant profits.  Ticket sales alone for a single NBA finals game have reported greater than 10 million dollars in profit for the home team, if a team's star player gets injured just before the playoffs and the team does not succeed, that is a lot of money lost.  These injuries can have an effect no matter the time of year, regular season ticket sales have been known to fluctuate with injuries from the team's top performers.  Besides ticket sales these injuries can also influence viewership, TV or streaming, and potentially lead to a greater loss in profits.  With the health of the players and so much money at stake NBA team organizations as a whole do their best to take care of their players and keep them injury free.

## 2. Background Research and Previous Work

The assumptions were made based on current literature as well. The injury return and limitations upon return of Anterior Cruciate Ligament (ACL) rupture (ACLR) are well documented and known. Interesting enough, forty percent of the players in the study occurred during the fourth quarter [^2]. This leads some credence to the idea that fatigue is a major factor in the occurrence of these injuries.

The current literature also shows that a second or third injury can occur more frequently due to minor injuries. *"When an athlete is recovering from an injury or surgery, tissue is already compromised and thus requires far more attention despite the recovery of joint motion and strength. Moreover, injuries and surgical procedures can create detraining issues that increase the likelihood of further injury"* [^3].

## 3. Dataset 

To compare performance and injury, a minimum of two datasets will be needed. The first is a dataset of injuries for players [^4]. This dataset created the samples necessary for review.

Once the controls for injuries were established, the next requirement was to establish pre-injury performance parameters and post-injury parameters. These areas were where the feature engineering took place. The datasets needed had to include appropriate basketball performance stats to establish a metric to encompass a player's performance. One example that ESPN has tried in the past is the Player Efficiency Rating (PER). To accomplish this, it was important to review player performance within games such as in the *NBA games data* [^5] dataset because of how it allowed the team to evaluate the player performance throughout the season, and not just the average stats across the year. In addition to that the data from the *NBA games data* [^6] dataset was valuable in order to compare the calculated performance metrics just before an injury or after recovery to the player's overall performance that season or in seasons prior. That comparison provided a solid baseline to understand how injuries can effect a player's performance. With in depth information about each game of the season, and not just the teams and players aggregated stats, added to the data provided from the injury dataset [^4] the team was be able to compose new metrics to understand how these injuries are actually affecting the players performance.

Along the way attempted to discover if there is also a causal relationship to the severity of some of the injuries, based on how the player was performing just before the injury. The term *load management* has become popular in recent years to describe players taking rest periodically throughout the season in order to prevent injury from overplaying. This new practice has received both support for the player safety it provides and also criticism around players taking too much time off. Of course not all injuries are entirely based on the recent strain under the players body, but a better understanding about how that affects the injury as a whole could give better insight into avoiding more injuries. It is important to remember though that any pattern identification would not lead to an elimination of all injuries, any contact sport will continue to have injuries, especially one as high impact as the NBA. There is value to learn from why some players are able to return from certain injuries more quickly and why some return to almost equivalent or better playing performance than before the injury. This comparison of performance was attempted by deriving metrics based on varying ranges of games immediately leading up to injury and then immediately after returning from injury. In addition to that performed comparisons to the players known peak performance to better understand how the injury affected them. Another factor that was important to include is the length of time recovering from the injury. Different players take differing amounts of time off, sometimes even with similar injuries. Something will be said about the player’s dedication to recovery and determination to remain at peak performance, even through injury, when looking at how severe their injury was, how much time was taken for recovery, and how they performed upon returning.

These datasets were chosen because they allow for a review of individual game performance, for each team, throughout each season in the recent decade. Aggregate statistics such as points per game (ppg) can be deceptive because duration of the metric is such a large period of time. The large sample of 82 games can lead to a perception issue when reviewing the data. These datasets include more variables to help the team determine effects to player injury, such as minutes per game (mpg) to understand how strenuous the pre-injury performance or how fatigue may have played a factor in the injury. Understanding more of the variables such as fouls given or drawn can help determine if the player or other team seemed to be the primary aggressor before any injury. 

### 3.1 Data Transformations and Calculations

Using the Kaggle package the datasets were downloaded direct from the website and unzipped to a directory accessible by the ‘project_dateEngineering.ipynb’ notebook. The 7 unzipped datasets are then loaded into the notebook as pandas data frames using the ‘.read_csv()’ function. The data engineering performed in the notebook includes removal of excess data and data type transformations across almost all the data frames loaded. This data transformation includes transforming the games details column ‘MIN’, meaning minutes played, from a timestamp format to a numerical format that could have calculations like summation or average performed on it. This was a crucial transformation since minutes played have a direct correlation to player fatigue, which can increase a player’s chance of injury.

One of the more difficult tasks was transforming the Injury dataset into something that would provide more information through machine learning and analysis. The dataset is loaded as one data set where 2 columns ‘Relinquished’ and ‘Acquired’ defined if the row in questions was a player leaving the roster due to injury or returning from injury, respectively.  In this case for each for one of those two columns contained a players name and the other was blank. Besides that the data frame contained information like the date, notes, and the team name. In order to appropriately understand each injury as whole the data frame needs to be transformed into one where each row contains the player, the start date of the injury, and the end date of the injury. In order to do this first the original Injury dataset was separated into rows marking the start of an injury and those marking the end of an injury. Data frames from the *NBA games data* [^5] data set were used to join TeamID and PlayerID columns to the Injury datasets. An ‘iterrows():’ loop was then used on the data frame marking the start of an injury to specifically locate the corresponding row in the Injury End data frame with the same PlayerID and where the return date was the closest date after the injury date. As this new data frame was being transformed, it was noted that sometimes a Player would have multiple rows with the same Injury ending date but different injury start dates, this can happen if an injury worsens or the player did not play due to last minute decision. In order to solve this the table was grouped by the PlayerID and InjuryEnd Date while keeping the oldest Injury Start date, since the model will want to see the full length of the injury. From there it was simple to calculate the difference in days for each row between the Injury start and end dates. This data frame is called ‘df_Injury_length’ in the notebook and is much easier to use for improved understanding of NBA injuries than the original format of the Injury data set.

Once created, the ‘df_Injury_length’ data frame was copied and built upon. Using ‘iterrows():’ loop again to filter down the games details data frame rows with the same PlayerId, over 60 calculated columns are created to produce the ‘df_Injury_stats’ data frame. The data frame includes performance statistics specifically from the game the player was injured and the game the player returned from that injury. In addition to this aggregate performance metrics were calculated based on the 5 games prior to the injury and the 5 games post returning from injury. At this time the season of when the injury occurred and when the player returned is also stored in the dataframe. This will allow comparisons between the ‘df_Injury_stats’ data frame and the ‘df_Season_stats’ data frame which contains the players average performance metrics for entire seasons. 

A few interesting figures were generated within the Exploratory Data Analysis (EDA) stage. **Figure 1** gave a view of the load of the player returning from injury. The load to the player will show how recovered the player is upon completion of rehab. Many teams decide to slowly work a returning player in. Additionally, the amount of time for an injury can be seen on this graph. The longer the injury, the more unlikely the player will return to action.

![Average Minutes Played in First Five Games Upon Return over Injury Length in Days](https://github.com/cybertraining-dsc/fa20-523-301/raw/main/project/images/avg_min_played_post5.png)

**Figure 1:** Average Minutes Played in First Five Games Upon Return over Injury Length in Days*

**Figure 2** shows the frequency in which a player is injured. The idea behind this graph is to see a relationship between the time leading up to the injury. Interesting enough, there is no key indication of where injury is more likely to occur. It can be assumed that there is a rarity of players who see playing time greater than 30 minutes. The histogram only shows a near flat relationship; which was surprising.

![Frequency of Injuries by Average Minutes Played in Prior Five Games](https://github.com/cybertraining-dsc/fa20-523-301/raw/main/project/images/frequencies_by_average_minutes.png)

**Figure 2:** Frequency of Injuries by Average Minutes Played in Prior Five Games*

**Figure 3** shows the length of injury over number of injuries. By reviewing this data, it can be seen that most injuries occur fewer rather than more often. A player that is deemed injury prone will be a lot more likely to be cut from the team. This data makes sense.

![Injury Length in Days over Number of Injuries](https://github.com/cybertraining-dsc/fa20-523-301/raw/main/project/images/injury_length.png)

**Figure 3:** Injury Length in Days over Number of Injuries

**Figure 4** shows the injury length over average minutes played in the five games before injury. This graph attempts to show all of the previous games and the impacts to the players injury. The data looks evenly distributed, but the majority of plaers do not play close to 40 minutes per game. By looking at this data, it shows that minutes played does likely contribute to the injury severity.

![Injury Length in Days over Avg Minutes Played in Prior 5 Games](https://github.com/cybertraining-dsc/fa20-523-301/raw/main/project/images/injury_length_over_avg_min.png)

**Figure 4:** Injury Length in Days over Avg Minutes Played in Prior 5 Games

**Figure 5** shows that in general the number of games played does not have a significant relationship to the length of the injury.  There is a darker cluster between 500-1000 days injured that exists over the 40-82 games played, this could suggest that as more games are played there is likeliness for more severe injury.

![Injury Length in Days over Player Games Played that Season](https://github.com/cybertraining-dsc/fa20-523-301/raw/main/project/images/injurylength_gamesplayed.png)

**Figure 5:** Injury Length in Days over Player Games Played that Season

**Figures 6**, **Figure 7**, and **Figure 8** attempt to demonstrate if any relationship exists visually between a player's injury length and their age, weight, or height.  For the most part **Figure 6** shows most severe injuries occurring to younger players, which could make sense considering they can perform more difficult moves or have more stamina than older players.  Some severe injuries still exist among the older players, this also makes sense considering their bodies have been under stress for many years and are more prone to injury. It should be noted that there are more players in the league that fall into the younger age bucket than the older ages. It is difficult to identify any pattern on **Figure 7**.  If anything the graph is somewhat normally shaped similar to the heights of players across the league. Suprisingly the injuries on **Figure 8** are clustered a bit towards the left, being the lighter players.  This could be explained through the fact that the lighter players are often more athletic and perform more strenuous moves than heavier players.  It is also somewhat surprising since the argument that heavier players are putting more strain on their bodies could be used as a reason why heavier players would have worse injuries. One possible explanation could be the musculature adding more of the dense body mass could add protection to weakened joints. More investigation would be needed to identify an exact reason.

![Injury Length in Days over Player Age that Season](https://github.com/cybertraining-dsc/fa20-523-301/raw/main/project/images/injurylength_playerage.png)

**Figure 6:** Injury Length in Days over Player Age that Season

![Injury Length in Days over Player Height in Inches](https://github.com/cybertraining-dsc/fa20-523-301/raw/main/project/images/injurylength_playerHeight.png)

**Figure 7:** Injury Length in Days over Player Height in Inches

![Injury Length in Days over Player Weight in Kilograms](https://github.com/cybertraining-dsc/fa20-523-301/raw/main/project/images/injurylength_playerWeight.png)

**Figure 8:** Injury Length in Days over Player Weight in Kilograms

Finally, the team decided to use the z-score to normalize all of the data. By using the Z-score from the individual data in a column of df_Injury_stats, the team was able to limit variability of multiple metrics across the dataframe. A player's blocks and steals should be a miniscule amount compared to minutes or points of some players. The same can be said of assists, technical fouls, or any other statistic in the course of an NBA game. The Z-score, by nature of the metric from the mean, allows for much less variability across the columns. 

## 4. Methodology

The objective of this project was to develop performance indicators for injured players returning to basketball in the NBA. It is unreasonable to expect a player to return to the same level of play post injury immediately upon starting back up after recovery. It often takes a player months if not years to return to the same level of play as pre-injury, especially considering the severity of the injuries. In order to successfully analyze this information from the datasets, a predictive model will need to be created using a large set of the data to train. 

From this point, a test run was used to gauge the validity and accuracy of the model compared to some of the data set aside. The model created was able to provide feature importance to give a better understanding of which specific features are the most crucial when it comes to determining how bad the effects of an injury may or may not be on player performance. Feature engineering was performed prior to training the model in order to improve the chances of higher accuracy from the predictions. This model could be used to keep an eye out for how a player's performance intensity and the engineered features could affect how long a player takes to recover from injury, if there are any warning signs prior to an injury, and even how well they perform when returning.

### 4.1 Development of Models

To help with review of the data, conditioned data was used to save resources on Google Colab. By conditioning the data and saving the files as a .CSV, the team was able to create a streamlined process. Additionally, the team found benefit by uploading these files to Google Drive to quickly import data near real time. After operating in this fashion for some time, the team was able to load the datasets into Github and utilize that feature. By loading the datasets up to Github, a url could be used to link the files directly to the files saved on Github without using a token like with Kaggle or Google Drive. The files saved were the following:

**Table 1:** Datasets Imported

| **Dataframe**     | **Title** |
| :---  |    :----:  |
| 1.   | df_Injury_stats      |
| 2.   | df_Injury_length       |
| 3.   | df_Season_stats       |
| 4.   | games      |
| 5.   | df_Games_gamesDetails       |
| 6.   | injuries_2010-2018       |
| 7.   | players      |
| 8.   | ranking       |
| 9.   | teams       |

Every time Google Colab loads data, it takes time and resources. The team was able to utilize the cross platform connectivity of the Google utilities. The team could then focus on building models as opposed to conditioning data every time the code was ran.

#### 4.1.1 Evaluation Metrics

The metrics chosen were designed to give results on  Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and the Explained Variance (EV) Score. MAE is a measure of errors between paired observations experiencing the same expression. RMSE is the standard deviation of the prediction errors for our dataset. EV is the relationship between the train data and the test data. By using these metrics, the team is capable of reviewing the data in a statistical manner.

#### 4.1.2 Gradient Boost Regression

The initial model that was used was a Gradient Boosting Regressor (GBR) model. This model produced the results shown in Table 2. The GBR model builds in a stage-wise fashion; similarly to other boosting methods. GBR also generalizes the data and attempts to optimize the results utilizing a loss function. An example of the algorithm can be seen in **Figure 5**.

![Gradient Boosting Regressor](https://github.com/cybertraining-dsc/fa20-523-301/raw/main/project/images/gbr.png)

**Figure 5:** Gradient Boosting Regressor [^GBReg]

 The team saw a relationship given the data. **Table 2** shows the results of that model. The results were promising given the speed and utility of a GBR model. The team reviewed the data multiple times after multiple stages of conditioning the data. 

**Table 2:** GBR Results

| **Category**     | **Value** |
| :---  |    :----:  |
| MAE Mean   | -10.787      |
| MAE STD   | 0.687      |
| RMSE Mean   |    -115.929     |
| RMSE STD  |    96.64     |
| EV Mean   |   1.0      |
| EV STD  |  0.0       |

After running a GBR model, the decision was made to try multiple models to see what gives the best results. The team settled on LightGBM and a Deep Learning model utilizing Keras built on the TensorFlow platform. These results will be seen in *4.1.2* and *4.1.3*.

#### 4.1.2 LightGBM Regression

Another algorithm chosen was a Light Gradient Boost Machine (LightGBM) model. LightGBM is known for its lightweight and resource sparse abilities. The model is built from decision tree algorithms and used for ranking, classification, and other machine learning tasks. By choosing LightGBM data scientists are able to analyze larger data a faster approach. LightGBM can often over fit a model if the data is too small, but fortunately for the purpose of this assignment the data available for NBA injuries and stats is extremely large. Availability of data allowed for smooth operation of the LightGBM model. Mandot explains the model really well in The Medium. Mandot said, *"Light GBM can handle the large size of data and takes lower memory to run. Another reason of why Light GBM is popular is because it focuses on accuracy of results. LGBM also supports GPU learning and thus data scientists are widely using LGBM for data science application development"* [^a]. There are a lot of benefits available to this algorithm.

![LightGBM Algorithm: Leafwise searching](https://github.com/cybertraining-dsc/fa20-523-301/raw/main/project/images/lightGBM_regressor.png)

**Figure 6:** LightGBM Algorithm: Leafwise searching [^a]

When running the model **Table 3** was generated. This table uses the same metrics as the GBR Results Table (**Table 2**). After reviewing the results, the GBR model still appeared to be a viable avenue. The Keras model will be evaluated next to see most optimal model to use for repeatable fresults.

**Table 3:** LightGBM Results

| **Category**     | **Value** |
| :---  |    :----:  |
| MAE Mean   | -0.011      |
| MAE STD   | 0.001      |
| RMSE Mean   |    -0.128     |
| RMSE STD  |    0.046     |
| EV Mean   |   0.982      |
| EV STD  |  0.013       |

#### 4.1.3 Keras Deep Learning Models

The final model attempted was a Deep Learning model. A few runs of different layers and epochs were chosen. They can be seen in **Table 4** (shown later). The model was sequentially ran through the test layers to refine the model. When this is done, each predecessor layer acts as an input to the next layer's input for the model. The results can produce accurate results while using unsupervised learning. The visualization for this model can be seen in the following figure:

![Neural Network](https://github.com/cybertraining-dsc/fa20-523-301/raw/main/project/images/simple_neural_network_vs_deep_learning.jpg)

**Figure 7:** Neural Network [^NeuNet]

When the team ran the Neural Networks, the data went through three layers. Each layer was built upon the previous similarly to the figure. This allowed for the team to capture information from the processing. **Table 4** shows the results for the deep learning model.

**Table 4:** Epochs and Batch Sizes Chosen

| **Number** | **Regressor Epoch**     | **Regressor Batch Sizes** | **KFolds**  | **Model Epochs** |  **R2** |
| :---  |    :----:  |    :----:  |   :----:  |  :----:  |    :----:  |
|*1.*   | *25*   |  *25* | *10* | *10* | *0.985* |
| 2.   | 40   |  25 | 20 | 10 | 0.894 |
| 3.   | 20   |  25 | 20 | 10 | 0.966 |
| 4.   | 20 | 20 | 10 | 10 | 0.707 |
| 5.   | 25   |  25 | 10 | 5 | 0.611 |
| 6.   | 25 | 25 | 10 | 20 | 0.982 |

The team has decided that the results for the Deep Learning are the most desirable. This model would be the one that the team would recommend based on the results from the metrics available. The parameters the team recommends are italicized in *Line 1* of **Table 4**.

## 5. Inference

With the data available, some conclusions can be made. Not all injuries are of the same severity. By treating an ACL tear in the same manner as a bruise, the team doctors would take terrible approaches to rehab. The severity of the injury is a part of the approach to therapy. This detail is nearly impossible to capture in the model.

Another aspect to come to a conclusion is that not every player recovers in the same timetable as another. Genetics, diet, effort, and mental health can all harm or reinforce the efforts from the medical staff. These areas are hard to capture in the data and cannot be appropriately reviewed with this model. 

It is also difficult to indicate where a previous injury may have contributed to a current injury. The kinetic chain is a structure of the musculoskeletal system that moves the body using the muscles and bones. If one portion of the chain is compromised, the entire chain will need to be modified to continue movement. This modification can result in more injuries. The data cannot provide this information.  It is important to remember these possible confounding variables when interpreting the results of the model.

## 6. Conclusion

After reviewing the results, the team created a robust model to predict the performance of a player after an injury. The coefficient of determination for the deep learning model shows a strong relationship between the training and test sets. After conditioning the data, the results can be seen in **Table 2**, **Table 3**, and **Table 5**. The team had an objective to find this correlation and build it to the point where injury and performance can be modeled. The team was able to accomplish this goal.
 
Additionally, these results are consistent with the current scientific literature [^2] [^3]. The biological community has been able to record these results for decades. By leveraging this effort, the scientific community could move to a more proactive approach as opposed to reactive with respect to injury controls. This data will also allow for proper contract negotiations to take place in the NBA, considering potential decisions to avoid injury may include less playing time. The negotiations are pivotal to ensuring that expectations are met in the future seasons; especially when injury occurs in the final year of a player's contract. Teams with an improved understanding of how players can or will return from injury have an opportunity to make the best of scenarios where other teams may be hesitant to sign an injured player.  These different opportunities for a team's front office could be the difference between a championship ring and missing the playoffs entirely.

## 6.1 Limitations

With respect to the current work, the models could be continued to be refined. Currently the results are to the original intentions of the team, but improvements can be made. Feature Engineering is always an area where the models can improve. Some valuable features to be created in the future are the calculations for the player's efficiency overall, as well as offensinve and defensive efficiencies in each game. The team would also like to develop a model to use the stats of a player in pre-injury and apply that to the post-injury set of metrics. Also, the team would like to move to where the same could be applied given the length of the injury to the player while considering the severity of the injury. Longer and more severe injury will lead to different future results than say a long not severe injury, or a short injury that was somewhat severe.  The number of varaibles that could provide more valuable information to the model are endless.

## 7. Acknowledgements

The authors would like to thank Dr. Gregor von Laszewski, Dr. Geoffrey Fox, and the associate instructors in the *FA20-BL-ENGR-E534-11530: Big Data Applications* course (offered in the Fall 2020 semester at Indiana University, Bloomington) for their continued assistance and suggestions with regard to exploring this idea and also for their aid with preparing the various drafts of this article. In addition to that the community of students from the *FA20-BL-ENGR-E534-11530: Big Data Applications* course also deserve a thanks from the author for the support, continued engagement, and valuable discussions through Piazza.

### 7.1 Work Breakdown

For the effort developed, the team split tasks between each other to cover more ground. The requirements for the investigation required a more extensive effort for the teams in the ENGR-E 534 class. To accomplish the requirements, the task was expanded by addressing multiple datasets within the semester and building in multiple models to display the results. The team members were responsible for committing in Github multiple times throughout the semester. The tasks were divided as follows:

1. Chelsea Gorius
   * Exploratory Data Analysis
   * Feature Engineering
   * Keras Deep Learning Model
2. Gavin Hemmerlein
   * Organization of Items
   * Model Development
3. Both
   * Report
   * All Outstanding Items

## 8. References

[^1]: A. Mehra, *Sports Medicine Market worth $7.2 billion by 2025*, [online] Markets and Markets.
 <https://www.marketsandmarkets.com/PressReleases/sports-medicine-devices.asp> [Accessed Oct. 15, 2020].

[^2]: J. Harris, B. Erickson, B. Bach Jr, G. Abrams, G. Cvetanovich, B. Forsythe, F. McCormick, A. Gupta, B. Cole,
*Return-to-Sport and Performance After Anterior Cruciate Ligament Reconstruction in National Basketball Association Players*, Sports Health. 2013 Nov;5(6):562-8. doi: 10.1177/1941738113495788. [Online serial]. Available: <https://pubmed.ncbi.nlm.nih.gov/24427434> [Accessed Oct. 24, 2020].

[^3]: W. Kraemer, C. Denegar, and S. Flanagan, *Recovery From Injury in Sport: Considerations in the Transition From Medical Care to Performance Care*, Sports Health. 
2009 Sep; 1(5): 392–395.[Online serial]. Available: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3445177>  [Accessed Oct. 24, 2020].

[^4]: R. Hopkins, *NBA Injuries from 2010-2020*, [online] Kaggle. <https://www.kaggle.com/ghopkins/nba-injuries-2010-2018> [Accessed Oct. 9, 2020].

[^5]: N. Lauga, *NBA games data*, [online] Kaggle. <https://www.kaggle.com/nathanlauga/nba-games?select=games_details.csv> [Accessed Oct. 9, 2020].

[^6]: J. Cirtautas, *NBA Players*, [online] Kaggle. <https://www.kaggle.com/justinas/nba-players-data> [Accessed Oct. 9, 2020].

[^a]: P. Mandon, *What is LightGBM, How to implement it? How to fine tune the parameters?*, [online] Medium. <https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc> [Accessed Nov., 9 2020].

[^GBReg]: V. Aliyev, *A hands-on explanation of Gradient Boosting Regression*, [online] Medium. <https://medium.com/@vagifaliyev/a-hands-on-explanation-of-gradient-boosting-regression-4cfe7cfdf9e> [Accessed Nov., 9 2020].

[^NeuNet]: The Data Scientist, *What deep learning is and isn’t*, [online] The Data Scientist.  <https://thedatascientist.com/what-deep-learning-is-and-isnt> [Accessed Nov., 9 2020].





