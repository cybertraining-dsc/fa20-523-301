# NBA Performance and Injury

[![Check Report](https://github.com/cybertraining-dsc/fa20-523-301/workflows/Check%20Report/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-301/actions)


* Gavin Hemmerlein, fa20-523-301 
* Chelsea Gorius, fa20-523-344
* [Edit](https://github.com/cybertraining-dsc/fa20-523-301/blob/master/project/project.md)

{{% pageinfo %}}

## Abstract

Abstract to be added when content is finalized.

Contents

{{< table_of_contents >}}

{{% /pageinfo %}}

**Keywords:** basketball, NBA, injury, performance, salary, rehabilitation, artificial intelligence, convolutional neural network, Logistic Regressor, LightGBM.

## 1. Introduction

The topic to be investigated is basketball player performance as it relates to injury. The topic of injury and recovery is a multi-billion dollar industry.  The Sports Medicine field is expected to reach $7.2 billion dollars by 2025 [^1].  The scope of this effort is to explore National Basketball Association(NBA) teams, but the additional uses of a topic such as this could expand into other realms such as the National Football League, Major League Baseball, the Olympic Committees, and many other avenues.  For leagues with salaries, projecting an expected return on the investment can assist in contract negotiations and cater expectations.  Competing at such a high level of intensity puts these players at a greater risk to injury than the average athlete because of the intense and constant strain on their bodies.  The overall valuation of the NBA in recent years is over 2 billion dollars, meaning each team is spending millions of dollars in the pursuit of a championship every season.  Injuries to players can cost teams not only wins but also significant profits.  Ticket sales alone for a single NBA finals game have reported greater than 10 million dollars in profit for the home team, if a team's star player gets injured just before the playoffs and the team does not succeed, that is a lot of money lost.  These injuries can have an effect no matter the time of year, regular season ticket sales have been known to fluctuate with injuries from the team's top performers.  Besides ticket sales these injuries can also influence viewership, TV or streaming, and potentially lead to a greater loss in profits.  With the health of the players and so much money at stake NBA team organizations as a whole do their best to take care of their players and keep them injury free.

## 2. Background Research and Previous Work

The assumptions were made based on current literature as well. The injury return and limitations upon return of Anterior Cruciate Ligament (ACL) rupture (ACLR) are well documented and known. Interesting enough, forty percent of the players in the study occurred during the fourth quarter [^2]. This leads some credence to the idea that fatigue is a major factor in the occurrence of these injuries.

The current literature also shows that a second or third injury can occur more frequently due to minor injuries. "When an athlete is recovering from an injury or surgery, tissue is already compromised and thus requires far more attention despite the recovery of joint motion and strength. Moreover, injuries and surgical procedures can create detraining issues that increase the likelihood of further injury" [^3].

## 3. Dataset 

To compare performance and injury, a minimum of two datasets will be needed. The first is a dataset of injuries for players [^4]. This dataset will create the samples necessary for review.

Once the controls for injuries are established, the next requirement will be to establish  pre-injury performance parameters and post-injury parameters.  These areas will be where the feature engineering will take place.  The datasets needed must dive into appropriate basketball performance stats to establish a metric to encompass a player’s performance. One example that ESPN has tried in the past is the Player Efficiency Rating (PER).  To accomplish this, it will be important to review player performance within games such as in the “NBA games data” [^5] dataset because of how it will allow us to see how the player was performing throughout the season, and not just their average stats across the year..  In addition to that the data from the “NBA games data” [^6] dataset will be valuable in order to compare the calculated performance metrics just before an injury or after recovery to their overall performance that season or in seasons prior.  That comparison will provide a solid baseline to understand how injuries can effect a player’s performance. With in depth information about each game of the season, and not just the teams and players aggregated stats, added to the data provided from the injury dataset [^4] we will be able to compose new metrics to understand how these injuries are actually affecting the players performance.  

Along the way we look forward to discovering if there is also a causal relationship to the severity of some of the injuries, based on how the player was performing just before the injury.  The term “load management” has become popular in recent years to describe players taking rest periodically throughout the season in order to prevent injury from overplaying.  This new practice has received both support for the player safety it provides and also criticism around players taking too much time off.  Of course not all injuries are entirely based on the recent strain under the players body, but a better understanding about how that affects the injury as a whole could give better insight into avoiding more injuries.  It is important to remember though that any pattern identification would not lead to an elimination of all injuries, any contact sport will continue to have injuries, especially one as high impact as the NBA.  There is value to learn from why some players are able to return from certain injuries more quickly and why some return to almost equivalent or better playing performance than before the injury.  This comparison of performance will be made by deriving metrics based on varying ranges of games immediately leading up to injury and then immediately after returning from injury.  In addition to that we will perform comparisons to the players known peak performance to better understand how the injury affected them.  Another factor it will be important to include is the length of time recovering from the injury. Different players take differing amounts of time off, sometimes even with similar injuries.  Something will be said about the player’s dedication to recovery and determination to remain at peak performance, even through injury, when looking at how severe their injury was, how much time was taken for recovery, and how they performed upon returning.

These datasets were chosen because they allow for a review of individual game performance, for each team, throughout each season in the recent decade.  Aggregate statistics such as points per game (ppg) can be deceptive because duration of the metric is such a large period of time.  The large sample of 82 games can lead to a perception issue when reviewing the data.  These datasets include more variables to help us determine effects to player injury, such as minutes per game (mpg) to understand how strenuous the pre-injury performance or how fatigue may have played a factor in the injury.  Understanding more of the variables such as fouls given or drawn can help determine if the player or other team seemed to be the primary aggressor before any injury.  

### 3.1 Data Transformations and Calculations

Using the Kaggle package the datasets are downloaded direct from the website and unzipped to a directory accessible by the ‘project_dateEngineering.ipynb’ notebook.  The 7 unzipped datasets are then loaded into the notebook as pandas data frames using the ‘.read_csv()’ function.  The data engineering performed in the notebook includes removal of excess data and data type transformations across almost all the data frames loaded. This data transformation includes transforming the games details column ‘MIN’, meaning minutes played, from a timestamp format to a numerical format that could have calculations like summation or average performed on it. This was a crucial transformation since minutes played have a direct correlation to player fatigue, which can increase a player’s chance of injury.

One of the more difficult tasks was transforming the Injury dataset into something that would provide more information through machine learning and analysis.  The dataset is loaded as one data set where 2 columns ‘Relinquished’ and ‘Acquired’ defined if the row in questions was a player leaving the roster due to injury or returning from injury, respectively.   In this case for each for one of those two columns contained a players name and the other was blank. Besides that the data frame contained information like the date, notes, and the team name.  In order to appropriately understand each injury as whole the data frame needs to be transformed into one where each row contains the player, the start date of the injury, and the end date of the injury.  In order to do this first the original Injury dataset was separated into rows marking the start of an injury and those marking the end of an injury. Data frames from the *NBA games data* [^5] data set were used to join TeamID and PlayerID columns to the Injury datasets. An ‘iterrows():’ loop was then used on the data frame marking the start of an injury to specifically locate the corresponding row in the Injury End data frame with the same PlayerID and where the return date was the closest date after the injury date.  As this new data frame was being transformed, it was noted that sometimes a Player would have multiple rows with the same Injury ending date but different injury start dates, this can happen if an injury worsens or the player did not play due to last minute decision. In order to solve this the table was grouped by the PlayerID and InjuryEnd Date while keeping the oldest Injury Start date, since the model will want to see the full length of the injury.  From there it was simple to calculate the difference in days for each row between the Injury start and end dates. This data frame is called ‘df_Injury_length’ in the notebook and is much easier to use for improved understanding of NBA injuries than the original format of the Injury data set.

Once created, the ‘df_Injury_length’ data frame was copied and built upon.  Using ‘iterrows():’ loop again to filter down the games details data frame rows with the same PlayerId, over 60 calculated columns are created to produce the ‘df_Injury_stats’ data frame.  The data frame includes performance statistics specifically from the game the player was injured and the game the player returned from that injury.  In addition to this aggregate performance metrics were calculated based on the 5 games prior to the injury and the 5 games post returning from injury.  At this time the season of when the injury occurred and when the player returned is also stored in the dataframe. This will allow comparisons between the ‘df_Injury_stats’ data frame and the ‘df_Season_stats’ data frame which contains the players average performance metrics for entire seasons. 


![Average Minutes Played in First Five Games Upon Return over Injury Length in Days](https://github.com/cybertraining-dsc/fa20-523-301/raw/master/project/images/avg_Minutes_Played_in_Post_5_per_injury_length.PNG)

**Figure 1:** Average Minutes Played in First Five Games Upon Return over Injury Length in Days*

.

![Frequency of Injuries by Average Minutes Played in Prior Five Games](https://github.com/cybertraining-dsc/fa20-523-301/raw/master/project/images/frequencies_by_average_minutes.png)

**Figure 2:** Frequency of Injuries by Average Minutes Played in Prior Five Games*

.

![Injury Length in Days over Number of Injuries](https://github.com/cybertraining-dsc/fa20-523-301/raw/master/project/images/injury_length.png)

**Figure 3:** Injury Length in Days over Number of Injuries

.

![Injury Length in Days over Avg Minutes Plaed in Prior 5 Games](https://github.com/cybertraining-dsc/fa20-523-301/raw/master/project/images/injury_length_over_avg_min.png)

**Figure 4:** Injury Length in Days over Avg Minutes Plaed in Prior 5 Games

## 4. Methodology

The objective of this project is to develop performance indicators for injured players returning to basketball in the NBA.  It is unreasonable to expect a player to return to the same level of play post injury immediately upon starting back up after recovery.  It often takes a player months if not years to return to the same level of play as pre-injury, especially considering the severity of the injuries.  In order to successfully analyze this information from the datasets, a predictive model will need to be created using a large set of the data to train. 

From this point, a test run will be used to gauge the validity and accuracy of the model compared to some of the data set aside.  The model created will be able to provide feature importance to give a better understanding of which specific features are the most crucial when it comes to determining how bad the effects of an injury may or may not be on player performance.  Feature engineering will be performed prior to training the model in order to improve the chances of higher accuracy from the predictions.  This model could be used to keep an eye out for how a player's performance intensity and the engineered features could affect how long a player takes to recover from injury, if there are any warning signs prior to an injury, and even how well they perform when returning.

### 4.1 Development of Models

To help with review of the data, conditioned data was used to save resources on Google Colab. By conditioning the data and saving the files as a .CSV, the team was able to create a streamlined process. Additionally, the team found benefit by uploading these files to Google Drive to quickly import data near real time. The files saved were the following:

| Dataframe     | Title |
| :---        |    :----:  |
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

#### 4.1.1 Logistics Regression

The initial model that was used was a Logistic Regression model. This model produced results of *X*. These results

![Logistic Regressor](https://github.com/cybertraining-dsc/fa20-523-301/raw/master/project/images/logistic-regression-example.jpg)

**Figure 1:** Logistic Regressor [^LogReg]

After running a Logistic Regression model, the decision was made to try multiple models to see what gives the best results. The team decided to use a Linear Regression model.

#### 4.1.2 LightGBM Regression

Another algorithm chosen was a Light Gradient Boost Machine (LightGBM) model. LightGBM is known for its lightweight and resource sparse abilities. The model is built from decision tree algorithms and used for ranking, classification, and other machine learning tasks. By choosing LightGBM data scientists are able to analyze larger data a faster approach. LightGBM  can often over fit a model if the data is too small, but fortunately for the purpose of this assignment the data available for NBA injuries and stats is extremely large. Availability of data allowed for smooth operation of the LightGBM model. Mandot explains the model really well in The Medium. Mandot said, "Light GBM can handle the large size of data and takes lower memory to run. Another reason of why Light GBM is popular is because it focuses on accuracy of results. LGBM also supports GPU learning and thus data scientists are widely using LGBM for data science application development" [^a]. There are a lot of benefits available to this algorithm.

![LightGBM Algorithm: Leafwise searching](https://github.com/cybertraining-dsc/fa20-523-301/raw/master/project/images/lightGBM_regressor.png)

**Figure 2:** LightGBM Algorithm: Leafwise searching [^LGBMReg]

When running the model, we saw promising results.

#### 4.1.3 Keras Deep Learning Models

The final model attempted was a Deep Learning model. A few runs of different layers and epochs were chosen. The models sequentially ran through the test layers to refine the model. When this is done, each predecessor layer acts as an input to the next layer's model. The results can produce accurate results while using unsupervised learning. The visualization for this model can be seen in the following figure:

![Neural Network](https://github.com/cybertraining-dsc/fa20-523-301/raw/master/project/images/simple_neural_network_vs_deep_learning.jpg)

**Figure 3:** Neural Network [^NeuNet]

When the team ran the Neural Networks, the data went through three layers. Each layer was built upon the previous similarly to the figure. This allowed for the team to capture information from the processing.

## 5. Inference

With the data available, some conclusions can be made. Not all injuries are of the same severity. By treating an ACL tear in the same manner as a bruise, the team doctors would take terrible approaches to rehab. The severity of the injury is a part of the approach to therapy.

Another aspect to come to a conclusion is that not every player recovers in the same timetable as another. Genetics, diet, and mental health can all harm or reinforce the efforts from the medical staff. These areas are hard to capture in the data and cannot be appropriately reviewed. 

It is also difficult to indicate where a previous injury may have contributed to a current injury. The kinetic chain is a structure of the musculoskeletal system that moves the body using the muscles and bones. If one portion of the chain is compromised, the entire chain will need to be modified to continue movement. This modification can result in more injuries. The data cannot provide this information.

## 6. Conclusion

This section will be addressed upon project completion.
 
 After reviewing the results, performance does indeed appear to degrade over time. 
 
 These results are consistent with the current scientific literature [^2] [^3].

Initial Project Report - Gorius, Hemmerlein
Predictive Model - breakdown undetermined at this time

## 7. Acknowledgements

The author would like to thank Dr. Gregor Von Laszewski, Dr. Geoffrey Fox, and the associate instructors in the *FA20-BL-ENGR-E534-11530: Big Data Applications* course (offered in the Fall 2020 semester at Indiana University, Bloomington) for their continued assistance and suggestions with regard to exploring this idea and also for their aid with preparing the various drafts of this article.

### 7.1 Work Breakdown

For the effort developed, the team split tasks between each other to cover more ground. The requirements for the investigation required a more extensive effort for the teams in the ENGR-E 534 class. To accomplish the requirements, the task was expanded by addressing multiple datasets within the semester and building in multiple models to display the results. The team members were responsible for committing in Github multiple times throughout the semester. The tasks were divided as follows:

1. Chelsea Gorius
   * Exploratory Data Analysis
   * Feature Engineering
2. Gavin Hemmerlein
   * Organization of items
   * Model Development
3. Both
   * All outstanding items

## 8. References

[^1]: A. Mehra, *Sports Medicine Market worth $7.2 billion by 2025*, Markets and Markets.
 <https://www.marketsandmarkets.com/PressReleases/sports-medicine-devices.asp>

[^2]: J. Harris, B. Erickson, B. Bach Jr, G. Abrams, G. Cvetanovich, B. Forsythe, F. McCormick, A. Gupta, B. Cole,
*Return-to-Sport and Performance After Anterior Cruciate Ligament Reconstruction in National Basketball Association Players*, Sports Health. 2013 Nov;5(6):562-8. doi: 10.1177/1941738113495788. [Online serial]. Available: <https://pubmed.ncbi.nlm.nih.gov/24427434>  [Accessed Oct. 24, 2020].

[^3]: W. Kraemer, C. Denegar, and S. Flanagan, *Recovery From Injury in Sport: Considerations in the Transition From Medical Care to Performance Care*, Sports Health. 
2009 Sep; 1(5): 392–395.[Online serial]. Available: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3445177>   [Accessed Oct. 24, 2020].

[^4]: R. Hopkins, *NBA Injuries from 2010-2020*, Kaggle. <https://www.kaggle.com/ghopkins/nba-injuries-2010-2018>

[^5]: N. Lauga, *NBA games data*, Kaggle.  <https://www.kaggle.com/nathanlauga/nba-games?select=games_details.csv>

[^6]: J. Cirtautas, *NBA Players*, Kaggle. <https://www.kaggle.com/justinas/nba-players-data>

[^a]: P. Mandon, *What is LightGBM, How to implement it? How to fine tune the parameters?*, Medium. <https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc>

[^LogReg]: TEXT MISSING <https://helloacm.com/a-short-introduction-logistic-regression-algorithm>

[^LGBMReg]: TEXT MISSING <https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc>

[^NeuNet]:  TEXT MISSING <https://thedatascientist.com/what-deep-learning-is-and-isnt>





