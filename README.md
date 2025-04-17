# Project Title: Predicting engagement for Tik Tok posts
This project focuses on analyzing TikTok post data to understand engagement patterns and optimize content strategies. The analysis includes exploratory data analysis (EDA), data cleaning, and machine learning modeling to predict post performance.

## Problem Statement: 
The goal of this project is to analyze the various factors that influence TikTok engagement, specifically measured by the number of likes that a video receives. Likes serve as a passive indicator of user interest and provide valuable insights into audience behavior. By understanding how posting factors such as profile followers, posting time, and video duration can affect engagement. This analysis aims to uncover actionable strategies to optimize content reach and better connect with the target audience.

## Project Structure
TikTok - Posts.csv # Raw TikTok post data
main.ipynb # Notebook for EDA, data cleaning, machine learning, and insights

## Dataset Description: 
This dataset contains information about TikTok posts, including various engagement metrics, post metadata, and account information. It is used for analyzing user engagement and optimizing content strategies on the TikTok platform.

## Methodology:
EDA:
Data cleaning:
Machine Learning: Several machine learning models were trained to predict like count:
Linear Regression
Decision Tree
Random Forest
To further improve performance, we used log transformation on the like count before training the final model. This helped reduce the impact of outliers and allowed the models to better learn the underlying patterns, since our like count was very right-skewed.
We then used MSE and R-square to evalute the performance of each model, for both the train and test data.

## Conclusion: 
In general, other than the content of the video, specific details about how the TikTok is posted do affect the engagement.
If an account is verified or has more followers, it tends to get higher engagement.
Posting on certain days, especially Sundays, also shows better performance.
There's a strong correlation between like count and play count, meaning the more views a video gets, the more likely it is to receive likes.
Shorter videos usually lead to more engagement compared to longer ones.

## Recommendations:
1. If the company account is not verified or has low follower count, it's recommended to collaborate with influencers who are verified and have a big following to further boost engagement.
2. Plan the timing of posts—posting on Sundays tends to bring better results.
3. In general, keep videos short and engaging. Instead of making long videos, focus on shorter ones, which viewers are more likely to watch. This will help increase both views and likes.

## Individual Contributions: 
Ruiyang was in charge of the machine learning components and implemented the interactive prediction feature on the webpage.
Lily carried out exploratory data analysis (EDA) and took the lead in cleaning and preparing the dataset for modeling.
Jingping defined the problem statement, structured the overall project direction, and created the final presentation slides.
