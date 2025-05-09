# Project Title: Predicting Engagement for TikTok Posts
This project focuses on analyzing TikTok post data to understand engagement patterns and optimize content strategies. The analysis includes exploratory data analysis (EDA), data cleaning, and machine learning modeling to predict post performance.
Team members: 
Lily Ulfa Binte Anuar
Du Ruiyang
Lim Jing Ping

## Problem Statement: 
The goal of this project is to analyze the various factors that influence TikTok engagement, specifically measured by the number of likes that a video receives. Likes serve as a passive indicator of user interest and provide valuable insights into audience behavior. By understanding how posting factors such as profile followers, posting time, and video duration can affect engagement. This analysis aims to uncover actionable strategies to optimize content reach and better connect with the target audience.



## Project Files

### Data
- `TikTok - Posts.csv` # Raw TikTok post data
### Analysis and Model Development
- `tiktok_sc1015_project_main.ipynb` # Jupyter notebook
### Web Application
- `server.py` # Flask server
- `prediction_tool.html` # Interface of the tool
### Model Files
- `trained_model.joblib` # Saved Random Forest model
- `ohe_encoder.joblib` # Saved One-Hot Encoder for categorical features
- `model.json` # Model configuration and metadata
### Dependencies
- `requirements.txt` # List of Python packages required to run the application

### Presentation Slides
- `sc1015_slides.pptx` # Presentation slides


## Dataset Description: 
This dataset contains information about TikTok posts, including various engagement metrics, post metadata, and account information. It is used for analyzing user engagement and optimizing content strategies on the TikTok platform.

## Methodology:
### EDA:
We used correlation heatmaps and plots to see which features has the strongest relations to engagement (e.g., like count). Next, we also looked at how posting day, account verification, follower count, and video length affect engagement.

### Data cleaning: 
We cleaned the dataset to remove duplicate or missing values. Then, we reformatted the column names for easier and more organised interpretation. To preserve the data set with real values, we did not remove outliers. 

### Machine Learning: 
Several machine learning models were trained to predict like count: Linear Regression, Decision Tree, and Random Forest. 
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
Lily carried out exploratory data analysis (EDA) and took the lead in cleaning and preparing the dataset for modeling.

Ruiyang was in charge of the machine learning components and implemented the interactive prediction feature on the webpage.

Jingping defined the problem statement, structured the overall project direction, and created the final presentation slides.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.


## Acknowledgments
- [Bright Data](https://brightdata.com/cp/datasets/browse/gd_lu702nij2f790tmv9h?id=hl_0a9bc27d&tab=sample) for providing sample data access.
- [Plotly](https://plotly.com/) for visualization tools.
- [Scikit-learn](https://scikit-learn.org/) for machine learning algorithms.
- [Business Dasher](https://www.businessdasher.com/social-media-for-business-statistics/) for providing statistics on social media use in business.  
- [Investopedia](https://www.investopedia.com/terms/s/social-media-marketing-smm.asp) for insights into social media marketing (SMM) concepts.  
- [South China Morning Post](https://www.scmp.com/tech/article/2155580/tik-tok-hits-500-million-global-monthly-active-users-china-social-media-video) for reporting TikTok's global user growth.  
- [LinkedIn Article by Zqoue](https://www.linkedin.com/pulse/importance-social-media-why-weekends-ideal-engaging-posts-zqoue/) for perspectives on optimal posting times and social media engagement.  



## Getting Started

### Prerequisites

Make sure you have **Python 3.x** installed on your machine.  

You will also need the following Python libraries:  
- `numpy`  
- `pandas`  
- `seaborn`  
- `matplotlib`  
- `scikit-learn`  
- `plotly`  

###  Installation

To install all the required libraries, simply run the command below in your terminal or command prompt:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn plotly
pip install -r requirements.txt
```

### How to run the prediction tool

1. Start the Flask server:
   ```bash
   python server.py
   ```
   The server will start running on `http://localhost:5000`

2. Open `prediction_tool.html` in your web browser

The tool will then show you the predicted number of likes for a Tik Tok video with the characteristics that you input.

