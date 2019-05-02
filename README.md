# SMS-Spam-detector
Spam detection plays an important role in business and other areas. It uses the concept of Natural Language Processing. In this project, I implemented the classification algorithms like Naive Bayes, KNN, Decision Tree, Random Forest and Logistic Regression. Then combined these different algorithms to make a voting classifier to increase the reliability of the output. Then deployed this project on Heroku using Flask.
[See live demo](https://machineer-3.herokuapp.com/)
## HOW TO RUN ON LOCAL COMPUTER -

1. OPEN CMD
2. NAVIGATE TO PROJECT FOLDER
3. SIMPLY TYPE - python app.py
4. IT WILL SHOW YOU AN URL TO OPEN 
5. COPY IT AND PASE IN WEB BROWSER.
6. IT SHOULD BE UP AND RUNNING, ENJOY.

## HOW TO DEPLOY ON HEROKU CLOUD:

1. OPEN CMD
2. NAVIGATE TO PROJECT FOLDER
3. TYPE git init
4. TYPE heroku login
5. TYPE heroku git:clone -a YOUR-APP-NAME
6. TYPE git add .
7. TYPE git commit -m "YOUR ANY MESSAGE"
8. TYPE git push heroku master

**IF EVERYTHING GOES WELL THE APP SHOULD BE UP AND RUNNING ON HEROKU CLOUD.
** NEED TO INCLUDE Procfile AND requirements.txt IN ROOT FOLDER, BUT IN THIS CASE I HAVE DONE IT, SO JUST DIRECTLY RUN ABOVE COMMANDS. ENJOY 
