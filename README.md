# Deep Learning Challenge #
## Background ##
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.
From Alphabet Soup’s business team, you have received access to a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:
+ EIN and NAME—Identification columns
+ APPLICATION_TYPE—Alphabet Soup application type
+ AFFILIATION—Affiliated sector of industry
+ CLASSIFICATION—Government organization classification
+ USE_CASE—Use case for funding
+ ORGANIZATION—Organization type
+ STATUS—Active status
+ INCOME_AMT—Income classification
+ SPECIAL_CONSIDERATIONS—Special considerations for application
+ ASK_AMT—Funding amount requested
+ IS_SUCCESSFUL—Was the money used effectively
## Step 1: Preprocess the Data ##
Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.
1.	From the provided cloud URL, read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
+ What variable(s) are the target(s) for your model?
![image](https://github.com/user-attachments/assets/62b12b3e-0674-426f-b3c5-609f351dad08)
 
2.	Drop the EIN and NAME columns.
![image](https://github.com/user-attachments/assets/12ff44ba-ad8a-4aa7-a8a3-2878481b533e)

3.	Determine the number of unique values for each column.
![image](https://github.com/user-attachments/assets/b7c2c7ce-b60f-43ad-9791-f3487e3a4033)

4.	For columns that have more than 10 unique values, determine the number of data points for each unique value.
![image](https://github.com/user-attachments/assets/dca4372f-5c4f-4672-8947-2b50f0ae24ac)

5.	Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, Other, and then check if the replacement was successful.
![image](https://github.com/user-attachments/assets/937df429-a0d0-4da7-bc8c-e6878d3c8f7e)

6.	Use pd.get_dummies() to encode categorical variables.
![image](https://github.com/user-attachments/assets/c0f91017-1a1e-461f-a2a1-edc508bc1016)

7.	Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
![image](https://github.com/user-attachments/assets/62567df3-694e-4dc1-a6e7-18b85069b767)

9.	Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.
![image](https://github.com/user-attachments/assets/2a3c761c-c2e6-4e64-a438-507d9a4cea3d)

## Step 2: Compile, Train, and Evaluate the Model ##
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.
1.	Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
2.	Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
3.	Create the first hidden layer and choose an appropriate activation function.
4.	If necessary, add a second hidden layer with an appropriate activation function.
5.	Create an output layer with an appropriate activation function.
6.	Check the structure of the model.
![image](https://github.com/user-attachments/assets/74cdba08-b307-4b3c-8b32-90d4d0e1ce5e)

7.	Compile and train the model.
![image](https://github.com/user-attachments/assets/052fd1ed-9d61-41bb-946b-9c15190c3dc9)

8.	Create a callback that saves the model's weights every five epochs.
![image](https://github.com/user-attachments/assets/bd9d1515-6485-48f9-a2a0-065f8c38b8f1)

9.	Evaluate the model using the test data to determine the loss and accuracy.
![image](https://github.com/user-attachments/assets/30033beb-bdc0-4c1e-a306-1452aeb3b7c0)

10.	Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5. - Attached in the main branch

## Step 3: Optimize the Model ##

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.
Use any or all of the following methods to optimize your model:
•	Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
+ Dropping more or fewer columns.
+ Creating more bins for rare occurrences in columns.
+ Increasing or decreasing the number of values for each bin.
+ Add more neurons to a hidden layer.
+ Add more hidden layers.
+ Use different activation functions for the hidden layers.
 + Add or reduce the number of epochs to the training regimen.
Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.
1.	Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.
2.	Import your dependencies and read in the charity_data.csv to a Pandas DataFrame from the provided cloud URL.
3.	Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
![image](https://github.com/user-attachments/assets/69849ba7-6e7b-4998-ae16-51d2876078a4)
![image](https://github.com/user-attachments/assets/fc6633a6-c3a5-4126-8811-72e399e6d895)

4.	Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

5.	Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5. - attached to main branch.
   
## Step 4: Write a Report on the Neural Network Model ##

For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.
The report should contain the following:
1.	Overview of the analysis: Explain the purpose of this analysis.
### The goal of this project is to build a binary classification model using deep learning to help Alphabet Soup, a nonprofit foundation, determine which applicants are most likely to be successful if funded. The dataset provided includes over 34,000 historical records of funded organizations, with features describing organizational type, funding amount, use case, and more.Our objective is to preprocess the data and develop a neural network using TensorFlow/Keras to predict the IS_SUCCESSFUL target variable with an accuracy of at least 75%. ### 
2.	Results: Using bulleted lists and images to support your answers, address the following questions:
A) Data Preprocessing
+ What variable(s) are the target(s) for your model?
### IS_SUCCESSFUL – indicates whether the funding led to successful outcomes. ### 
+ What variable(s) are the features for your model?
### All remaining columns after dropping identification columns and encoding categoricals, for example: APPLICATION_TYPE, CLASSIFICATION, USE_CASE, ORGANIZATION, ASK_AMT, INCOME_AMT, etc. ### 
+ What variable(s) should be removed from the input data because they are neither targets nor features?
### EIN – an identification number, NAME – not relevant for training and introduces high cardinality. ###

B) Compiling, Training, and Evaluating the Model
+ How many neurons, layers, and activation functions did you select for your neural network model, and why? 
+ Were you able to achieve the target model performance?
+ What steps did you take in your attempts to increase model performance?
### We built and tested multiple neural network models to optimize performance:
Final Optimized Model Architecture:
Input Layer: 116 input features (after encoding)
Hidden Layer 1: 128 neurons, ReLU activation
Hidden Layer 2: 64 neurons, ReLU activation
Hidden Layer 3: 32 neurons, ReLU activation
Output Layer: 1 neuron, Sigmoid activation
Why this structure?
We gradually increased the complexity of the model.
More neurons and hidden layers allowed the network to capture more complex relationships in the data.
ReLU is an effective and standard choice for hidden layers, while Sigmoid suits binary output.
Model Performance
Best Achieved Accuracy: ~75.2%
Target Met: Yes, after several iterations. ### 
3.	Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
### The optimized deep learning model was able to achieve just over 75% accuracy, meeting Alphabet Soup’s performance goal. However, further improvements may be limited by the structure of the data or hidden biases in the features. Using Random Forest Classifier or XGBoost may inprove the results. By experimenting with these alternative methods or hybrid approaches, the foundation may achieve even higher prediction accuracy and insight into applicant success factors. ###
