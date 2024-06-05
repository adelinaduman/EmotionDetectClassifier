
### EmotionDetectClassifier: Machine Learning for Emotion Recognition in Speech

## Project Overview
EmotionDetectClassifier is a comprehensive project focused on recognizing emotions in speech using machine learning techniques. The project leverages the Emotional Prosody Speech and Transcripts dataset, containing 2,324 WAV files from seven speakers, labeled with 15 distinct emotions. The aim is to extract meaningful features, perform classification experiments, and conduct error analysis to improve emotion 
recognition accuracy.

# Dataset
The speech segments are extracted from the Emotional Prosody Speech and Transcripts dataset. The dataset includes:
- Files: 2,324 WAV files
- Speakers: 7 speakers (cc, cl, gg, jg, mf, mk, mm)
- Emotions: 15 emotions (anxiety, boredom, cold-anger, contempt, despair, disgust, elation, happy, hot-anger, interest, neutral, panic, pride, sadness, shame)

# Notebooks: 
- Features.ipynb: In this notebook I perform Feature Analysis: extract six features from each speech segment:
    - Min, max, mean of pitch
    - Min, max, mean of intensity
-  Classification_features.ipynb:  Used openSMILE toolkit and Random Forest classifier for emotion prediction. 

# How to Run
Ensure Praat, Parselmouth, and openSMILE are installed.
Place your recordings in the recordings folder.
Run the feature_extraction.ipynb script to extract features.
Run the classification.ipynb script for classification experiments.

# Tech Stack
- Praat Software: Praat
- Parselmouth Library: Parselmouth
- openSMILE Toolkit: openSMILE

## Possible Improvements:
-- Feature Engineering, to improve performance, I can introduce new features or refine existing ones to capture more nuances in emotional speech. For example,  features like intonation patterns, speech tempo, and energy levels might enhance the classifier's ability to differentiate between subtly different emotions that performed not well - like despair, neural, sadness etc.

-- Experiment with advanced Modeling Techniques, trying other models that can capture temporal patterns like LSTM or using a more nuanced ensemble method(Boosted Tree) and performing more hyperparameter tuning, might also produce better results. Specifically to my case – that 200 trees and a depth of 20 provided best results so far, experimenting within a close range to these values while adjusting other parameters like min_samples_split and min_samples_leaf 

-- Using techniques like SMOTE for oversampling the minority classes or adjusting class weights in classifiers might help improve the recall for underrepresented emotions. In addition, as outlined above the dataset was imbalanced, some speakers like gg had more examples(around 400), which produced better results. Making the dataset more consistent and having similar distributions is also a way for me to improve the 

-- Randomforest classifier. 
Hyperparameter Optimization: Fine-tuning the model's parameters can sometimes lead to better performance. Considering a broader or more focused range based on preliminary results can help. Given that 200 trees and a depth of 20 provided the best results so far, experimenting within a close range to these values while adjusting other parameters like min_samples_split and min_samples_leaf could refine performance 



## Detailed Overview of the Results 

# Model Type: RandomForestClassifier
I chose this particular model type because it's an ensemble learning method for classification. For classification tasks, the output is determined by the majority voting of decision trees, making the RandomForest model —> especially robust against overfitting and effective in handling complex data structures. Because it can handle non-linear relationships, and is robust to overfitting, I decided to proceed with this model type. 

# Architecture — Parameters and Configuration:
- n_estimators: This parameter controls the number of trees in the forest. In script, two values are tested: 100 and 200, which provide a good range to evaluate best performance, but still grid is not too large to make it too computationally expensive
- max_depth: maximum depth of each tree. Script tests trees with maximum depths of 10 and 20.
- min_samples_split: min # of samples required to split internal nodes. I tested values – 2 and 5, which influenced - how detailed each tree in the forest can get.
- min_samples_leaf: min # of samples required to be at leaf node.  I tested - 1 and 2, that also control granularity of learned patterns at leaf level. This parameter, therefore  affecting both bias and variance. I did not include a lot of values here to not make it too varied, but still I did not want to underfit

# Hyperparameter Tuning Part – 
-- I Used Grid Search(with cv=3), to explore combinations of the above parameters. I wanted to optimize the RandomForestClassifier performance. Model training did not take a long time, and tuning improved my performance from 0.17 aggregate accuracy to 0.21 which is a good improvement. 

-  I also used —- Leave-One-Group-Out Cross-Validation which made the model more generalizable across unseen data. I believe that this method is particularly effective for this dataset where  data may have group-specific characteristics – in this case, the different speakers.

# Execution and Evaluation: 
I used classification reports from sklearn. I return aggregated average accuracy and aggregated average F1 over the 7 experiments. 
- Detailed classification report Output for Each Fold: best-performing model parameters and the corresponding performance metrics are outputted, offering detailed insights into how model settings impact performance.



# Analysis of the Results: 
Analyzing results from my best performing leave-one-speaker-out experiment – It was Third fold with an aggregate accuracy of 0.28, we see mixed results across different emotions, indicating varying levels of difficulty in emotion classification.


# Best classifier parameters were: RandomForestClassifier(max_depth=20, n_estimators=200, random_state=42)
- These results were improvements compared to the base model I had which did not do any tuning, that had an overall accuracy of around 0.17. However the model is still lacking, it is performing better than random chance, there's still  substantial room for improvement. In addition, the poor performance and errors, could be due to class imbalance. The speaker gg has more examples, compared to the cc which has only 265, this caused variability inr results and lowered the performance(lower aggregate accuracy).


# Best Predicted/Easiest Classes:
-- Overall across my results, Elation and Hot-Anger emotions –  showed relatively high precision and recall. The reason why I think they were predicted the best, is becaus these particular  emotions might have distinct or exaggerated features in speech which makes them more distinguishable by the classifier. Here the f-1 score is almost 0.57 which is really high compared to lower results that have values around 0.05--0.10. My Random Forest classifier here  is doing a good job  recognizing happiness from speech. I believe this is due to specific acoustic characteristics associated with happiness, such as — higher pitch, increased speech rate, and overall positive affect, likely contributing to its higher ease of prediction and overall high results.
- Hot-anger also shows relatively high precision, recall, and F1-score across most folds. Hot-anger  for the third fold has    0.50  for precision,    0.64  for recall which is pretty high and    0.56 for f1-score and it is pretty consistent across all folds.   The strong intensity and characteristic vocal features that are usually  associated with anger, and can be observed in extracted features –like raised pitch, rapid speech rate, and heightened volume are reasons why it is easier to predict. Therefore, it is more distinguishable from other emotions, resulting in better predictions
- Disgust is the 3rd emotion that is also easily predicted.  Disgust exhibits reasonably high precision, but lower recall and F1-score compared to happy and hot-anger(disgust had 0.59 precision,       0.20 recall   and  0.29 f-1score ) High precision suggests that my classifier is relatively successful in identifying disgust from other forms of speech. I believe that specific vocal cues like vocal tension, nasality, and harshness associated with disgust that are really specific to this emotion — contribute to its relatively easier recognition.

# Most Difficult Classes:
- Despair and sadness- across almost all folds, scores for despair are very low(0.05) or zero, especially in precision and recall.  This indicates that the model often fails to correctly identify or predict moments of despair correctly. The reason for this - subtlety of Expression(individual for each one) and acoustic similarity. Some emotions like 'Despair' or 'Neutral' have fewer examples than others, leading to less training data and poorer generalization

- Neutral emotions like boredom,  consistently showed low precision, recall, and F1-score across all folds(it had 0 values in a lot of cases). Reason is absence of strong emotional markers and clear vocal cues, complicates classification tasks for neutral emotions. Pride is another one that has really low precision, recall, and F1-score across most folds. Reason - subjective nature and variability in vocal expression associated with pride make it challenging.

- Boredom emotion has variable performance across folds, with generally low precision, recall, and F1-score(around 0.14), but still performs better than anxiety. Reasons for this is I think  low energy and lack of engagement characteristic of boredom might not manifest clearly in speech, making it quite difficult for the classifier to differentiate boredom from other emotions. Again boredom is a bit subjective emotion similar to anxiety, and can be easily mismatched with neural – because usually intensity(is lower), pitch is lower and classifiers might find it difficult to distinguish between those two. . 

- Anxiety consistently shows low precision, recall, and F1-score across all folds(around 0.10 usually which is pretty low). I think the reason for that is subtle and nuanced expression of anxiety, everyone has personal perception and way of expressing anxiety, this emotion is individual. Therefore there is a lot of inconsistencies, and more variability in results → making it difficult to predict. From personal experience, anxiety manifests differently, it lacks distinct vocal features, makes it challenging to detect solely from speech. For these reasons my RandomForest Classifiers fails to capture complexity of anxiety-related vocal patterns.


