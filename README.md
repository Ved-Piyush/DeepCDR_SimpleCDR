# DeepCDR_SimpleCDR

Download the files from drive: [https://drive.google.com/drive/folders/1Aeio6Gah60S_Sz2GcL3CxVEJfUQXF-hL?usp=drive_link](https://drive.google.com/drive/folders/1Aeio6Gah60S_Sz2GcL3CxVEJfUQXF-hL?usp=drive_link) into the data folder in this repository.


Scripts to fit the three types of multi-arm CDR models are described below: 

1.  **DeepCDR:**  [SimplerCDR_Exact_Network_more_dropout.ipynb](https://github.com/Ved-Piyush/DeepCDR_SimpleCDR/blob/main/SimplerDeepCDR/SimplerCDR_Exact_Network_more_dropout.ipynb)

2.  **DualGCN:**  [SimplerGCN_Exact_Network_more_dropout.ipynb](https://github.com/Ved-Piyush/DeepCDR_SimpleCDR/blob/main/SimplerDeepCDR/SimplerGCN_Exact_Network_more_dropout.ipynb)

3. **FullGCN:**  [SimplerCDRGCN_Exact_Network_more_dropout.ipynb](https://github.com/Ved-Piyush/DeepCDR_SimpleCDR/blob/main/SimplerDeepCDR/SimplerCDRGCN_Exact_Network_more_dropout.ipynb)

## Dropout Scenarios

Dropout can be activated during the prediction stage to give a range of predictions for each test sample. We tested out various dropout activation scenarios with the **FullGCN:** model: 

1. Dropout is active both during training and during prediction. In this case, we can obtain a band of predictions for each test sample and compute empirical 95% prediction intervals. [Active Dropout both Training and Prediction](https://github.com/Ved-Piyush/DeepCDR_SimpleCDR/blob/main/SimplerDeepCDR/SimplerCDRGCN_Exact_Network_more_dropout_active_both.ipynb)

2. Dropout is active during training but not during prediction. Since dropout is not activated during prediction for each test sample only one prediction can be obtained, and therefore, no empirical prediction intervals can be calculated. [Active Dropout during Training but not during Prediction](https://github.com/Ved-Piyush/DeepCDR_SimpleCDR/blob/main/SimplerDeepCDR/SimplerCDRGCN_Exact_Network_more_dropout_active_only_train_not_pred.ipynb)

3. We can use the model trained in 1 where dropout was active during training and prediction to assess different dropout probabilities during the prediction stage. This makes it easier than training many different **FullGCN:** models with different dropout probabilities, as the **FullGCN:** model has many learnable parameters and is computationally intensive to train. Scroll to the bottom of the code to see the two main plots that show how the RMSE and Pearson Correlation and the coverage and average prediction interval widths change with different dropout proportions. [Different Dropout Probabilities with **FullGCN:**](https://github.com/Ved-Piyush/DeepCDR_SimpleCDR/blob/main/SimplerDeepCDR/SimpleCDRGCN_Dropout_Intervals.ipynb)


