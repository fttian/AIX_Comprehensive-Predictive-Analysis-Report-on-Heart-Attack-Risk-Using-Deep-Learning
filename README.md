# Comprehensive Predictive Analysis Report on Heart Attack Risk Using Deep Learning

**Dataset Source:** [Kaggle Heart Attack Prediction Dataset](https://www.kaggle.com/datasets/ahmedmohamedibrahim1/heart-attack-prediction-dataset)  

**Members:** 범유가, 무용학과, fttian9@naver.com
---

## 1. Executive Summary

This report evaluates the performance of two distinct deep learning architectures—a **Standard Multi-Layer Perceptron (MLP)** and a **Residual Multi-Layer Perceptron (Residual MLP)**—configured to predict heart attack risks. Using a tabular health dataset of 8,763 patients, both models were implemented, trained, and tested.

The evaluation metrics reveal a classic challenge in clinical machine learning: **severe class imbalance and the limitations of deep learning on tabular data**. While both models boast an overall validation accuracy of approximately 64%, a deep dive into the recall metrics reveals that both models struggle significantly to isolate the positive class (patients genuinely at risk). This report breaks down the underlying bottlenecks and provides actionable solutions.

---

## 2. Dataset Overview and Exploratory Insights

### 2.1 Data Profile

The dataset consists of 8,763 rows and 25 features tracking demographic profiles, clinical measurements, and lifestyle choices.

* **Missing Values:** 0 missing values across all columns.
* **Target Feature:** `Heart Attack Risk` (Binary: `0` for No Risk, `1` for Risk).
<img width="708" height="915" alt="image" src="https://github.com/user-attachments/assets/7c07c365-aef7-4e1c-ba96-33f800492f9f" />
<img width="438" height="894" alt="image" src="https://github.com/user-attachments/assets/75707153-20c3-4463-b501-0db87fd47c3b" />

### 2.2 The Class Imbalance Bottleneck

The training target label is distributed as follows:

* **Class 0 (No Risk):** 64.18%
* **Class 1 (Risk):** 35.82%

```


```

> **Critical Observation:** Because Class 0 constitutes nearly two-thirds of the data, a naive model that predicts "No Risk" for every single patient would automatically achieve an accuracy of **64.18%**. This explains the baseline threshold observed during training.

---

## 3. Data Preprocessing Pipeline

To guarantee stable gradient descent within the neural networks, a structured preprocessing pipeline was executed:

1. **Feature Engineering:** The raw string feature `Blood Pressure` (e.g., "158/88") was extracted and split into two distinct numerical components: `Systolic_BP` and `Diastolic_BP`. High-cardinality geographical indicators (`Country`, `Continent`, `Hemisphere`) were dropped to reduce noise.
2. **Numerical Standardization:** Features such as `Age`, `Cholesterol`, `BMI`, and `Triglycerides` were scaled using `StandardScaler` to exhibit a mean ($\mu = 0$) and a standard deviation ($\sigma = 1$).
3. **Categorical Encoding:** Nominal variables (`Sex`, `Diet`) were processed via `OneHotEncoder(drop='first')` to protect against multicollinearity.
4. **Data Splitting:** Data was split into **80% Training (7,010 samples)** and **20% Testing (1,753 samples)**, utilizing stratified sampling to preserve the target class ratio.
<img width="558" height="150" alt="image" src="https://github.com/user-attachments/assets/6ac6e141-a0e6-4649-b08e-25bbe058dd0c" />

---

## 4. Model Architectures

Two deep learning models were compared to test whether structural optimization could extract deeper tabular correlations:

### Model 1: Standard Multi-Layer Perceptron (MLP)

A feed-forward network implementing dense layer contractions accompanied by Batch Normalization (BN) and Dropout layers to manage regularized weights.

* **Layer Sequence:** Input (23 features) $\rightarrow$ Dense(128, ReLU) $\rightarrow$ BN $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(64, ReLU) $\rightarrow$ BN $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(32, ReLU) $\rightarrow$ BN $\rightarrow$ Dropout(0.2) $\rightarrow$ Dense(1, Sigmoid).

### Model 2: Residual Multi-Layer Perceptron (Residual MLP)

Designed to retain original feature embeddings across deep propagation using skip connections (identity shortcut addition).

* **Layer Sequence:** Contains parallel pathways where the transformations of an intermediate block are directly summed back into the prior layer's input mapping: $x_{l} = x_{l-1} + \mathcal{F}(x_{l-1})$.
<img width="918" height="758" alt="image" src="https://github.com/user-attachments/assets/78499e83-720e-4f8f-96b8-8d8e68b21cc9" />

---

## 5. Training Dynamics & Convergence Behavior

Both networks were trained across a maximum of 50 epochs utilizing the `Adam` optimizer, `Binary Crossentropy` loss, and automated callback policies (`EarlyStopping` with a patience threshold of 10; `ReduceLROnPlateau`).

### 5.1 Standard MLP Training Logs

* **Early Termination:** The model triggered `EarlyStopping` at **Epoch 21**.
* **Loss Behavior:** Validation loss hit a minimum floor at **0.6620** during Epoch 11, after which it stagnated. Meanwhile, training AUC continued to crawl upward to 0.6152, highlighting minor overfitting.
<img width="1836" height="1035" alt="image" src="https://github.com/user-attachments/assets/02f346e2-8e0c-471e-838c-c0951f7f75d3" />

### 5.2 Residual MLP Training Logs

* **Early Termination:** The model terminated even earlier, at **Epoch 10**.
* **Loss Behavior:** Training loss decreased rapidly (0.8156 $\rightarrow$ 0.5804) while training AUC shot up aggressively to **0.7172**. However, validation loss concurrently worsened (0.6626 $\rightarrow$ 0.7002), and validation AUC flatlined near **0.5148**.
<img width="1842" height="612" alt="image" src="https://github.com/user-attachments/assets/0ffd5603-a112-4c37-b479-d1ccfca9550c" />

```



```
<img width="1165" height="470" alt="image" src="https://github.com/user-attachments/assets/75825b02-4698-4563-a6f1-7c8125c718cd" />
<img width="1165" height="470" alt="image" src="https://github.com/user-attachments/assets/304fb97e-fa7f-46b1-a4a6-14b23e116811" />

> **Analysis:** The training curves show that the Residual MLP has too much capacity for this specific dataset. It rapidly memorizes the statistical variations of the training sample split (overfitting) without mapping generalizable boundaries for unseen validation records.

---

## 6. Evaluation and Performance Analysis

Upon final inference mapping against the isolated validation subset (1,753 records consisting of 1,125 Class 0 items and 628 Class 1 items), the performance matrices show stark limitations:

### 6.1 Classification Reports

#### Standard MLP

```
              precision    recall  f1-score   support

           0       0.64      0.99      0.78      1125
           1       0.25      0.00      0.01       628

    accuracy                           0.64      1753

```
<img width="452" height="393" alt="image" src="https://github.com/user-attachments/assets/6e83c2e2-ea63-4303-8c89-54e1f9a5b775" />

#### Residual MLP

```
              precision    recall  f1-score   support

           0       0.64      0.99      0.78      1125
           1       0.35      0.01      0.02       628

    accuracy                           0.64      1753

```
<img width="452" height="393" alt="image" src="https://github.com/user-attachments/assets/1b9b16d1-5962-477e-992c-bcea84d52a4c" />

### 6.2 Deep Dive Metric Breakdown

* **The Accuracy Illusion:** The apparent 64% accuracy of both models is entirely driven by their ability to classify the majority class (Class 0).
* **The Recall Crisis:** The Standard MLP registered a Recall of **0.00** for Class 1, and the Residual MLP achieved only **0.01**. Out of 628 actual heart attack risk cases, the models caught almost none. The networks defaulted toward classifying almost all patients as safe.

```



```

### 6.3 ROC-AUC Evaluation

The Area Under the Receiver Operating Characteristic (ROC-AUC) score measures the probability that the model will rank a randomly chosen positive instance higher than a randomly chosen negative one.

```


```

<img width="702" height="547" alt="下载" src="https://github.com/user-attachments/assets/be9b47f7-405f-4eaf-8d95-17c42421e7f2" />


With final validation AUC metrics hovering near **0.51–0.52**, both neural networks are performing only slightly better than a random coin toss on unseen data.

---

## 7. Conclusions and Key Takeaways

1. **Tabular Deep Learning Pitfalls:** This outcome aligns with established machine learning literature: deep neural networks often struggle with heterogeneous, low-sample tabular benchmarks compared to ensemble tree-based methods. Without spatial/temporal invariants (like pixels in CNNs), MLPs easily overfit or fail to learn meaningful representations from weak signals.
2. **The Danger of Raw Accuracy:** Relying heavily on raw accuracy in an imbalanced scenario can lead to dangerous clinical outcomes, as patients at high risk are misclassified as safe.

---

## 8. Actionable Next Steps & Recommendations

To pivot this project toward high clinical utility, the following iterative adjustments are strongly recommended:

* **Implement Class Weight Adjustments:** Inject a corrective penalty scaling matrix into the loss configuration during training:
```python
class_weights = {0: 1.0, 1: (64.18 / 35.82)}
model.fit(..., class_weight=class_weights)

```


* **Apply Synthetic Minority Over-sampling Technique (SMOTE):** Balance the training sample partition prior to feeding the neural network by synthetically manufacturing minority representations.
* **Transition to Tree-Based Classifiers:** Pivot away from Deep Learning and implement **XGBoost, LightGBM, or CatBoost**. These architectures natively handle unscaled, tabular feature limits and leverage gradient-boosted decision trees that are inherently more robust for this data structure.
* **Adjust Prediction Thresholds:** Lower the classification probability threshold from 0.50 down to 0.30 or 0.25 to prioritize optimizing **Recall** over Precision, casting a wider net to catch high-risk patients.

---

## 9. References

1. Shwartz-Ziv, R., & Armon, A. (2022). *Tabular data: Deep learning is not all you need*. Information Fusion, 81, 84-90.
2. Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS Datasets and Benchmarks Track.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. Proceedings of the IEEE conference on computer vision and pattern recognition, 770-778.
