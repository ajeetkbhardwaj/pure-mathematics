
<!--header: "Ajeet Kumar & ajeetskbp9843@gmail.com"-->

# <!--fit-->MCQ-Problems
# <!--fit-->Machine Learning : SL, UL, SSL and RL
---
**1. What is machine learning ?**
   - The selective acquisition of knowledge through the use of computer programs
   - The selective acquisition of knowledge through the use of manual programs
   - The autonomous acquisition of knowledge through the use of computer programs
   - The autonomous acquisition of knowledge through the use of manual programs
> Explanation: Machine learning is the autonomous acquisition of knowledge through the use of computer programs.

---

**2. K-Nearest Neighbors (KNN) is classified as what type of machine learning algorithm ?**
a) Instance-based learning
b) Parametric learning
c) Non-parametric learning
d) Model-based learning

> Explanation: KNN doesn’t build a parametric model of the data. Instead, it directly classifies new data points based on the k nearest points in the training data.

---

**3. Which of the following is not a supervised machine learning algorithm ?**
a) K-means
b) Naïve Bayes
c) SVM for classification problems
d) Decision tree

> Explanation: Decision tree, SVM (Support vector machines) for classification problems and Naïve Bayes are the examples of supervised machine learning algorithm. K-means is an example of unsupervised machine learning algorithm.
---


**4.  What’s the key benefit of using deep learning for tasks like recognizing images ?**
a) They need less training data than other methods.
b) They’re easier to explain and understand than other models.
c) They can learn complex details from the data on their own.
d) They work faster and are more efficient computationally.

> Explanation: Deep learning is great at figuring out intricate details from data, especially in tasks like recognizing images.
---


**5. Which algorithm is best suited for a binary classification problem ?**
a) K-nearest Neighbors
b) Decision Trees
c) Random Forest
d) Linear Regression

> Explanation: Decision Trees are versatile and can be used for classification problems, particularly for binary classification, where the output is divided into two classes.

---


**6.  What is the key difference between supervised and unsupervised learning ?**
a) Supervised learning requires labeled data, while unsupervised learning does not.
b) Supervised learning predicts labels, while unsupervised learning discovers patterns.
c) Supervised learning is used for classification, while unsupervised learning is used for regression.
d) Supervised learning is always more accurate than unsupervised learning.

> Explanation: The presence or absence of labeled data in the training set distinguishes supervised and unsupervised learning approaches.

---


**7. Which type of machine learning algorithm falls under the category of “unsupervised learning” ?**
a) Linear Regression
b) K-means Clustering
c) Decision Trees
d) Random Forest

> Explanation: K-means Clustering is an example of unsupervised learning used for clustering unlabeled data based on similarities.

---

**8. Which of the following statements is true about AdaBoost?
a) It is particularly prone to overfitting on noisy datasets
b) Complexity of the weak learner is important in AdaBoost
c) It is generally more prone to overfitting
d) It improves classification accuracy

> Explanation: AdaBoost is generally not more prone to overfitting but is less prone to overfitting. And it is prone to overfitting on noisy datasets. If you use very simple weak learners, then the algorithms are much less prone to overfitting and it improves classification accuracy. So Complexity of the weak learner is important in AdaBoost.
---

**9. Which one of the following models is a generative model used in machine learning ?**
a) Support vector machines
b) Naïve Bayes
c) Logistic Regression
d) Linear Regression

> Explanation: Naïve Bayes is a type of generative model which is used in machine learning. Linear Regression, Logistic Regression and Support vector machines are the types of discriminative models which are used in machine learning.
---

**10. An artificially intelligent car decreases its speed based on its distance from the car in front of it. Which algorithm is used ?**
a) Naïve-Bayes
b) Decision Tree
c) Linear Regression
d) Logistic Regression

> Explanation: The output is numerical. It determines the speed of the car. Hence it is not a classification problem. All the three, decision tree, naïve-Bayes, and logistic regression are classification algorithms. Linear regression, on the other hand, outputs numerical values based on input. So, this can be used.
---

**11. Which of the following statements is false about Ensemble learning ?**
a) It is a supervised learning algorithm
b) It is an unsupervised learning algorithm
c) More random algorithms can be used to produce a stronger ensemble
d) Ensembles can be shown to have more flexibility in the functions they can represent

> Explanation: Ensemble learning is not an unsupervised learning algorithm. It is a supervised learning algorithm that combines several machine learning techniques into one predictive model to decrease variance and bias. It can be trained and then used to make predictions. And this ensemble can be shown to have more flexibility in the functions they can represent.
---
**12. Which of the following statements is true about stochastic gradient descent ?**
a) It processes one training example per iteration
b) It is not preferred, if the number of training examples is large
c) It processes all the training examples for each iteration of gradient descent
d) It is computationally very expensive, if the number of training examples is large

> Explanation: Stochastic gradient descent processes one training example per iteration. That is it updates the weight vector based on one data point at a time. All other three are the features of Batch Gradient Descent.
---

**13. Decision tree uses the inductive learning machine learning approach.**
a) False
b) True

> Explanation: Decision tree uses the inductive learning machine learning approach. Inductive learning enables the system to recognize patterns and regularities in previous knowledge or training data and extract the general rules from them. A decision tree is considered to be an inductive learning task as it uses particular facts to make more generalized conclusions.
---

**14. What elements describe the Candidate-Elimination algorithm?**
a) depends on the dataset
b) just a set of candidate hypotheses
c) just a set of instances
d) set of instances, set of candidate hypotheses

> Explanation: A set of instances is required. A set of candidate hypotheses are given. These are applied to the training data and the list of accurate hypotheses is output in accordance with the candidate-elimination algorithm.
---

<!--header: "" -->
**15. Which of the following statements is not true about boosting?**
a) It mainly increases the bias and the variance
b) It tries to generate complementary base-learners by training the next learner on the mistakes of the previous learners
c) It is a technique for solving two-class classification problems
d) It uses the mechanism of increasing the weights of misclassified data in preceding classifiers

> Explanation: Boosting does not increase the bias and variance but it mainly reduces the bias and the variance. It is a technique for solving two-class classification problems. And it tries to generate complementary base-learners by training the next learner (by increasing the weights) on the mistakes (misclassified data) of the previous learners.

---
1. Formal Learning Models
   - Statistical Learning Frameworks
   - Empirical Risk Minimization
   - PAC Learning
2. Version Spaces :  find-s algorithm and candidate elimination algorithm.
   - Version Spaces
   - Find S-Algorithm
   - Candidate Elimination Algorithm

---

**16. How are the points in the domain set given as input to the algorithm?**
a) Vector of features
b) Scalar points
c) Polynomials
d) Clusters
> Explanation: The variables are converted into a vector of features, and then given as an input to the algorithm. The vector is of the size (number of features x number of training data sets). The output of the learner is usually given as a polynomial.

---

**17. To which input does the learner has access to?
a) Testing Data
b) Label Data
c) Training Data
d) Cross-Validation Data
> Explanation: The learner gets access to a particular set of data on which it trains. This data is called as training data. Testing Data is used for testing of the learner’s outputs. The best outputs are then used on the cross-validation data. The label data is a representation of different types of the dependent variables.

---

**18.  The set which represents the different instances of the target variable is known as ______**
a) domain set
b) training set
c) label set
d) test set
> Explanation: Label Set denotes all the possible forms the target variable can take (for e.g. {0,1} or {yes, no} in a logistic regression problem). Domain Set represents the vector of features, given as input to the learner. Training Set and Test Set are parts of the Domain Set which are used for training and testing respectively.
---

**19.  What is the learner’s output also called?**
a) Predictor, or Hypothesis, or Classifier
b) Predictor, or Hypothesis, or Trainer
c) Predictor, or Trainer, or Classifier
d) Trainer, or Hypothesis, or Classifier
> Explanation: The output is called a predictor when it is used to predict the type or the numerical value of the target variable. It is called a hypothesis when it is a general statement about the data set. It is called a classifier when it is used to classify the training  set in two or more types.

---

**20.  It is assumed that the learner has prior knowledge about the probability distribution which generates the instances in a training set.**
a) True
b) False

> Explanation: The learner has no prior knowledge about the distribution. It is assumed that the distribution is completely arbitrary. It is also assumed that there is a function which “correctly” labels the training examples. The learner’s job is to find out this function.

---
**21.  The labeling function is known to the learner in the beginning.**
a) True
b) False

> Explanation: The function is unknown to the learner as this is what the learner is trying to find out. In the beginning, the learner just knows about the training set and the corresponding label set.

---

**22. The papaya learning algorithm is based on a dataset that consists of three variables – color, softness, tastiness of the papaya. Which is more likely to be the target variable?**
a) Tastiness
b) Softness
c) Papaya
d) Color

> Explanation: The tastiness is dependent on how ripe the papaya is. The ripeness is determined by the color and softness. Hence color and softness are the independent variables and the tastiness is the dependent variable or target variable.

---
**23.  The error of classifier is measured with respect to _________**
a) variance of data instances
b) labeling function
c) probability distribution
d) probability distribution and labeling function
> Explanation: The error is the probability of choosing a random instance from the data set and then misclassifying it using the labeling function.

---
**24. What is not accessible to the learner?**
a) Training Set
b) Label Set
c) Labeling Function
d) Domain Set

> Explanation: The learner has access to the domain set, from which it extracts the training set. The label set is also given. Then the algorithm is applied to the training set to teach the learner, a function to determine the correct label of a given instance. This is the labeling function.

---
**25. What is the possiblility that Subets of a Dataset D are A, B and C as 80%, 10% and 10% sampled resp. Then**
a) A – Training Set, B – Domain Set, C – Cross-Validation Set
b) A – Training Set, B – Test Set, C – Cross-Validation Set
c) A – Training Set, B – Test Set, C – Domain Set
d) A – Test Set, B – Domain Set, C – Training Set
> Explanation: Domain Set comprises of the total input data set. It is usually divided into a training set, a test set and a cross-validation set in the ratio 3:1:1. Since the learner learns about the data set from the training set, the later is usually larger than the test and cross-validation set.

---
**26. The true error is available to the learner.
a) True
b) False
> Explanation: True error is calculated with respect to the probability distribution of the generation of dataset instances and labeling function. These two are not available to the learner. Hence, the learner cannot calculate the true error.

---
**27.  What is one of the drawbacks of Empirical Risk Minimization?**
a) Underfitting
b) Both Overfitting and Underfitting
c) Overfitting
d) No drawbacks
> Explanation: Empirical Risk Minimization makes the learner output a predictor which gives the minimum error on the training set. This often leads to a predictor that is specifically designed to be accurate on the training data set but fails to be highly accurate on the test set, as the predictor was training set-specific. This is overfitting.

---
**28. The error available to the learner is ______**
a) true error
b) error of the classifier
c) training error
d) testing error

> Explanation: The learner only knows about the error it incurred over the training set instances. It is minimized by the learner to produce the labeling function. This is then used on the testing set to generate a testing error. The error produced by randomly selecting an instance from the dataset, and misclassifying it using the labeling function.

---
**29. Which is the more desirable way to reduce overfitting?**
a) Giving an upper bound to the size of the training set
b) Making the test set larger than the training set
c) Giving an upper bound to the accuracy obtained on the training set
d) Overfitting cannot be reduced

Explanation: More the number of training set examples, more specific the predictor is going to be to the training set. Hence reducing it can reduce overfitting. Making the test set larger than the training set will lead to underfitting, which is not desirable. Giving an upper bound on accuracy can abruptly stop the learner at a premature stage.

---
**30.  What is the relation between Empirical Risk Minimization and Training Error?**
a) ERM tries to maximize training error
b) ERM tries to minimize training error
c) It depends on the dataset
d) ERM is not concerned with training error
> Explanation: ERM makes the learner develop a predictor which works well on the training data (data available to the learner). Its aim is to minimize the error. Lesser the error, the better is the predictor (not considering overfitting).

---

**31. What is assumed while using empirical risk minimization with inductive bias?**
a) The learner has some prior knowledge about training data
b) The learner has some knowledge about labeling function
c) Reduction of overfitting may lead to underfitting
d) No assumptions are made

> Explanation: The learner must choose a hypothesis from a set of H, reduced hypothesis space. Since the choice is determined before seeing the training set, the learner needs to have prior knowledge of training data.
---

**32. The hypothesis space H for inductive bias is a finite class.**
a) False
b) True

> Explanation: The hypothesis space H contains a finite number of hypothesizes. The learner is restricted to chose from only these hypothesizes. If hypothesis space is not finite, there is no question of restriction.
---
**33. The assumption that the training set instances are independently and identically distributed is known as the __________**
a) empirical risk assumption
b) inductive bias assumption
c) i.i.d. assumption
d) training set rule

> Explanation: The three letters of i.i.d stands for independently and identically distributed. The instances are not dependent on each other. Every one of them is unique, and it is assumed that the instances follow a certain distribution.
**34. Delta is the __________ parameter of the prediction.**
a) training
b) confidence
c) accuracy
d) computing

> Explanation: The confidence parameter is used to state that the chosen hypothesis will give a successful outcome with a certain probability. This probability is given by (1 – delta).
Sanfoundry Glob

---
**35.  Who introduced the concept of PAC learning?**
a) Francis Galton
b) Reverend Thomas Bayes
c) J.Ross Quinlan
d) Leslie Valiant
> Explanation: Valiant introduced PAC learning. Galton introduced Linear Regression. Quinlan introduced the Decision Tree. Bayes introduced Bayes’ rule and Naïve-Bayes theorem.

---

**36. When was PAC learning invented?**
a) 1874
b) 1974
c) 1984
d) 1884
> Explanation: Leslie Valiant was the inventor of PAC learning. He described it in 1984. It was introduced as a part of the Computational Learning Theory.

---

**37. The full form of PAC is ______**
a) Partly Approximation Computation
b) Probability Approximation Curve
c) Probably Approximately Correct
d) Partly Approximately Correct

> Explanation: Probably Approximately Correct tries to build a hypothesis that can predict with low generalization error (approximately correct), with high probability (probably).

---

**38. What can be explained by PAC learning?**
a) Sample Complexity
b) Overfitting
c) Underfitting
d) Label Function

> Explanation: PAC learning can give a rough estimate of the number of training examples required by the learning algorithm to develop a hypothesis with the desired accuracy.

---

5. What is the significance of epsilon in PAC learning?
a) Probability of approximation < epsilon
b) Maximum error < epsilon
c) Minimum error > epsilon
d) Probability of approximation = delta – epsilon

> Explanation: A concept is PAC learnable by L if L can output a hypothesis with error < epsilon. Hence the maximum error obtained by the hypothesis should be less than epsilon. Epsilon is usually 5% or 1%.

---

6. What is the significance of delta in PAC learning?
a) Probability of approximation < delta
b) Error < delta
c) Probability = 1 – delta
d) Probability of approximation = delta – epsilon

> Explanation: A concept C is PAC learnable by L if L can predict a hypothesis with a certain error with probability equal to 1-delta. Delta is usually very low.
---
7. One of the goals of PAC learning is to give __________
a) maximum accuracy
b) cross-validation complexity
c) error of classifier
d) computational complexity

> Explanation: PAC learning tells us about the amount of effort required for computation so that a learner can come up with a successful hypothesis with high probability.

---

8. A learner can be deemed consistent if it produces a hypothesis that perfectly fits the __________
a) cross-validation data
b) overall dataset
c) test data
d) training data

> Explanation: PAC learning is concerned with the behavior of the learning algorithm. The learner has access to only the training data set.

---
9. Number of hypothesizes |H| = 973, probability = 95%, error < 0.1. Find minimum number of training examples, m, required.
a) 98.8
b) 99.8
c) 99
d) 98

> Explanation: Probability = 1 – delta (d) = 0.95; d = 0.05. Error < epsilon (e); e = 0.1.
m >= 1/e (ln |H| + ln (1/d)) i.e. m> = 1/0.1(ln 973 + ln (1/0.05)) i.e. m>=98.8
Since number of training examples, m, has to be an integer, answer is 99.

---

10. In PAC learning, sample complexity grows as the logarithm of the number of hypothesizes.
a) False
b) True

> Explanation: Sample complexity is the number of training examples required to converge to a successful hypothesis. It is given by m >= 1/e (ln |H| + ln (1/d)), where m is the number of training examples and H is hypothesis space.