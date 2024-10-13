# NuminaJS

NuminaJS is a comprehensive data science package for JavaScript, providing a wide range of tools for data preprocessing, feature selection, data transformation, machine learning algorithms, and data visualization. It's designed to simplify complex data science tasks and make them accessible to JavaScript developers.

## Table of Contents

1. [Installation](#installation)
2. [Features](#features)
3. [Usage](#usage)
   - [Data Reading](#data-reading)
   - [Data Preprocessing](#data-preprocessing)
   - [Feature Selection](#feature-selection)
   - [Data Transformation](#data-transformation)
   - [Handling Imbalanced Data](#handling-imbalanced-data)
   - [Supervised Learning Algorithms](#supervised-learning-algorithms)
   - [Unsupervised Learning Algorithms](#unsupervised-learning-algorithms)
   - [Model Evaluation Metrics](#model-evaluation-metrics)
   - [Cross-Validation](#cross-validation)
   - [Data Visualization](#data-visualization)
   - [Data Export](#data-export)
4. [API Reference](#api-reference)
5. [Contributing](#contributing)
6. [License](#license)

## Installation

```bash
npm install numina-js
```

## Features

- CSV file reading
- Data preprocessing (handling missing values, outlier detection, normalization)
- Feature selection (correlation analysis, chi-square test, ANOVA, Lasso, Ridge, Elastic Net)
- Data transformation
- Handling imbalanced datasets
- Supervised learning algorithms (SVM, Linear Regression, Logistic Regression, Decision Trees, Random Forests, KNN, Naive Bayes, Gradient Boosting, AdaBoost, Voting Classifier)
- Unsupervised learning algorithms (K-Means, Hierarchical Clustering, DBSCAN)
- Comprehensive model evaluation metrics
- Cross-validation techniques
- Data visualization with Chart.js integration
- Data export to various formats (HTML, JSON, PDF)

## Usage

### Data Reading

```javascript
const { readCSV } = require('numina-js');

const data = readCSV('path/to/your/file.csv');
console.log(data);
```

### Data Preprocessing

```javascript
const { handleMissingValues, detectAndHandleOutliers, normalizeData } = require('numina-js');

// Handle missing values
let processedData = handleMissingValues(data, 'column_name', 'mean');

// Detect and handle outliers
processedData = detectAndHandleOutliers(processedData, 'column_name', 'remove');

// Normalize data
processedData = normalizeData(processedData, 'column_name', 'min-max');
```

### Feature Selection

```javascript
const { selectFeatures, chiSquareTest, anovaFTest } = require('numina-js');

// Select features based on correlation
const selectedFeatures = selectFeatures(data, 'correlation', 0.5);

// Perform chi-square test
const chiSquareResults = chiSquareTest(data, targetColumn);

// Perform ANOVA F-test
const anovaResults = anovaFTest(data, targetColumn);
```

### Data Transformation

```javascript
const { transformData } = require('numina-js');

// Apply log transformation
const transformedData = transformData(data, 'column_name', 'log');
```

### Handling Imbalanced Data

```javascript
const { handleImbalance } = require('numina-js');

// Oversample minority class
const balancedData = handleImbalance(data, 'oversample');
```

### Supervised Learning Algorithms

```javascript
const { linearRegression, logisticRegression, svm, randomForests } = require('numina-js');

// Linear Regression
const linearModel = linearRegression(data, targetColumn);

// Logistic Regression
const logisticModel = logisticRegression(data, targetColumn);

// Support Vector Machine
const svmModel = svm(data, targetColumn);

// Random Forests
const rfModel = randomForests(data, targetColumn);
```

### Unsupervised Learning Algorithms

```javascript
const { kMeans, hierarchicalClustering, dbscan } = require('numina-js');

// K-Means Clustering
const kMeansResult = kMeans(data, 3); // 3 clusters

// Hierarchical Clustering
const hierarchicalResult = hierarchicalClustering(data);

// DBSCAN
const dbscanResult = dbscan(data, 0.5, 5); // epsilon = 0.5, minPoints = 5
```

### Model Evaluation Metrics

```javascript
const { accuracy, precision, recall, f1Score, confusionMatrix, specificity, falsePositiveRate, trueNegativeRate, areaUnderROC, meanSquaredError, rootMeanSquaredError, meanAbsoluteError, rSquared } = require('numina-js');

// Classification metrics
const accuracyScore = accuracy(trueLabels, predictedLabels);
const precisionScore = precision(trueLabels, predictedLabels);
const recallScore = recall(trueLabels, predictedLabels);
const f1 = f1Score(trueLabels, predictedLabels);
const confMatrix = confusionMatrix(trueLabels, predictedLabels);
const specificityScore = specificity(trueLabels, predictedLabels);
const fpr = falsePositiveRate(trueLabels, predictedLabels);
const tnr = trueNegativeRate(trueLabels, predictedLabels);
const auc = areaUnderROC(trueLabels, predictedScores);

// Regression metrics
const mse = meanSquaredError(trueValues, predictedValues);
const rmse = rootMeanSquaredError(trueValues, predictedValues);
const mae = meanAbsoluteError(trueValues, predictedValues);
const r2 = rSquared(trueValues, predictedValues);
```

### Cross-Validation

```javascript
const { kFoldCrossValidation, stratifiedKFoldCrossValidation } = require('numina-js');

// K-Fold Cross-Validation
const kFoldResults = kFoldCrossValidation(data, labels, model, 5);

// Stratified K-Fold Cross-Validation
const stratifiedResults = stratifiedKFoldCrossValidation(data, labels, model, 5);
```

### Data Visualization

```javascript
const { plotGraph } = require('numina-js');

// Create a bar chart
const barChartData = {
  labels: ['January', 'February', 'March', 'April', 'May'],
  datasets: [{
    label: 'Sales',
    data: [12, 19, 3, 5, 2],
    backgroundColor: 'rgba(75, 192, 192, 0.6)'
  }]
};

plotGraph(800, 600, 'bar', barChartData, { title: { display: true, text: 'Monthly Sales' } }, 'sales_chart');
```

### Data Export

```javascript
const { exportToHTML, exportToJSON, exportToPDF } = require('numina-js');

exportToHTML(data, 'output.html');
exportToJSON(data, 'output.json');
exportToPDF(data, 'output.pdf');
```

## API Reference

### Data Preprocessing
- `handleMissingValues(data, column, method, specificValue)`
- `detectAndHandleOutliers(data, column, method)`
- `normalizeData(data, column, method)`
- `encodeCategorical(data, column, method)`
- `cleanData(data)`

### Feature Selection
- `selectFeatures(data, method, threshold, column1, column2)`
- `chiSquareTest(data, target)`
- `anovaFTest(data, target)`
- `lassoRegularization(data, target, alpha)`
- `ridgeRegularization(data, target, alpha)`

### Supervised Learning
- `linearRegression(data, target)`
- `logisticRegression(data, target, learningRate, iterations)`
- `svm(data, target, C, iterations, learningRate)`
- `decisionTrees(data, target)`
- `randomForests(data, target, numTrees)`
- `kNearestNeighbors(data, target, k)`
- `naiveBayes(data, target)`
- `gradientBoosting(data, target, numTrees, learningRate)`
- `adaBoost(data, target, numEstimators)`
- `votingClassifier(data, target, models)`

### Unsupervised Learning
- `kMeans(data, k, maxIterations)`
- `hierarchicalClustering(data)`
- `dbscan(data, epsilon, minPoints)`

### Model Evaluation Metrics
- `accuracy(trueLabels, predictedLabels)`
- `precision(trueLabels, predictedLabels)`
- `recall(trueLabels, predictedLabels)`
- `f1Score(trueLabels, predictedLabels)`
- `confusionMatrix(trueLabels, predictedLabels)`
- `specificity(trueLabels, predictedLabels)`
- `falsePositiveRate(trueLabels, predictedLabels)`
- `trueNegativeRate(trueLabels, predictedLabels)`
- `areaUnderROC(trueLabels, predictedScores)`
- `meanSquaredError(trueValues, predictedValues)`
- `rootMeanSquaredError(trueValues, predictedValues)`
- `meanAbsoluteError(trueValues, predictedValues)`
- `rSquared(trueValues, predictedValues)`

### Cross-Validation
- `kFoldCrossValidation(data, labels, model, k)`
- `stratifiedKFoldCrossValidation(data, labels, model, k)`

### Data Visualization
- `plotGraph(width, height, graphType, data, options, filename)`

### Data Export
- `exportToHTML(data, filePath)`
- `exportToJSON(data, filePath)`
- `exportToPDF(data, filePath)`

## Contributing

We welcome contributions to NuminaJS! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## License

NuminaJS is released under the MIT License. See the [LICENSE](LICENSE) file for details.
