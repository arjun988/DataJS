//./utils/supervisedAlgo.js
const _ = require('lodash');
const math = require('mathjs');
const jStat = require('jstat');

// Sigmoid function
const sigmoid = (z) => {
  return z.map(val => 1 / (1 + Math.exp(-val))); // Apply sigmoid element-wise
};

const svm = (data, target, C = 1, iterations = 1000, learningRate = 0.001) => {
    const x = data.map(row => row.slice(0, -1));  // Features
    const y = data.map(row => (row[target] === 1 ? 1 : -1)); // Convert labels to 1 or -1
  
    const X = math.concat(math.ones([x.length, 1]), x);  // Adding intercept
    let weights = math.zeros(X[0].length);  // Initialize weights as a zero array of correct length
  
    for (let i = 0; i < iterations; i++) {
      for (let j = 0; j < X.length; j++) {
        const margin = y[j] * math.dot(X[j], weights);  // Calculate margin (y * (X * weights))
  
        if (margin < 1) {
          // Misclassified, adjust weights
          const gradient = math.subtract(weights, math.multiply(y[j], X[j]));
          const regularized = math.multiply(gradient, learningRate);
          weights = math.subtract(weights, regularized);
        } else {
          // Correctly classified, just regularize
          weights = math.subtract(weights, math.multiply(weights, learningRate));
        }
      }
    }
  
    const predict = (newData) => {
      const Xnew = math.concat([1], newData); // Adding intercept
      const z = math.dot(Xnew, weights);
      return z >= 0 ? 1 : 0;  // Predict class 0 or 1
    };
  
    return { model: 'SVM', weights, predict };
  };
  
// Linear Regression
const linearRegression = (data, target) => {
  const x = data.map(row => row.slice(0, -1)); // Features
  const y = data.map(row => row[target]); // Target values
  
  const X = math.concat(math.ones(x.length, 1), x); // Adding intercept
  const beta = math.multiply(math.inv(math.multiply(math.transpose(X), X)), math.multiply(math.transpose(X), y));
  
  const predict = (newData) => {
    const Xnew = math.concat([1], newData); // Adding intercept
    return math.dot(Xnew, beta); // Predicting using the learned weights
  };
  
  return { model: 'Linear Regression', coefficients: beta, predict };
};

// Logistic Regression
const logisticRegression = (data, target, learningRate = 0.01, iterations = 1000) => {
  const x = data.map(row => row.slice(0, -1));  // Features
  const y = data.map(row => row[target]);       // Target values
  
  const X = math.concat(math.ones([x.length, 1]), x);  // Adding intercept
  let weights = math.zeros(X[0].length);  // Initialize weights as a zero array of correct length

  for (let i = 0; i < iterations; i++) {
    const z = math.multiply(X, weights);  // X * weights (dot product)
    const predictions = sigmoid(z);       // Apply sigmoid
    const errors = math.subtract(y, predictions);  // Errors between y and predictions

    // Update weights using gradient descent
    const gradient = math.multiply(math.transpose(X), errors);
    const weightUpdate = math.multiply(gradient, learningRate / X.length);  // Scaled by learning rate and number of samples
    weights = math.add(weights, weightUpdate);  // Update weights
  }

  const predict = (newData) => {
    const Xnew = math.concat([1], newData); // Adding intercept
    const z = math.dot(Xnew, weights);
    return sigmoid([z])[0] >= 0.5 ? 1 : 0; // Predict class 0 or 1
  };

  return { model: 'Logistic Regression', weights, predict };
};

// Decision Trees
const decisionTrees = (data, target) => {
  const buildTree = (data, depth = 0) => {
    const classes = _.uniq(data.map(row => row[target]));
    if (classes.length === 1) return { class: classes[0] };

    if (depth > 5) return { class: _.maxBy(classes, c => _.size(data.filter(row => row[target] === c))) };

    const bestSplit = findBestSplit(data, target);
    if (!bestSplit) return { class: _.maxBy(classes, c => _.size(data.filter(row => row[target] === c))) };

    const left = data.filter(row => row[bestSplit.feature] <= bestSplit.value);
    const right = data.filter(row => row[bestSplit.feature] > bestSplit.value);
    return {
      feature: bestSplit.feature,
      value: bestSplit.value,
      left: buildTree(left, depth + 1),
      right: buildTree(right, depth + 1)
    };
  };

  const findBestSplit = (data, target) => {
    let bestGain = -Infinity;
    let bestSplit = null;

    for (let feature = 0; feature < data[0].length - 1; feature++) {
      const values = _.uniq(data.map(row => row[feature]));

      values.forEach(value => {
        const left = data.filter(row => row[feature] <= value);
        const right = data.filter(row => row[feature] > value);

        const gain = calculateGain(data, left, right, target);
        if (gain > bestGain) {
          bestGain = gain;
          bestSplit = { feature, value };
        }
      });
    }
    return bestSplit;
  };

  const calculateGain = (data, left, right, target) => {
    const totalEntropy = entropy(data, target);
    const leftEntropy = entropy(left, target);
    const rightEntropy = entropy(right, target);

    const leftWeight = left.length / data.length;
    const rightWeight = right.length / data.length;
    const weightedEntropy = leftWeight * leftEntropy + rightWeight * rightEntropy;

    return totalEntropy - weightedEntropy;
  };

  const entropy = (data, target) => {
    const classes = _.uniq(data.map(row => row[target]));
    return -classes.reduce((acc, c) => {
      const p = _.size(data.filter(row => row[target] === c)) / data.length;
      return acc + (p ? p * Math.log2(p) : 0);
    }, 0);
  };

  const tree = buildTree(data);

  const predict = (newData, treeNode = tree) => {
    if (treeNode.class !== undefined) return treeNode.class;
    const featureValue = newData[treeNode.feature];
    return featureValue <= treeNode.value
      ? predict(newData, treeNode.left)
      : predict(newData, treeNode.right);
  };

  return { model: 'Decision Tree', tree, predict };
};

// Random Forests
const randomForests = (data, target, numTrees = 100) => {
  const trees = [];

  for (let i = 0; i < numTrees; i++) {
    const bootstrapSample = _.sampleSize(data, Math.floor(data.length * 0.8));
    const tree = decisionTrees(bootstrapSample, target);
    trees.push(tree);
  }

  const predict = (newData) => {
    const predictions = trees.map(tree => tree.predict(newData));
    return _.maxBy(predictions, p => _.size(predictions.filter(v => v === p)));
  };

  return { model: 'Random Forest', trees, predict };
};

// K-Nearest Neighbors
const kNearestNeighbors = (data, target, k = 3) => {
  const predict = (newPoint) => {
    const distances = data.map(row => {
      const dist = math.distance(row.slice(0, -1), newPoint);
      return { row, dist };
    });
    
    const neighbors = _.orderBy(distances, 'dist').slice(0, k);
    const votes = _.countBy(neighbors.map(neighbor => neighbor.row[target]));
    return _.maxBy(Object.keys(votes), key => votes[key]);
  };

  return { model: 'K-Nearest Neighbors', predict };
};

// Naive Bayes
const naiveBayes = (data, target) => {
  const total = data.length;
  const classCounts = _.countBy(data, row => row[target]);
  const probabilities = {};

  for (const className in classCounts) {
    probabilities[className] = {
      prior: classCounts[className] / total,
      likelihoods: {}
    };

    const classData = data.filter(row => row[target] === className);
    for (const feature in classData[0]) {
      if (feature !== target) {
        const values = classData.map(row => row[feature]);
        probabilities[className].likelihoods[feature] = _.countBy(values);
      }
    }
  }

  const predict = (newPoint) => {
    const classScores = {};

    for (const className in probabilities) {
      classScores[className] = probabilities[className].prior;
      for (const feature in newPoint) {
        if (feature !== target && probabilities[className].likelihoods[feature]) {
          const featureValueCount = probabilities[className].likelihoods[feature][newPoint[feature]] || 0;
          const likelihood = featureValueCount / classCounts[className];
          classScores[className] *= likelihood;
        }
      }
    }
    return _.maxBy(Object.keys(classScores), key => classScores[key]);
  };

  return { model: 'Naive Bayes', predict };
};

// General predict function
const generalPredict = (model, newData) => {
  if (model && typeof model.predict === 'function') {
    return model.predict(newData);
  } else {
    throw new Error("The model provided does not support prediction.");
  }
};
const gradientBoosting = (data, target, numTrees = 100, learningRate = 0.1) => {
    const x = data.map(row => row.slice(0, -1)); // Features
    const y = data.map(row => row[target]); // Target values

    const trees = [];
    
    // Initial prediction (mean value for regression)
    let predictions = Array(y.length).fill(_.mean(y)); // Start with mean for regression

    for (let i = 0; i < numTrees; i++) {
        // Compute residuals (the errors)
        const residuals = y.map((val, index) => val - predictions[index]);
        
        // Train a new decision tree on the residuals
        const tree = decisionTrees(data.map((row, index) => [...row.slice(0, -1), residuals[index]]), target);
        
        // Update predictions
        predictions = predictions.map((pred, index) => {
            const row = data[index]; // Get the current row for the prediction
            return pred + learningRate * tree.predict(row.slice(0, -1)); // Use current row for prediction
        });
        trees.push(tree);
    }

    const predict = (newData) => {
        let prediction = _.mean(y); // Start with mean for regression
        for (const tree of trees) {
            prediction += learningRate * tree.predict(newData);
        }
        return prediction;
    };

    return { model: 'Gradient Boosting', trees, predict };
};
// AdaBoost Implementation
const adaBoost = (data, target, numEstimators = 50) => {
  const weakClassifiers = [];
  const alphas = [];
  const weights = Array(data.length).fill(1 / data.length); // Initialize weights

  for (let i = 0; i < numEstimators; i++) {
      // Train a weak classifier (using Decision Tree with depth = 1)
      const tree = decisionTrees(data, target);
      weakClassifiers.push(tree);
      
      // Make predictions
      const predictions = data.map(row => tree.predict(row.slice(0, -1)));
      
      // Calculate error
      const errors = weights.reduce((sum, weight, index) => {
          return sum + (predictions[index] !== target[index] ? weight : 0);
      }, 0);
      
      // Calculate alpha
      const alpha = 0.5 * Math.log((1 - errors) / (errors + 1e-10));
      alphas.push(alpha);
      
      // Update weights
      weights.forEach((weight, index) => {
          weights[index] *= Math.exp(-alpha * (predictions[index] === target[index] ? 1 : -1));
      });
      
      // Normalize weights
      const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
      weights.forEach((weight, index) => {
          weights[index] /= totalWeight;
      });
  }

  const predict = (newData) => {
      const finalPredictions = weakClassifiers.map((tree, i) => {
          return alphas[i] * tree.predict(newData);
      });
      const finalScore = _.sum(finalPredictions);
      return finalScore >= 0 ? 1 : 0;
  };

  return { model: 'AdaBoost', weakClassifiers, alphas, predict };
};



// Voting Classifier/Regressor
const votingClassifier = (data, target, models) => {
  const predict = (newData) => {
      const votes = models.map(model => model.predict(newData));
      return _.maxBy(_.uniq(votes), key => _.size(_.filter(votes, v => v === key)));
  };

  return { model: 'Voting Classifier', models, predict };
};
// Exporting all models and general predict
module.exports = {
    svm,
  linearRegression,
  logisticRegression,
  decisionTrees,
  randomForests,
  kNearestNeighbors,
  naiveBayes,
  generalPredict,
  gradientBoosting,
  adaBoost,
  //stacking,
  votingClassifier
};
