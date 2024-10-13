const _ = require('lodash');
const ss = require('simple-statistics');
const jStat = require('jstat');
const { LassoRegression, RidgeRegression, ElasticNetRegression } = require('ml-regression');

module.exports = {
  // Select features based on correlation
  selectFeatures: (data, method = 'correlation', threshold = 0.5, column1 = null, column2 = null) => {
    const labelEncode = (columnData) => {
      const uniqueValues = _.uniq(columnData);
      const encodingMap = {};
      uniqueValues.forEach((value, index) => {
        encodingMap[value] = index;
      });
      return columnData.map(value => encodingMap[value]);
    };

    const isNumeric = (val) => !isNaN(parseFloat(val)) && isFinite(val);

    const prepareColumn = (columnName) => {
      const columnData = data.map(row => row[columnName]);
      if (isNumeric(columnData[0])) {
        return columnData; // Column is already numeric
      } else {
        console.log(`Label encoding applied to column: ${columnName}`);
        return labelEncode(columnData);
      }
    };

    const columns = Object.keys(data[0]);
    const processedColumns = {};
    columns.forEach(column => {
      processedColumns[column] = prepareColumn(column);
    });

    // If specific columns are provided, calculate correlation between them
    if (column1 && column2) {
      if (processedColumns[column1] && processedColumns[column2]) {
        const correlation = ss.sampleCorrelation(
          processedColumns[column1],
          processedColumns[column2]
        );
        console.log(`Correlation between ${column1} and ${column2}: ${correlation}`);
        return correlation;
      } else {
        console.error("Both columns must be convertible to numeric to calculate correlation.");
        return null;
      }
    }

    // Calculate correlations for all column pairs and build a correlation matrix
    const correlations = {};
    columns.forEach((col1, index) => {
      for (let i = index + 1; i < columns.length; i++) {
        const col2 = columns[i];
        try {
          const corr = ss.sampleCorrelation(
            processedColumns[col1],
            processedColumns[col2]
          );
          correlations[`${col1}-${col2}`] = corr;
          console.log(`Correlation between ${col1} and ${col2}: ${corr}`);
        } catch (e) {
          console.error(`Error calculating correlation between ${col1} and ${col2}: ${e.message}`);
        }
      }
    });

    return Object.keys(correlations).filter(key => Math.abs(correlations[key]) > threshold);
  },

  // CorrelationMatrix module to print the correlation matrix
  CorrelationMatrix: (data) => {
    const labelEncode = (columnData) => {
      const uniqueValues = _.uniq(columnData);
      const encodingMap = {};
      uniqueValues.forEach((value, index) => {
        encodingMap[value] = index;
      });
      return columnData.map(value => encodingMap[value]);
    };

    const isNumeric = (val) => !isNaN(parseFloat(val)) && isFinite(val);

    const prepareColumn = (columnName) => {
      const columnData = data.map(row => row[columnName]);
      if (isNumeric(columnData[0])) {
        return columnData; // Column is already numeric
      } else {
        return labelEncode(columnData); // Apply label encoding to categorical columns
      }
    };

    const columns = Object.keys(data[0]);
    const processedColumns = {};
    columns.forEach(column => {
      processedColumns[column] = prepareColumn(column);
    });

    const correlationMatrix = Array(columns.length).fill(null).map(() => Array(columns.length).fill(0));

    columns.forEach((col1, index1) => {
      columns.forEach((col2, index2) => {
        if (index1 === index2) {
          correlationMatrix[index1][index2] = 1; // Correlation with itself is 1
        } else if (index1 < index2) {
          try {
            const corr = ss.sampleCorrelation(
              processedColumns[col1],
              processedColumns[col2]
            );
            correlationMatrix[index1][index2] = corr;
            correlationMatrix[index2][index1] = corr; // Fill both upper and lower triangles
          } catch (e) {
            console.error(`Error calculating correlation between ${col1} and ${col2}: ${e.message}`);
          }
        }
      });
    });

    // Print the correlation matrix
    console.log('\nCorrelation Matrix:');
    console.log('    ', columns.map(col => col.padEnd(10)).join(' '));
    correlationMatrix.forEach((row, i) => {
      console.log(columns[i].padEnd(5), row.map(val => val.toFixed(3).padEnd(10)).join(' '));
    });

    return correlationMatrix;
  },

  chiSquareTest: (data, target) => {
    const chiSquareValues = {};
    Object.keys(data[0]).forEach((col) => {
      const columnData = data.map(row => row[col]);
      const contingencyTable = _.countBy(target, (t, i) => `${t}-${columnData[i]}`);
      const observed = Object.values(contingencyTable);
      const total = _.sum(observed);
      const expected = observed.map(o => total / observed.length);
      
      // Calculate Chi-Square statistic manually
      const chiSq = observed.reduce((sum, o, i) => {
        const e = expected[i] || 0;
        return sum + ((o - e) ** 2) / e;
      }, 0);

      const pValue = jStat.chisquare.cdf(chiSq, observed.length - 1);
      chiSquareValues[col] = { chiSquare: chiSq, pValue };
      console.log(`Chi-Square value for ${col}: ${chiSq}, p-value: ${pValue}`);
    });
    return chiSquareValues;
  },

  // ANOVA F-Test for continuous features with categorical target
 // ANOVA F-Test for continuous features with categorical target
anovaFTest: (data, target) => {
  const anovaResults = {};
  const uniqueTargets = _.uniq(target); // Unique target categories

  // Group data by target category
  uniqueTargets.forEach((uniqueTarget) => {
    const groupData = data.filter((row, index) => target[index] === uniqueTarget);
    anovaResults[uniqueTarget] = Object.keys(groupData[0]).map(col => {
      return groupData.map(row => row[col]);
    }).filter(arr => arr.length > 0); // Keep only non-empty arrays
  });

  // Perform ANOVA
  Object.keys(anovaResults).forEach((group) => {
    try {
      const fScore = jStat.anovafscore(anovaResults[group]); // Get F-statistic
      const pValue = jStat.anovaftest(...anovaResults[group]); // Get p-value
      console.log(`ANOVA F-value for group ${group}: ${fScore}, p-value: ${pValue}`);
      anovaResults[group] = { fScore, pValue };
    } catch (e) {
      console.error(`Error calculating ANOVA for group ${group}: ${e.message}`);
    }
  });

  return anovaResults;
},


  // Variance Threshold for removing low-variance features
  varianceThreshold: (data, threshold = 0.1) => {
    const lowVarianceFeatures = {};
    Object.keys(data[0]).forEach((col) => {
      const columnData = data.map(row => row[col]);
      const columnVariance = jStat.variance(columnData);
      if (columnVariance > threshold) {
        lowVarianceFeatures[col] = columnVariance;
        console.log(`Variance for ${col}: ${columnVariance}`);
      }
    });
    return lowVarianceFeatures;
  },

  // Lasso Regularization (L1) for feature selection
  lassoRegularization: (data, target, alpha = 0.1) => {
    const features = Object.keys(data[0]);
    const X = data.map(row => Object.values(row));
    const y = target;
    const lasso = new LassoRegression(alpha);
    lasso.fit(X, y);
    const selectedFeatures = features.filter((_, i) => lasso.weights[i] !== 0);
    console.log(`Lasso selected features: ${selectedFeatures}`);
    return selectedFeatures;
  },

  // Ridge Regularization (L2) for feature selection
  ridgeRegularization: (data, target, alpha = 1.0) => {
    const features = Object.keys(data[0]);
    const X = data.map(row => Object.values(row));
    const y = target;
    const ridge = new RidgeRegression(alpha);
    ridge.fit(X, y);
    console.log(`Ridge selected weights: ${ridge.weights}`);
    return features;
  },

  // Elastic Net (L1 + L2 Regularization)
  elasticNetRegularization: (data, target, alpha = 0.1, l1Ratio = 0.5) => {
    const features = Object.keys(data[0]);
    const X = data.map(row => Object.values(row));
    const y = target;
    const elasticNet = new ElasticNetRegression(alpha, l1Ratio);
    elasticNet.fit(X, y);
    const selectedFeatures = features.filter((_, i) => elasticNet.weights[i] !== 0);
    console.log(`Elastic Net selected features: ${selectedFeatures}`);
    return selectedFeatures;
  },

  // T-Test for binary target variable
  tTest: (data, target) => {
    const tTestResults = {};
    const group1 = data.filter((_, i) => target[i] === 0);
    const group2 = data.filter((_, i) => target[i] === 1);
    Object.keys(data[0]).forEach((col) => {
      const group1Values = group1.map(row => row[col]);
      const group2Values = group2.map(row => row[col]);
      const tValue = jStat.ttest(group1Values, group2Values, 2);
      tTestResults[col] = tValue;
      console.log(`T-Test result for ${col}: ${tValue}`);
    });
    return tTestResults;
  },

  // Information Gain for categorical target variable
  informationGain: (columnData, target) => {
    if (!Array.isArray(columnData) || !Array.isArray(target)) {
        throw new Error("Both columnData and target must be arrays.");
    }
  
    const baseEntropy = calculateEntropy(target);

    const uniqueValues = _.uniq(columnData);
    const subsetEntropy = uniqueValues.map((value) => {
        const subset = target.filter((_, index) => columnData[index] === value);
        return (subset.length / target.length) * calculateEntropy(subset);
    });

    const totalSubsetEntropy = subsetEntropy.reduce((sum, ent) => sum + ent, 0);
    const gain = baseEntropy - totalSubsetEntropy;

    return gain;
},

  

  // Wrapper for multiple methods
  featureSelection: (data, target, methods = ['correlation', 'chiSquare', 'anova', 'lasso', 'ridge', 'elasticNet']) => {
    const results = {};
    if (methods.includes('correlation')) {
      results.correlation = this.selectFeatures(data);
    }
    if (methods.includes('chiSquare')) {
      results.chiSquare = this.chiSquareTest(data, target);
    }
    if (methods.includes('anova')) {
      results.anova = this.anovaFTest(data, target);
    }
    if (methods.includes('lasso')) {
      results.lasso = this.lassoRegularization(data, target);
    }
    if (methods.includes('ridge')) {
      results.ridge = this.ridgeRegularization(data, target);
    }
    if (methods.includes('elasticNet')) {
      results.elasticNet = this.elasticNetRegularization(data, target);
    }
    return results;
  },
};

const calculateEntropy = (data) => {
  const total = data.length;
  const frequencyMap = {};

  // Count the frequency of each class label
  data.forEach((value) => {
      if (frequencyMap[value]) {
          frequencyMap[value]++;
      } else {
          frequencyMap[value] = 1;
      }
  });

  // Calculate entropy
  let entropy = 0;
  Object.values(frequencyMap).forEach((count) => {
      const probability = count / total;
      entropy -= probability * Math.log2(probability);
  });

  return entropy;
};


