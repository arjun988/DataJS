const _ = require('lodash');
const math = require('mathjs');
const jStat = require('jstat');

module.exports = {
  handleMissingValues: (data, column, method = 'mean', specificValue = null) => {
    
    if (!_.isNumber(data[0][column])) {
      // Non-numeric column
      data.forEach(row => {
        if (row[column] === null) {
          row[column] = specificValue !== null ? specificValue : '';
        }
      });
    } else {
      // Numeric column
      if (method === 'mean') {
        const meanValue = Number(_.round(_.mean(data.map(row => row[column])), 2));
        data.forEach(row => {
          if (row[column] === null) {
            row[column] = meanValue;
          }
        });
      } else if (method === 'median') {
        try {
          const medianValue = math.median(data.map(row => Number(row[column])));
          data.forEach(row => {
            if (row[column] === null) {
              row[column] = medianValue;
            }
          });
        } catch (e) {
          console.warn(`Error calculating median for ${column}: ${e.message}`);
          // Fallback to mean method
          const meanValue = Number(_.round(_.mean(data.map(row => row[column])), 2));
          data.forEach(row => {
            if (row[column] === null) {
              row[column] = meanValue;
            }
          });
        }
      } else if (specificValue !== null) {
        data.forEach(row => {
          if (row[column] === null) {
            row[column] = specificValue;
          }
        });
      }
    }
    return data;
  },

  
  detectAndHandleOutliers: (data, column, method = 'remove') => {
    const values = data.map(row => row[column]);
    const sortedValues = _.sortBy(values);
  
    const q1 = sortedValues[Math.floor(sortedValues.length * 0.25)];
    const q3 = sortedValues[Math.floor(sortedValues.length * 0.75)];
    const iqr = q3 - q1;
  
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;
  
    if (method === 'remove') {
      return data.filter(row => row[column] >= lowerBound && row[column] <= upperBound);
    } else if (method === 'impute') {
      const meanValue = _.mean(values.filter(val => val >= lowerBound && val <= upperBound));
      data.forEach(row => {
        if (row[column] < lowerBound || row[column] > upperBound) {
          row[column] = meanValue;
        }
      });
    }
    return data;
  },
  
  normalizeData: (data, column, method = 'min-max') => {
    if (method === 'min-max') {
      const min = _.min(data.map(row => row[column]));
      const max = _.max(data.map(row => row[column]));
      data.forEach(row => {
        row[column] = (row[column] === null) ? null : ((row[column] - min) / (max - min));
      });
    } else if (method === 'standard') {
      const values = data.map(row => row[column]);
      const filteredValues = values.filter(val => val !== null);
      
      if (filteredValues.length === 0) {
        console.warn("No valid numeric values found for standard normalization.");
        return data; // Return original data if no valid values
      }
  
      const mean = _.mean(filteredValues);
      const stdDev = math.std(filteredValues);
  
      data.forEach(row => {
        row[column] = (row[column] === null) ? null : ((row[column] - mean) / stdDev);
      });
    }
    return data;
  },
  
  
  

  encodeCategorical: (data, column, method = 'one-hot') => {
    if (method === 'one-hot') {
      const uniqueValues = _.uniq(data.map(row => row[column]));
      uniqueValues.forEach(value => {
        data.forEach(row => {
          row[`${column}_${value}`] = row[column] === value ? 1 : 0;
        });
      });
    } else if (method === 'label') {
      const uniqueValues = _.uniq(data.map(row => row[column]));
      const valueToLabel = {};
      uniqueValues.forEach((value, index) => {
        valueToLabel[value] = index;
      });
      data.forEach(row => {
        row[column] = valueToLabel[row[column]];
      });
    }
    return data;
  },

  cleanData: (data) => {
    // Remove duplicates
    data = _.uniqBy(data, JSON.stringify);
    // Handling inconsistent entries can be implemented here as needed
    return data;
  },

  getDimensions: (data) => {
    const numRows = data.length;
    const numCols = Object.keys(data[0]).length;
    return { rows: numRows, columns: numCols };
  },

  // New Feature: Print all column names
  getColumnNames: (data) => {
    return Object.keys(data[0]);
  },

  getNullValues: (data) => {
    const nullCounts = {};

    Object.keys(data[0]).forEach(column => {
      const nullCount = data.filter(row => row[column] === null||'').length;
      nullCounts[column] = nullCount;
    });

    // Print the null counts
    Object.entries(nullCounts).forEach(([column, count]) => {
      console.log(`${column}: ${count} null values`);
    });

    return nullCounts; // Return the result if needed
  },

  deleteColumn: (data, column) => {
    data.forEach(row => {
      delete row[column];
    });
    return data;
  },

  modifyColumnName: (data, oldName, newName) => {
    data.forEach(row => {
      row[newName] = row[oldName];
      delete row[oldName];
    });
    return data;
  },

  // New Feature: Convert data types
  convertDataType: (data, column, newType) => {
    data.forEach(row => {
      if (newType === 'number') {
        row[column] = Number(row[column]);
      } else if (newType === 'string') {
        row[column] = String(row[column]);
      } else if (newType === 'boolean') {
        row[column] = Boolean(row[column]);
      }
    });
    return data;
  },

  // New Feature: Data Splitting
  splitData: (data, trainSize = 0.7, validationSize = 0.15) => {
    const totalSize = data.length;
    const trainEnd = Math.floor(trainSize * totalSize);
    const validationEnd = trainEnd + Math.floor(validationSize * totalSize);

    const trainingData = data.slice(0, trainEnd);
    const validationData = data.slice(trainEnd, validationEnd);
    const testingData = data.slice(validationEnd);

    return { trainingData, validationData, testingData };
  },

  slice: (data, start, end) => {
    return data.slice(start, end);
  },

  transpose: (data) => {
    if (!Array.isArray(data) || !Array.isArray(data[0])) {
      throw new TypeError('Input data must be a 2D array.');
    }
    return data[0].map((_, colIndex) => data.map(row => row[colIndex]));
  },
  

  sum: (data, column) => {
    return _.sum(data.map(row => row[column] || 0));
  },

  deviation: (data, column) => {
    const mean = _.mean(data.map(row => row[column] || 0));
    return data.map(row => (row[column] || 0) - mean);
  },

  product: (data, column) => {
    return data.reduce((acc, row) => acc * (row[column] || 1), 1);
  },

  sumRow: (row) => {
    return _.sum(Object.values(row));
  },

  variance: (data, column) => {
    const values = data.map(row => row[column] || 0);
    return jStat.variance(values);
  },

  kurtosis: (data, column) => {
    const values = data.map(row => row[column] || 0);
    return jStat.kurtosis(values);
  },

  skewness: (data, column) => {
    const values = data.map(row => row[column] || 0);
    return jStat.skewness(values);
  },

  covariance: (data, columnX, columnY) => {
    const xValues = data.map(row => row[columnX] || 0);
    const yValues = data.map(row => row[columnY] || 0);
    return jStat.covariance(xValues, yValues);
  },

  stdev: (data, column) => {
    const values = data.map(row => row[column] || 0);
    return jStat.stdev(values);
  },
};
