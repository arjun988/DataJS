const { SMOTE } = require('smote');
const _ = require('lodash');

module.exports = {
  handleImbalance: (data, method = 'oversample') => {
    const majorityClass = _.maxBy(data, 'target').target; // Assuming 'target' is your label
    const minorityClassData = data.filter(row => row.target !== majorityClass);
    const majorityClassData = data.filter(row => row.target === majorityClass);

    if (method === 'oversample') {
      while (minorityClassData.length < majorityClassData.length) {
        const newSample = _.sample(majorityClassData)[0];
        minorityClassData.push({ ...newSample });
      }
      return [...majorityClassData, ...minorityClassData];
    } else if (method === 'smote') {
      return SMOTE(data, { target: 'target' });
    }
    
    // Implement other imbalance handling methods
    return data; // Return original data if no method matches
  },
};
