const numeric = require('numeric');

// Helper function to sort eigenvalues and get indices
function argsort(arr) {
  const sortedIndices = [];
  let minIndex = 0;
  let currentIndex;

  for (let i = 0; i < arr.length; i++) {
    minIndex = i;
    for (let j = i + 1; j < arr.length; j++) {
      if (arr[j] > arr[minIndex]) {
        minIndex = j;
      }
    }

    currentIndex = sortedIndices.push(minIndex);
    if (currentIndex <= i) {
      break;
    }
  }

  return sortedIndices.reverse();
}

module.exports = {
  extractFeatures: (data) => {
    // Step 1: Center the data by subtracting the mean
    const mean = numeric.div(numeric.sum(data, 0), data.length);
    const centeredData = data.map(row => numeric.sub(row, mean));

    // Step 2: Calculate the covariance matrix
    const covarianceMatrix = numeric.mul(
      numeric.dot(numeric.transpose(centeredData), centeredData),
      1 / (data.length - 1)
    );

    // Step 3: Perform Eigen decomposition
    const eigen = numeric.eig(covarianceMatrix);
    const { E: eigenVectors, lambda: eigenValues } = eigen;

    // Step 4: Sort eigenvalues and eigenvectors
    const sortedIndices = argsort(eigenValues);
    const sortedEigenVectors = sortedIndices.map(i => eigenVectors[i]);

    // Step 5: Project data onto the first few principal components (e.g., first two)
    const numComponents = 2; // Adjust as needed
    const reducedData = numeric.dot(centeredData, sortedEigenVectors.slice(-numComponents));

    // Convert the result to a standard JavaScript array
    return reducedData.data;
  },
};
