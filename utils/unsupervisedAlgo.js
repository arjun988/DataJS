const _ = require('lodash');
const math = require('mathjs');
//const PCA=require('pca-js');
// K-Means Clustering
const kMeans = (data, k, maxIterations = 100) => {
  const centroids = _.sampleSize(data, k);
  let assignments = new Array(data.length).fill(0);
  
  for (let iter = 0; iter < maxIterations; iter++) {
    assignments = data.map(point => {
      const distances = centroids.map(centroid => math.distance(point, centroid));
      return distances.indexOf(Math.min(...distances));
    });

    const newCentroids = [];
    for (let i = 0; i < k; i++) {
      const clusterPoints = data.filter((_, index) => assignments[index] === i);
      if (clusterPoints.length > 0) {
        const newCentroid = math.mean(clusterPoints, 0);
        newCentroids.push(newCentroid);
      } else {
        newCentroids.push(centroids[i]);
      }
    }

    if (_.isEqual(newCentroids, centroids)) {
      break;
    }
    centroids.splice(0, centroids.length, ...newCentroids);
  }

  return { centroids, assignments };
};
// Hierarchical Clustering
const hierarchicalClustering = (data) => {
    const distanceMatrix = [];
    const n = data.length;

    // Create distance matrix
    for (let i = 0; i < n; i++) {
        distanceMatrix[i] = [];
        for (let j = 0; j < n; j++) {
            distanceMatrix[i][j] = i === j ? 0 : math.distance(data[i], data[j]);
        }
    }

    // Initialize clusters
    let clusters = data.map((value, index) => [index]);

    while (clusters.length > 1) {
        let minDistance = Infinity;
        let closestPair = [-1, -1];

        // Find the closest clusters
        for (let i = 0; i < clusters.length; i++) {
            for (let j = i + 1; j < clusters.length; j++) {
                const distance = distanceMatrix[clusters[i][0]][clusters[j][0]];
                if (distance < minDistance) {
                    minDistance = distance;
                    closestPair = [i, j];
                }
            }
        }

        // Merge the closest clusters
        const [index1, index2] = closestPair;
        const mergedCluster = [...clusters[index1], ...clusters[index2]];
        clusters = clusters.filter((_, index) => index !== index1 && index !== index2);
        clusters.push(mergedCluster);
    }

    return clusters[0]; // Return the final merged cluster
};

// DBSCAN
const dbscan = (data, epsilon, minPoints) => {
    const visited = new Array(data.length).fill(false);
    const clusters = [];
    const noise = [];

    const regionQuery = (pointIndex) => {
        const neighbors = [];
        for (let i = 0; i < data.length; i++) {
            if (math.distance(data[pointIndex], data[i]) <= epsilon) {
                neighbors.push(i);
            }
        }
        return neighbors;
    };

    const expandCluster = (pointIndex, neighbors) => {
        const cluster = [pointIndex];
        for (const neighbor of neighbors) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                const neighborNeighbors = regionQuery(neighbor);
                if (neighborNeighbors.length >= minPoints) {
                    neighbors.push(...neighborNeighbors);
                }
            }
            if (!cluster.includes(neighbor)) {
                cluster.push(neighbor);
            }
        }
        return cluster;
    };

    for (let i = 0; i < data.length; i++) {
        if (!visited[i]) {
            visited[i] = true;
            const neighbors = regionQuery(i);
            if (neighbors.length < minPoints) {
                noise.push(i); // Mark as noise
            } else {
                const cluster = expandCluster(i, neighbors);
                clusters.push(cluster);
            }
        }
    }

    return { clusters, noise };
};

// General predict function
const generalPredict = (model, newData) => {
  if (model && typeof model.predict === 'function') {
    return model.predict(newData);
  } else {
    throw new Error("The model provided does not support prediction.");
  }
};

// Exporting all models
module.exports = {
  kMeans,
  
  hierarchicalClustering,
    dbscan,
  generalPredict
};
