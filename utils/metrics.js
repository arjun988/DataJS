const math = require('mathjs');

// Classification Metrics
const accuracy = (trueLabels, predictedLabels) => {
    const correct = trueLabels.reduce((acc, label, index) => acc + (label === predictedLabels[index] ? 1 : 0), 0);
    return correct / trueLabels.length;
};

const precision = (trueLabels, predictedLabels) => {
    const truePositives = trueLabels.reduce((acc, label, index) => acc + (label === 1 && predictedLabels[index] === 1 ? 1 : 0), 0);
    const falsePositives = trueLabels.reduce((acc, label, index) => acc + (label === 0 && predictedLabels[index] === 1 ? 1 : 0), 0);
    return truePositives / (truePositives + falsePositives) || 0;
};

const recall = (trueLabels, predictedLabels) => {
    const truePositives = trueLabels.reduce((acc, label, index) => acc + (label === 1 && predictedLabels[index] === 1 ? 1 : 0), 0);
    const falseNegatives = trueLabels.reduce((acc, label, index) => acc + (label === 1 && predictedLabels[index] === 0 ? 1 : 0), 0);
    return truePositives / (truePositives + falseNegatives) || 0;
};

const f1Score = (trueLabels, predictedLabels) => {
    const prec = precision(trueLabels, predictedLabels);
    const rec = recall(trueLabels, predictedLabels);
    return (2 * prec * rec) / (prec + rec) || 0;
};

const confusionMatrix = (trueLabels, predictedLabels) => {
    const TP = trueLabels.reduce((acc, label, index) => acc + (label === 1 && predictedLabels[index] === 1 ? 1 : 0), 0);
    const TN = trueLabels.reduce((acc, label, index) => acc + (label === 0 && predictedLabels[index] === 0 ? 1 : 0), 0);
    const FP = trueLabels.reduce((acc, label, index) => acc + (label === 0 && predictedLabels[index] === 1 ? 1 : 0), 0);
    const FN = trueLabels.reduce((acc, label, index) => acc + (label === 1 && predictedLabels[index] === 0 ? 1 : 0), 0);
    
    return { TP, TN, FP, FN };
};

const specificity = (trueLabels, predictedLabels) => {
    const { TN, FP } = confusionMatrix(trueLabels, predictedLabels);
    return TN / (TN + FP) || 0;
};

const falsePositiveRate = (trueLabels, predictedLabels) => {
    const { FP, TN } = confusionMatrix(trueLabels, predictedLabels);
    return FP / (FP + TN) || 0;
};

const trueNegativeRate = (trueLabels, predictedLabels) => {
    return specificity(trueLabels, predictedLabels);
};

const areaUnderROC = (trueLabels, predictedScores) => {
    const sortedIndices = Array.from(Array(trueLabels.length).keys()).sort((a, b) => predictedScores[b] - predictedScores[a]);
    let TP = 0, FP = 0, totalPositive = trueLabels.reduce((acc, label) => acc + (label === 1 ? 1 : 0), 0);
    const TPR = []; // True Positive Rate
    const FPR = []; // False Positive Rate
    
    for (let i = 0; i < sortedIndices.length; i++) {
        const idx = sortedIndices[i];
        if (trueLabels[idx] === 1) {
            TP++;
        } else {
            FP++;
        }
        TPR.push(TP / totalPositive);
        FPR.push(FP / (FP + totalPositive - TP));
    }
    
    // Calculate AUC using the trapezoidal rule
    let auc = 0;
    for (let i = 1; i < TPR.length; i++) {
        auc += (FPR[i] - FPR[i - 1]) * (TPR[i] + TPR[i - 1]) / 2;
    }
    
    return auc;
};

// Cross-Validation Functions
const kFoldCrossValidation = (data, labels, model, k) => {
    const foldSize = Math.floor(data.length / k);
    let metrics = {
        accuracy: [],
        precision: [],
        recall: [],
        f1Score: []
    };

    for (let i = 0; i < k; i++) {
        const testData = data.slice(i * foldSize, (i + 1) * foldSize);
        const testLabels = labels.slice(i * foldSize, (i + 1) * foldSize);
        const trainData = [...data.slice(0, i * foldSize), ...data.slice((i + 1) * foldSize)];
        const trainLabels = [...labels.slice(0, i * foldSize), ...labels.slice((i + 1) * foldSize)];

        const predictions = model.train(trainData, trainLabels).predict(testData);
        metrics.accuracy.push(accuracy(testLabels, predictions));
        metrics.precision.push(precision(testLabels, predictions));
        metrics.recall.push(recall(testLabels, predictions));
        metrics.f1Score.push(f1Score(testLabels, predictions));
    }

    return {
        accuracy: math.mean(metrics.accuracy),
        precision: math.mean(metrics.precision),
        recall: math.mean(metrics.recall),
        f1Score: math.mean(metrics.f1Score)
    };
};

const stratifiedKFoldCrossValidation = (data, labels, model, k) => {
    const stratifiedData = {};
    const uniqueLabels = [...new Set(labels)];

    uniqueLabels.forEach(label => {
        stratifiedData[label] = data.filter((_, index) => labels[index] === label);
    });

    const stratifiedFolds = uniqueLabels.map(label => {
        const foldSize = Math.floor(stratifiedData[label].length / k);
        return Array.from({ length: k }, (_, i) => stratifiedData[label].slice(i * foldSize, (i + 1) * foldSize));
    });

    let metrics = {
        accuracy: [],
        precision: [],
        recall: [],
        f1Score: []
    };

    for (let i = 0; i < k; i++) {
        const testData = [];
        const testLabels = [];

        uniqueLabels.forEach(label => {
            testData.push(...stratifiedFolds[label][i]);
            testLabels.push(...Array(stratifiedFolds[label][i].length).fill(label));
        });

        const trainData = data.filter((_, index) => !testData.includes(data[index]));
        const trainLabels = labels.filter((_, index) => !testLabels.includes(labels[index]));

        const predictions = model.train(trainData, trainLabels).predict(testData);
        metrics.accuracy.push(accuracy(testLabels, predictions));
        metrics.precision.push(precision(testLabels, predictions));
        metrics.recall.push(recall(testLabels, predictions));
        metrics.f1Score.push(f1Score(testLabels, predictions));
    }

    return {
        accuracy: math.mean(metrics.accuracy),
        precision: math.mean(metrics.precision),
        recall: math.mean(metrics.recall),
        f1Score: math.mean(metrics.f1Score)
    };
};

// Regression Metrics
const meanSquaredError = (trueValues, predictedValues) => {
    return math.mean(trueValues.map((value, index) => Math.pow(value - predictedValues[index], 2)));
};

const rootMeanSquaredError = (trueValues, predictedValues) => {
    return Math.sqrt(meanSquaredError(trueValues, predictedValues));
};

const meanAbsoluteError = (trueValues, predictedValues) => {
    return math.mean(trueValues.map((value, index) => Math.abs(value - predictedValues[index])));
};

const rSquared = (trueValues, predictedValues) => {
    const ssTot = trueValues.reduce((acc, value) => acc + Math.pow(value - math.mean(trueValues), 2), 0);
    const ssRes = trueValues.reduce((acc, value, index) => acc + Math.pow(value - predictedValues[index], 2), 0);
    return 1 - (ssRes / ssTot);
};

// Exporting metrics
module.exports = {
    accuracy,
    precision,
    recall,
    f1Score,
    confusionMatrix,
    specificity,
    falsePositiveRate,
    trueNegativeRate,
    areaUnderROC,
    meanSquaredError,
    rootMeanSquaredError,
    meanAbsoluteError,
    rSquared,
    kFoldCrossValidation,
    stratifiedKFoldCrossValidation,
};
