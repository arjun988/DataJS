const fs = require('fs');
const path = require('path');
const { createCanvas } = require('canvas');
const Chart = require('chart.js/auto');
const logger = console;

// Function to ensure directory exists
function ensureDirectoryExists(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

// Create the Charts directory if it doesn't exist
ensureDirectoryExists(path.join(__dirname, '..', 'Charts'));

function createChartInstance(canvas, type, data, options = {}) {
  try {
    return new Chart(canvas, {
      type: type,
      data: data,
      options: options
    });
  } catch (error) {
    logger.error('Error creating chart instance:', error);
    throw error;
  }
}

async function saveChartAsImage(chart, filename) {
  try {
    const base64Img = chart.toBase64Image();
    const buffer = Buffer.from(base64Img.split(',')[1], 'base64');
    
    return new Promise((resolve, reject) => {
      const filePath = path.join(__dirname, '..', 'Charts', `${filename}.png`);
      fs.writeFile(filePath, buffer, (err) => {
        if (err) {
          logger.error('Error saving chart image:', err);
          reject(err);
        } else {
          resolve();
        }
      });
    });
  } catch (error) {
    logger.error('Error converting chart to image:', error);
    throw error;
  }
}

const defaultOptions = {
  responsive: true,
  plugins: {
    legend: {
      position: 'top',
      labels: {
        font: {
          size: 16,
          family: "'Arial', sans-serif"
        },
        color: '#333'
      },
      title: {
        display: true,
        text: 'Legend',
        font: {
          size: 18,
          family: "'Arial', sans-serif"
        },
        color: '#333'
      }
    },
    tooltip: {
      enabled: true,
      backgroundColor: 'rgba(127, 237, 107,0.95)',
      borderColor: 'rgb(33, 37, 41)',
      borderWidth: 1,
      cornerRadius: 5,
      padding: 10,
      caretPadding: 10,
      callbacks: {
        label: (context) => {
          let label = context.dataset.label || '';
          if (label) {
            label += ': ';
          }
          if (context.parsed.y !== null) {
            label += new Intl.NumberFormat('en-US', { style: 'decimal' }).format(context.parsed.y);
          }
          return label;
        }
      }
    }
  },
  scales: {
    y: {
      beginAtZero: true,
      grid: {
        drawOnChartArea: false,
        color: 'rgba(54, 162, 235, 0.2)'
      },
      ticks: {
        callback: (value) => {
          return new Intl.NumberFormat('en-US', { style: 'decimal' }).format(value);
        }
      }
    },
    x: {
      grid: {
        drawOnChartArea: false,
        color: 'rgba(54, 162, 235, 0.2)'
      }
    }
  }
};

async function plotGraph(width = 800, height = 600, graphType, data, options = {}, filename = null) {
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');

  try {
    const chartOptions = { ...defaultOptions, ...options };
    const chart = createChartInstance(ctx, graphType, data, chartOptions);
    
    const fileName = filename || `${graphType}_${Date.now()}`;
    await saveChartAsImage(chart, fileName);
    
    logger.info(`Chart saved as ${fileName}.png in the Charts folder`);
  } catch (error) {
    logger.error('Error creating or saving chart:', error);
  }
}

module.exports = {
  plotGraph,
};
