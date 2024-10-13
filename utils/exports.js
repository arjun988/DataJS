const { jsPDF } = require('jspdf'); // Updated import
const fs = require('fs');

module.exports = {
  exportToHTML: (data, filePath) => {
    const htmlContent = `<html><body>${JSON.stringify(data)}</body></html>`;
    fs.writeFileSync(filePath, htmlContent);
  },

  exportToJSON: (data, filePath) => {
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
  },

  exportToPDF: (data, filePath) => {
    const doc = new jsPDF(); // Correctly creating an instance
    doc.text(JSON.stringify(data), 10, 10);
    doc.save(filePath);
  },
};
