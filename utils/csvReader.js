const fs = require('fs');
const Papa = require('papaparse');

module.exports = {
  readCSV: (filePath) => {
    try {
      const file = fs.readFileSync(filePath, 'utf8');
      const parsedData = Papa.parse(file, {
        header: true,
        skipEmptyLines: true,  // Skip empty lines
        dynamicTyping: true,   // Automatically typecast values (numbers, booleans)
        transformHeader: (header) => header.trim(), // Trim whitespace from headers
        complete: (result) => {
          if (result.errors.length > 0) {
            console.warn('Parsing errors occurred:', result.errors);
          }
          return result.data;
        }
      });
      
      console.log("Parsed CSV Data:", parsedData);
      return parsedData.data; // Returning only the data part of the result
    } catch (error) {
      console.error(`Error reading or parsing CSV file: ${error.message}`);
      return []; // Return an empty array if parsing fails
    }
  },
};
