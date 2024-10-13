module.exports = {
    transformData: (data, column, method = 'log') => {
      data.forEach(row => {
        if (method === 'log' && row[column] > 0) {
          row[column] = Math.log(row[column]);
        } else if (method === 'sqrt') {
          row[column] = Math.sqrt(row[column]);
        }
      });
      return data;
    },
  };
  