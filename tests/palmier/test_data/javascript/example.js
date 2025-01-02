import { someFunction } from './utils';
import DefaultClass from './default';

/**
 * A class that processes data
 */
class DataProcessor {
    /**
     * Create a new processor
     * @param {string} name - The processor name
     */
    constructor(name) {
        this.name = name;
    }
    
    /**
     * Process the input data
     * @param {Array} data - Data to process
     * @returns {Object} Processed data
     */
    processData(data) {
        const result = {};
        data.forEach(item => {
            result[item] = item.length;
        });
        return result;
    }
    
    /**
     * Get the processor name
     * @returns {string} The name
     */
    getName() {
        return this.name;
    }
}

export { DataProcessor };
export default DataProcessor; 