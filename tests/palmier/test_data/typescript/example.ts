/**
 * Interface for data processing
 */
interface DataProcessor {
    name: string;
    processData(data: string[]): Record<string, number>;
    getName(): string;
}

/**
 * Implementation of DataProcessor
 */
class DataProcessorImpl implements DataProcessor {
    name: string;

    constructor(name: string) {
        this.name = name;
    }
    
    /**
     * Process the input data
     * @param data Data to process
     * @returns Processed data
     */
    processData(data: string[]): Record<string, number> {
        return data.reduce((acc, item) => {
            acc[item] = item.length;
            return acc;
        }, {} as Record<string, number>);
    }
    
    /**
     * Get the processor name
     */
    getName(): string {
        return this.name;
    }
} 