import React, { useState, useEffect } from 'react';

// Interface remains the same
interface DataProcessor {
    name: string;
    processData(data: string[]): Record<string, number>;
    getName(): string;
}

/**
 * React component that implements DataProcessor
 */
const DataProcessorComponent: React.FC<{ initialName: string }> = ({ initialName }) => {
    // State management with hooks
    const [name, setName] = useState(initialName);
    const [data, setData] = useState<string[]>([]);
    const [results, setResults] = useState<Record<string, number>>({});

    /**
     * Process the input data
     */
    const processData = (inputData: string[]): Record<string, number> => {
        return inputData.reduce((acc, item) => {
            acc[item] = item.length;
            return acc;
        }, {} as Record<string, number>);
    };

    // Effect hook to process data when it changes
    useEffect(() => {
        const processed = processData(data);
        setResults(processed);
    }, [data]);

    return (
        <div className="data-processor">
            <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
            />
            <button onClick={() => setData([...data, 'new item'])}>
                Add Item
            </button>
            <ul>
                {Object.entries(results).map(([key, value]) => (
                    <li key={key}>
                        {key}: {value}
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default DataProcessorComponent;
