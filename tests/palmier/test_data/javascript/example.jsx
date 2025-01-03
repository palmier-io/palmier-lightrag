import React, { useState, useEffect } from 'react';

/**
 * React component for processing data
 */
const DataProcessorComponent = ({ initialName }) => {
    // State management with hooks
    const [name, setName] = useState(initialName);
    const [data, setData] = useState([]);
    const [results, setResults] = useState({});

    /**
     * Process the input data
     */
    const processData = (inputData) => {
        return inputData.reduce((acc, item) => {
            acc[item] = item.length;
            return acc;
        }, {});
    };

    // Effect hook to process data when it changes
    useEffect(() => {
        const processed = processData(data);
        setResults(processed);
    }, [data]);

    return (
        <div className="data-processor">
            <h1>{name}</h1>
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

// PropTypes for runtime type checking
DataProcessorComponent.propTypes = {
    initialName: PropTypes.string.isRequired
};

export default DataProcessorComponent;
