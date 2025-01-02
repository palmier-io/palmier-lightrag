package processor

import (
    "fmt"
    "errors"
    "time"
)

// ProcessStatus represents the status of data processing
type ProcessStatus int

const (
    StatusPending ProcessStatus = iota
    StatusSuccess
    StatusError
)

// ProcessError represents custom error types
type ProcessError string

func (e ProcessError) Error() string {
    return string(e)
}

// DataProcessor interface defines the processing contract
type DataProcessor interface {
    ProcessData(data []string) (map[string]interface{}, error)
    GetName() string
    GetStatus() ProcessStatus
}

// DefaultProcessor implements DataProcessor
type DefaultProcessor struct {
    name   string
    status ProcessStatus
}

// NewDefaultProcessor creates a new processor instance
func NewDefaultProcessor(name string) *DefaultProcessor {
    return &DefaultProcessor{
        name:   name,
        status: StatusPending,
    }
}

// ProcessData processes the input data and returns results
func (dp *DefaultProcessor) ProcessData(data []string) (map[string]interface{}, error) {
    if len(data) == 0 {
        return nil, ProcessError("empty data")
    }
    
    results := make(map[string]interface{})
    dp.status = StatusSuccess
    return results, nil
}

// GetName returns the processor name
func (dp *DefaultProcessor) GetName() string {
    return dp.name
}

// GetStatus returns the current processing status
func (dp *DefaultProcessor) GetStatus() ProcessStatus {
    return dp.status
}

// Config holds processor configuration
type Config struct {
    MaxRetries  int
    TimeoutSecs int64
    Debug       bool
}

// GenericResult is a generic result wrapper
type GenericResult[T any] struct {
    Data    T
    Success bool
    Error   error
}

// Nested struct example
type ComplexProcessor struct {
    DefaultProcessor
    config Config
    results []GenericResult[string]
}

// Embedded struct fields
type ProcessorWithMetadata struct {
    *DefaultProcessor
    metadata map[string]interface{}
    created  time.Time
}

// Validator interface for input validation
type Validator interface {
    Validate() error
}

// Generic interface example
type Repository[T any] interface {
    Find(id string) (T, error)
    Save(item T) error
    Delete(id string) error
}

// Interface composition example
type AdvancedProcessor interface {
    DataProcessor
    Validator
    Reset() error
    GetMetrics() map[string]int64
}

// Empty interface (interface{}) example
type AnyProcessor interface {
    Process(data interface{}) interface{}
} 