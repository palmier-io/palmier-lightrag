use std::collections::HashMap;
use std::error::Error;
use std::fmt;

/// ProcessStatus represents the status of data processing
#[derive(Debug, PartialEq)]
pub enum ProcessStatus {
    Pending,
    Success,
    Error,
}

/// Custom error type for processing errors
/// Double line comment
#[derive(Debug)]
pub struct ProcessError(String);

impl fmt::Display for ProcessError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Process error: {}", self.0)
    }
}

impl Error for ProcessError {}

/// Trait for data processing operations
pub trait DataProcessor {
    fn process_data(&mut self, data: Vec<String>) -> Result<HashMap<String, i32>, ProcessError>;
    fn get_name(&self) -> &str;
    fn get_status(&self) -> ProcessStatus;
}

/// Main processor implementation
pub struct DefaultProcessor {
    name: String,
    status: ProcessStatus,
}

impl DefaultProcessor {
    /// Create a new DataProcessor instance
    pub fn new(name: String) -> Self {
        DefaultProcessor { 
            name,
            status: ProcessStatus::Pending,
        }
    }
}

impl DataProcessor for DefaultProcessor {
    fn process_data(&mut self, data: Vec<String>) -> Result<HashMap<String, i32>, ProcessError> {
        if data.is_empty() {
            return Err(ProcessError("Empty data provided".to_string()));
        }

        let mut results = HashMap::new();
        // Processing logic here
        self.status = ProcessStatus::Success;
        Ok(results)
    }

    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_status(&self) -> ProcessStatus {
        self.status
    }
}

/// Generic implementation example
pub struct GenericProcessor<T> {
    data: T,
}

impl<T> GenericProcessor<T> {
    pub fn new(data: T) -> Self {
        GenericProcessor { data }
    }
}

/// Example trait for testing the trait rule
pub trait ExampleTrait {
    fn example_method(&self) -> String;
    fn default_method(&self) -> bool {
        true
    }
}

/// Another trait with generic parameter
pub trait GenericTrait<T> {
    fn process(&self, value: T) -> Result<T, ProcessError>;
}

/// A standalone function example
pub fn process_items(items: Vec<String>) -> Result<(), ProcessError> {
    Ok(())
}

/// Generic function example
pub fn transform<T>(input: T) -> T {
    input
}

/// Function with multiple parameters
pub fn combine(first: String, second: String) -> String {
    type ProcessResult<T> = Result<T, ProcessError>;
    format!("{}{}", first, second)
}

/// Maximum number of retries
pub const MAX_RETRIES: u32 = 3;

/// Default timeout in seconds
pub const TIMEOUT_SECONDS: i32 = 30;

/// Complex constant with type annotation
pub const EMPTY_RESULT: Result<(), ProcessError> = Ok(());

/// Type alias example
pub type ProcessResult<T> = Result<T, ProcessError>;

// Static constant
pub static GLOBAL_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0); 