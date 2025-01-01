class ExampleClass:
    """A class that processes data."""
    
    def __init__(self, name: str):
        """Initialize with name."""
        self.name = name
        
    def process_data(self, data: list) -> dict:
        """
        Process the input data.
        
        Args:
            data: List of items to process
            
        Returns:
            Processed data as dictionary
        """
        result = {}
        for item in data:
            result[item] = len(item)
        return result
        
    def get_name(self) -> str:
        """Return the name."""
        return self.name 