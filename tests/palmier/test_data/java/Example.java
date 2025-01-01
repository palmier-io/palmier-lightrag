import java.util.List;
import java.util.Map;
import java.util.HashMap;

/**
 * A class that processes data
 */
public class Example {
    private final String name;
    
    /**
     * Create a new processor
     * @param name The processor name
     */
    public Example(String name) {
        this.name = name;
    }
    
    /**
     * Process the input data
     * @param data Data to process
     * @return Processed data
     */
    public Map<String, Object> processData(List<String> data) {
        Map<String, Object> result = new HashMap<>();
        for (String item : data) {
            result.put(item, item.length());
        }
        return result;
    }
    
    /**
     * Get the processor name
     * @return The name
     */
    public String getName() {
        return name;
    }
} 