import java.util.List;
import java.util.Map;
import java.util.HashMap;

// Line comment test
public class Example {  // Class test
    private final String name;
    
    // Enum test
    public enum Status {
        ACTIVE,
        INACTIVE
    }
    
    // Interface test
    private interface Processor {
        void process();
    }
    
    /**
     * Block comment test
     * Create a new processor
     * @param name The processor name
     */
    public Example(String name) {
        this.name = name;
    }
    
    // Method test
    public Map<String, Object> processData(List<String> data) {
        Map<String, Object> result = new HashMap<>();
        for (String item : data) {
            result.put(item, item.length());
        }
        return result;
    }
    
    public String getName() {
        return name;
    }
} 