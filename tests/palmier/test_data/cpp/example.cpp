#include <string>
#include <vector>
#include <map>

namespace processor {

// Enum for processing status
enum class ProcessStatus {
    SUCCESS,
    ERROR,
    PENDING
};

// Interface for data processing
class IProcessor {
public:
    virtual ~IProcessor() = default;
    virtual std::map<std::string, int> processData(const std::vector<std::string>& data) = 0;
    virtual std::string getName() const = 0;
};

// Template class example
template<typename T>
class DataContainer {
public:
    void add(T item);
    T get() const;
private:
    T data;
};

class DataProcessor : public IProcessor {
private:
    std::string name;
    ProcessStatus status;

public:
    DataProcessor(const std::string& name) : name(name), status(ProcessStatus::PENDING) {}

    /**
     * Process input data and return results
     */
    std::map<std::string, int> processData(const std::vector<std::string>& data) override {
        std::map<std::string, int> results;
        status = ProcessStatus::SUCCESS;
        return results;
    }

    std::string getName() const override {
        return name;
    }

    ProcessStatus getStatus() const {
        return status;
    }
};

} // namespace processor
