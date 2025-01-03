from pathlib import Path
from typing import List, Dict, Optional
import logging
import yaml

try:
    from ast_grep_py import SgRoot, SgNode
except ImportError:
    logging.error(
        "ast-grep-py is not installed. Please install it with 'pip install ast-grep-py'"
    )

logger = logging.getLogger(__name__)


class AstGrepClient:
    def __init__(self, rules_dir: Optional[Path] = None):
        self.rules_dir = rules_dir or Path(__file__).parent / "rules"

    def load_rules(self, language: str) -> List[Dict]:
        """
        Load rules from YAML file for a specific language.

        Args:
            language: The programming language to load rules for
        Returns:
            List of rule dictionaries compatible with ast-grep Python API
        """
        rules_file = self.rules_dir / f"{language}.yml"
        if not rules_file.exists():
            logger.error(f"Rules file not found: {rules_file}")
            return []

        try:
            with open(rules_file) as f:
                documents = yaml.safe_load_all(f)
                return [{"rule": doc["rule"]} for doc in documents if "rule" in doc]
        except Exception as e:
            logger.error(f"Error loading rules from {rules_file}: {e}")
            return []

    def scan(
        self, file_path: str, language: str, content: Optional[str] = None
    ) -> List[SgNode]:
        """
        Scan a file using ast-grep rules. Equivalent to `sg scan --rule <rule>.yml <file_path>`.

        Args:
            file_path: Path to the file to analyze
            language: Programming language of the file
            content: Optional - if not provided, the file at `file_path` will be read
        Returns:
            List of SgNode objects
        """
        try:
            rules = self.load_rules(language)
            if not rules:
                return []

            if content is None:
                content = Path(file_path).read_text()

            root = SgRoot(content, language)
            node = root.root()

            results = []
            for rule in rules:
                matches = node.find_all(rule)
                results.extend(matches)
            return results

        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")
            return []

    def get_skeleton(self, file_path: str, language: str) -> str:
        """
        Generate a skeleton structure of a file.
        A skeleton is a string containing the structural elements without implementation details.

        Args:
            file_path: Path to the file
            language: Programming language of the file
        Returns:
            String containing the structural elements without implementation details
        """
        content = Path(file_path).read_text()
        content_lines = content.splitlines()
        results = self.scan(file_path, language, content)
        ordered_lines = []
        seen = set()
        for result in results:
            line_number = result.range().start.line
            if line_number in seen:
                continue
            seen.add(line_number)
            ordered_lines.append(
                (line_number, content_lines[line_number])
            )  # Only include the first line of each match
        return "\n".join(text for _, text in ordered_lines)
