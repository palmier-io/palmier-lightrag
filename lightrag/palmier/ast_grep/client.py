from pathlib import Path
from typing import List, Dict, Optional, Tuple
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

    def load_rules(self, language: str, ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Load rules from YAML file for a specific language.

        Args:
            language: The programming language to load rules for
            ids: Optional - if provided, only rules with these IDs will be loaded
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
                return [
                    {"id": doc["id"], "rule": doc["rule"]}
                    for doc in documents
                    if "rule" in doc and (ids is None or doc["id"] in ids)
                ]
        except Exception as e:
            logger.error(f"Error loading rules from {rules_file}: {e}")
            return []

    def scan(
        self, file_path: str, language: str, rules: List[Dict], content: Optional[str] = None
    ) -> List[Tuple[str, SgNode]]:
        """
        Scan a file using ast-grep rules. Equivalent to `sg scan --rule <rule>.yml <file_path>`.

        Args:
            file_path: Path to the file to analyze
            language: Programming language of the file
            rules: List of rule dictionaries compatible with ast-grep Python API
            content: Optional - if not provided, the file at `file_path` will be read
        Returns:
            List of Tuple [rule_id, SgNode] objects
        """
        try:
            if content is None:
                content = Path(file_path).read_text()

            root = SgRoot(content, language)
            node = root.root()

            results = []
            for r in rules:
                matches = node.find_all(r)
                results.extend([(r["id"], match) for match in matches])
            return results

        except Exception as e:
            import traceback

            logger.error(
                f"Error scanning file {file_path}: {e}\n{traceback.format_exc()}"
            )
            return []
