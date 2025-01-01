import subprocess
import logging

logger = logging.getLogger(__name__)


def check_dependencies():
    try:
        subprocess.run(["sg", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("""
        Required CLI tool ast-grep (sg) is not installed.
        Please install it with:
            cargo install ast-grep

        For more information, visit: https://ast-grep.github.io/
        """)
        raise RuntimeError("Missing required dependency: ast-grep")


# Run dependency check on import
check_dependencies()
