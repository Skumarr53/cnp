from dotenv import load_dotenv
from centralized_nlp_package.utils import get_config


# load_dotenv()
# Initialize configuration first
config = get_config()

# Now set up logging using the initialized config
from centralized_nlp_package.utils import setup_logging


setup_logging()
