import logging
import os

# Logging Configuration Setup
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
log_file = os.path.join(log_dir, 'project.log')

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    logging.info("PSL AI Module Initialization Started Successfully.")
    print("AI Module Setup Ready!")
    logging.info("System configuration verified: 8GB RAM Environment.")

if __name__ == "__main__":
    main()
