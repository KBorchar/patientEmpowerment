# initializes flask_app package
# every file in this package has access to the 'app' variable which is the global server instance
# from the Flask framework
from flask import Flask

app = Flask(__name__)

from flask_app import routes