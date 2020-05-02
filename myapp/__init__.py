from flask import Flask
from config import Config
#from flask_bootstrap import Bootstrap
from flask_material import Material

application = Flask(__name__)
application.config.from_object(Config)
application.debug = True

#bootstrap = Bootstrap(application)
material = Material(application)

import myapp.routes
