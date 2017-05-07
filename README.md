# EmotionCommotion

## Build Guide 

### Requirements

Install Python 3.5, and the python3-tk package
`sudo apt-get install python3`
1. Get to "EmotionCommotion/EmotionCommotion/" (where manage.py exists)
2. Run `pip install -r requirements.txt`
3. Switch Keras backend to Theano from Tensorflow (Guide: https://keras.io/backend/)
4. Install the whitenoise Django middleware class (Guide: http://whitenoise.evans.io/en/stable/)
5. Install dj-database-url
6. Run `python manage.py runserver`
7. Usually runs on localhost:8000, in your web browser


Alternatively you can install the requirments in a virtual environment for this project (http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/)
