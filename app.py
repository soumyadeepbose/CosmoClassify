from flask import Flask, render_template, request, session, redirect, url_for
import joblib
import numpy as np
import os

app=Flask(__name__, static_url_path='/static')

model_path = os.path.join(os.path.dirname(__file__), 'DTStellarClass98.pkl')
model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template("space.html")

@app.route('/space.html')
def space():
    return render_template("space.html")

@app.route('/contact.html')
def contact():
    return render_template("contact.html")

@app.route('/analysis.html')
def analysis():
    return render_template("analysis.html")

@app.route('/predictions.html')
def predictions():
    return render_template("predictions.html")

@app.route('/predict', methods=['POST'])
def process():

    u = request.form.get('u')
    g = request.form.get('g')
    r = request.form.get('r')
    i = request.form.get('i')
    z = request.form.get('z')
    redshift = request.form.get('redshift')

    arr = np.array([u, g, r, i, z, redshift])
    arr = arr.reshape(1, -1)

    y_pred = model.predict(arr)

    message = {0: ('Galaxy', 'A galaxy is a vast, gravitationally bound system of stars, gas, dust, and dark matter, typically containing billions to trillions of stars. Galaxies come in various shapes and sizes, ranging from spiral and elliptical to irregular formations. Our Milky Way is an example of a spiral galaxy, and galaxies are the fundamental building blocks of the visible universe.', '/static/Galaxy.mp4'), 
               1: ('Star', 'A neutron star is the incredibly dense and collapsed core of a massive star that has undergone a supernova explosion. Neutron stars are composed almost entirely of neutrons and are remarkably compact, with a mass comparable to that of the Sun but a radius of only a few kilometers. They possess extreme gravitational forces, strong magnetic fields, and rotate rapidly, often emitting beams of radiation, making them observable as pulsars.', '/static/Neutron_star_collison.mp4'), 
               2: ('Quasar', 'A quasar is an incredibly luminous and distant celestial object, powered by the intense radiation emitted from the accretion of matter onto a supermassive black hole at its core. These objects are characterized by their high redshift, compact appearance, and distinct spectral features, and they are typically found at the centers of galaxies, playing a vital role in our understanding of the early universe and the behavior of supermassive black holes.', '/static/Quasar1.mp4')}

    return render_template('predictionresult.html', result=message[y_pred[0]][0], text=message[y_pred[0]][1], u=u, g=g, r=r, i=i, z=z, redshift=redshift, vidpath=message[y_pred[0]][2])

if __name__=="__main__":
    app.run(port=2209, debug=True)
