from flask import Flask
from flask import Flask, render_template
import pickle

#Unload model, patient features, and outcomes from jar
picklejar = pickle.load(open("model.pkl",'rb'))  
model=picklejar[0]
X_ten=picklejar[1]
y_ten=picklejar[2]

#Write some dummy patient names
patient_names=["Hamlet","Lysander","Henry","Juliet","Titania"]

#From imported model, calculate risk percentages
y_predict = model.predict(X_ten[:,1:])

#Write html file
with open('templates/home.html', 'w') as file:
    file.write("<!DOCTYPE html>\n<html>\n<head>\n")
    file.write("<title>Flask app</title>\n")
    file.write('''<link rel="stylesheet" href="{{ url_for('static', filename='css/main.css') }}">\n''')
    file.write("</head>\n</body>\n")
    file.write('''<table style="width:50%">\n''')
    file.write("<tr>\n<th>Name</th>\n<th>Creatinine Z-Score</th>\n<th>Probability of Acute Kidney Injury Mortality</th>\n")
    a=0;
    for patient in patient_names:
        file.write("<tr>\n")
        file.write("<td>%s</td>\n" % patient)
        file.write("<td>%.2f</td>\n" % X_ten[a,2])
        file.write("<td>%.2f</td>\n" % y_predict[a])
        file.write("</tr>\n")
        a=a+1
    file.write("</table>\n")
    file.write("</body>\n")
    file.write("</html>\n")
file.close()

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
if __name__ == '__main__':
    app.run(debug=True)
