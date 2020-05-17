from flask import Flask, render_template, request
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import analyse

Path("./uploads/").mkdir(exist_ok=True)
Path("./thumbnails/").mkdir(exist_ok=True)


UPLOAD_FOLDER = 'uploads'
THUMBNAIL_FOLDER = 'thumbnails'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['THUMBNAIL_FOLDER'] = THUMBNAIL_FOLDER


# Enter your mysql configuration here
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'username'
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'videoE'

mysql = MySQL(app)

# remove frames with more than 50% resemblance
scale = '0.5'


@app.route("/", methods=['GET', 'POST'])
def home():
    # IF A POST REQUEST OCCURS
    if request.method == 'POST':
        
        # IF A FILE IS SENT
        if request.files:
            video = request.files["video"]
            
            # secure version of the filename
            filename = secure_filename(video.filename)
            print(filename)
            
            # save video in uploads foler
            video.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # remove duplicate frames
            os.system('ffmpeg -i '+ 'uploads/'+ filename + ' -vf  "select=gt(scene\,'+ scale +'), scale=640:360" -vsync vfr thumbnails/'+ filename +'Thumb%03d.png')

            # create a cursor
            cur = mysql.connection.cursor()

            # iterating over each file
            for filename in os.listdir(app.config['THUMBNAIL_FOLDER']):
                print("analysing: ", filename)

                # calling start_analysis function and giving it png file
                result = analyse.start_analysis(os.path.join(app.config['THUMBNAIL_FOLDER'], filename))
                print("result: ", result)

                # if no error returned
                if result:
                    angerProb = result[1]['Anger proability']
                    disgustProb = result[1]['Disgust proability']
                    happyProb = result[1]['Happy proability']
                    sadProb = result[1]['Sad proability']
                    surprisedProb = result[1]['Surprised proability']

                    # inserting into mysql table
                    # change according to your needs
                    cur.execute("insert into prob(fileName, total_rounded_prob, Anger_prob, Disgust_prob, Happy_prob, Sad_prob, Surprised_prob) VALUES('"+filename+"', "+str(result[0])+", '"+angerProb+"', '"+disgustProb+"', '"+happyProb+"', '"+sadProb+"', '"+surprisedProb+"');")
                    mysql.connection.commit()

                # removing unnecessary files
                print("Deleting: ", filename)
                os.remove(os.path.join(app.config['THUMBNAIL_FOLDER'], filename))

            # closing the connection
            cur.close()
            
            
        return "<h1>Thank You!</h1>"

    return render_template("upload.html")


# running app
app.run(host='0.0.0.0')
