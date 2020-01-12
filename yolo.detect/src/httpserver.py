from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import random as rd
import mysql.connector

import config


conn = mysql.connector.connect(host=config.SERVER,
                 user=config.USER,
                 passwd=config.PASSWORD,
                 database=config.DATABASE)
curs = conn.cursor()
curs.execute('select name,price from products;')
ls = list()
for i in curs.fetchall():
    ls.append(str(i[0])+'-'+str(i[1])+'$')
curs.close()
conn.close()

app = Flask(__name__)
CORS(app)
appPort = '5000'


@app.route('/')
@cross_origin()
def index():
    return 'Wellcome flask to API!<br/>'+ls[rd.randint(0, len(ls)-1)]

@app.route('/hello_world', methods=['GET'])
@cross_origin()
def hello_world():
    staff_id = request.args.get('staff_id')
    return 'Hello {0}, {1}'.format(str(staff_id), 'MTA')


def main():
    app.run(debug=True, host='127.0.0.1', port=appPort)

if __name__ == "__main__":
    main()
    #https://github.com/thangnch/MiAI_Flask_2/blob/master/httpsvr.py
