from flask import Flask,render_template, request
# from dnn.dnn_recommend import get_dnn_recommendations
# from rl.rl_recommend import get_rl_recommendations
# from mf.mf_recommend import get_mf_recommendations
#

from misc_algo.all_recommend import get_rl_recommendations, get_mf_recommendations, get_dnn_recommendations

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',header='Recommendations for User ' , sub_header='', list_header="Recommendations!",
                       recommendations=get_user_recommendations(0, ""), site_title="Grp9")


@app.route('/get_user_id', methods=["POST", "GET"])
def get_user_id() :

    user_id = request.form.get('userid', "")
    algo_id = request.form.get("q1")

    algo_name = get_algo_name(algo_id)
    print ("User Id : {} Algo Id : {}".format(user_id, algo_id))

    algo_id = str(algo_id)
    user_id = int(user_id)
    return render_template('index.html',header='Recommendations for User ' + str(user_id) + " using " + algo_name, sub_header='', list_header="Recommendations!",

                       recommendations=get_user_recommendations(user_id, algo_id), site_title="Grp9")



def get_algo_name(algo_id) :
    map = {
        "a" : "Matrix Factorization",
        "b" : "DNN" ,
        "c" : "RL"
    }

    return map[algo_id]


# def get_dnn_recommendations(userid) :
#     return [("D", ""), ("N", ""), ("N", "")]

# def get_rl_recommendations(userid) :
#     return [("R", "4"), ("L", "1")]

# def get_mf_recommendations(userid) :
#     return [("M", "2"), ("F", "0")]


def get_user_recommendations(userid, algoid) :
    if algoid == "a" :
        return get_mf_recommendations(userid)
    elif algoid == "b" :
        return get_dnn_recommendations(userid)
    elif algoid == "c" :
        return get_rl_recommendations(userid)
    else :
        return []

if __name__ == '__main__':
    app.run(debug=True)
