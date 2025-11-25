from flask import Flask, request, jsonify, render_template
from llm_judges import deepseek_judge, mistral_self_judge


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.post("/evaluate")
def eval():
    try:
        rec = request.json
        print("Incoming JSON:", rec)
        return jsonify({
            "mistral": mistral_self_judge(rec),
            "deepseek": deepseek_judge(rec)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


app.run(port=8000)
