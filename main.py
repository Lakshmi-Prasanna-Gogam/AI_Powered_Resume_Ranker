import os
from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
from werkzeug.utils import secure_filename
from process_resumes import process_resumes

app = Flask(__name__, template_folder="templates")


# Config paths
UPLOAD_FOLDER = "uploads"
OUTPUT_FILE = "ranked_report.csv"
JOB_DESCRIPTION_FILE = "job_description.txt"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save Job Description
        jobdesc = request.form.get("jobdesc")
        with open(JOB_DESCRIPTION_FILE, "w", encoding="utf-8") as f:
            f.write(jobdesc)

        # Save uploaded resumes
        files = request.files.getlist("resumes")
        for file in files:
            if file and file.filename.lower().endswith(".pdf"):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        # Process resumes
        process_resumes()

        return redirect(url_for("list_candidates"))

    return render_template("index.html")

@app.route("/candidates")
def list_candidates():
    try:
        df = pd.read_csv(OUTPUT_FILE)
        results = df.values.tolist()
    except FileNotFoundError:
        results = None
    return render_template("candidates_list.html", results=results)

@app.route("/download")
def download():
    return send_file(OUTPUT_FILE, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
