import os
import csv
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
UPLOAD_FOLDER = "uploads/"
JOB_DESCRIPTION_FILE = "job_description.txt"
OUTPUT_FILE = "ranked_report.csv"

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file_path):
    """Extract text from PDF using PyPDF2"""
    import PyPDF2
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def preprocess(text):
    """Use spaCy for tokenization + lemmatization + stopword removal"""
    doc = nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and token.is_alpha
    ]
    return " ".join(tokens)

def process_resumes():
    if not os.path.exists(JOB_DESCRIPTION_FILE):
        print(f"❌ Job description file '{JOB_DESCRIPTION_FILE}' not found.")
        return

    if not os.path.exists(UPLOAD_FOLDER):
        print(f"❌ Upload folder '{UPLOAD_FOLDER}' not found.")
        return

    # Load Job Description
    with open(JOB_DESCRIPTION_FILE, "r", encoding="utf-8") as f:
        jd_text = f.read()
    jd_clean = preprocess(jd_text)

    resumes = []
    resume_names = []

    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            resume_text = extract_text_from_pdf(file_path)
            resumes.append(preprocess(resume_text))
            resume_names.append(filename)

    if not resumes:
        print("⚠️ No resumes found.")
        return

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    all_docs = [jd_clean] + resumes
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    jd_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]

    # Scoring = cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity(jd_vector, resume_vectors)[0]

    ranked = sorted(
        zip(resume_names, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Save results
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Resume", "Score"])
        for name, score in ranked:
            writer.writerow([name, round(score * 100, 2)])

    print("✅ Ranking complete. Results saved to:", OUTPUT_FILE)
