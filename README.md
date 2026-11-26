# Social Media Abusive Language and Toxic Slang Detector

**Team Semantic Scholars · Natural Language Processing Lab (AIP303)**

This Flask app screens English comments for human review. It shows Naive Bayes and rule-based decisions, reviewed terms, browser-session frequencies, and evaluation results. The saved JSON includes the trained model, preprocessing configuration, and evaluation.

## Run locally

Use Python 3.12 or later. From the folder containing `app.py`:

```sh
python -m pip install -r requirements.txt
python app.py
```

Open [the local dashboard](http://127.0.0.1:8765/) and select **Try 3 examples**. The health endpoint is `GET /health`, and inference uses `POST /api/check` with a JSON `text` field.

## Deploy on Render

Extract the ZIP and commit its contents at the GitHub repository root. In Render, create a **Web Service**, connect the repository, and use these settings. [Render Web Services](https://render.com/docs/web-services)

| Setting | Value |
| --- | --- |
| Language | Python 3 |
| Instance type | Free |
| Build command | `pip install -r requirements.txt` |
| Start command | `python app.py` |
| Environment variable | `APP_HOST=0.0.0.0` |
| Health check path | `/health` |

Render supplies `PORT`. The `.python-version` file selects Python 3.12. An optional Dockerfile supports the same app for container deployment.

After deployment, test the public URL and the **Try 3 examples** action. The Free instance may take about a minute to wake after 15 minutes without inbound traffic. [Render Free limitations](https://render.com/docs/free)

## Evaluation

| Test set | Comments | Rules F1 | Naive Bayes F1 |
| --- | ---: | ---: | ---: |
| Held-out Jigsaw | 2,698 | 67.7% | 75.2% |
| Synthetic chat | 40 | 41.2% | 58.5% |

The dashboard uses Naive Bayes with the validation-selected cutoff of 0.85. The evaluation panels show precision, recall and error counts. Results are review flags, and the 40 synthetic cases are a diagnostic sample. [Jigsaw challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)
