from textblob import TextBlob

def text_risk_score(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity < -0.4:
        return 2
    elif polarity < -0.2:
        return 1
    else:
        return 0
