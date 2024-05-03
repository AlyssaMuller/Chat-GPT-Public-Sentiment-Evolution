prep:
	python3 ./split.py --data-file english_tweets.csv
	python3 ./frequency.py

send:
python3 ./sentiment_analysis.py
