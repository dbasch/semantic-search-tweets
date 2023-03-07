# Simple semantic search for all your tweets

1. request your personal data archive. Follow the directions [here](https://help.twitter.com/en/managing-your-account/how-to-download-your-twitter-archive)
2. There will be a file called `data/tweets.js`. Mine had a single variable assigned to an array of tweet objects. Edit it, leave only the array and rename it to tweets.json.
3. `pip install -r requirements.txt`
4. run `python embed-tweets-instructor.py` to create embeddings for all your tweets using the instructorXL model. This model is large, and it will take a while to download. I tested this on a GPU with 8GB of VRAM, not sure if it works with something smaller. You can easily change this code to use smaller models.
5. run `python search-instructor.py` to run semantic queries.
