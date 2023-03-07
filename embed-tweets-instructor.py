import json
import hnswlib
from InstructorEmbedding import INSTRUCTOR
import time

model = INSTRUCTOR('hkunlp/instructor-xl', device="cuda")
alltweets = json.load(open("tweets.json"))
tweets = [t['tweet'] for t in alltweets if not t['tweet']['full_text'].startswith("RT")]
idx_instructor = hnswlib.Index(space = 'cosine', dim = 768)
idx_instructor.init_index(max_elements = 500000, ef_construction = 2000, M = 100)
toembed = [["embed this tweet for similarity retrieval", t["full_text"]] for t in tweets]

before = time.time()
ids = list(range(len(tweets)))

print("embedding with instructor...")
i = 0
batch_size = 20

while i < len(toembed):
    data = model.encode(toembed[i:i+batch_size])
    idx_instructor.add_items(data, ids[i:i+batch_size])
    i += batch_size
    print(f"embedded: {i}")
idx_instructor.save_index("tweets-instructor.bin")
t = time.time() - before
print(f"done, took {t} seconds.")
