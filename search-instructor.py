import json
import hnswlib
from InstructorEmbedding import INSTRUCTOR

model = INSTRUCTOR('hkunlp/instructor-xl', device="cuda")

p = hnswlib.Index(space='cosine', dim=768)
p.load_index("tweets-instructor.bin", max_elements = 500000)


alltweets = json.load(open("tweets.json"))
tweets = [t['tweet'] for t in alltweets if not t['tweet']['full_text'].startswith("RT")]

while True:
    query = input("Enter query:")
    vec = model.encode(query)
    labels, distances = p.knn_query(vec, k=10)
    for l in labels[0]:
        print(tweets[l]['full_text'])