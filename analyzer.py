import sys 
from vqa import VQA


system_prompt = "If the statement starts with 'animal:' then Reply with type of animal the sentence is talking about, using one wordm else If the statement starts with 'car:' then Reply with the type of vehicle the sentence is talking about."

url = 'https://ha.kennethroe.com/api/frigate/frigate/snapshot/1729512173.614739-pmvkiz?authSig=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YWQzYTEzYjk3YzY0MWJiOWNjZTQ1YWI3ODNkYzU0OSIsInBhdGgiOiIvYXBpL2ZyaWdhdGUvZnJpZ2F0ZS9zbmFwc2hvdC8xNzI5NTEyMTczLjYxNDczOS1wbXZraXoiLCJwYXJhbXMiOltdLCJpYXQiOjE3Mjk1MTcwOTQsImV4cCI6MTcyOTYwMzQ5NH0._Y2U7gTzq1tcKhvyrqvTH97WwouPnJj6UnN1fU8dC_0'


if len(sys.argv) == 1:
    print("ERROR: Pass image_url as argument.")
    exit()

image_url = sys.argv[1]

vqa = VQA(system_prompt)
out = vqa.analyze(url)

print(out)
