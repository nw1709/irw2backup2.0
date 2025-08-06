from openai import OpenAI
client = OpenAI()

response = client.responses.create(
  prompt={
    "id": "pmpt_68926fcc3d188193a1fe80d0772aaacb0b558abf1fd783b6",
    "version": "4"
  }
)
