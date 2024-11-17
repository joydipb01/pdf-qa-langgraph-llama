from qna_agent import *

question = input("Enter your question: ")

VERBOSE = False
inputs = {"messages": [("human", question)]}
for output in graph.stream(inputs):
    answer = output
    print("\n---\n")

print("FINAL ANSWER:")
print(answer['finalize_response']['messages'][0].content.split("Answer:", 1)[-1].strip())