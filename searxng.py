from langchain_community.utilities import SearxSearchWrapper
search = SearxSearchWrapper(searx_host="http://localhost:8080")
input = input("recherche : ")
moteur = search.run(input)
print(moteur)