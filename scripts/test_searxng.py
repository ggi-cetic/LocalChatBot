import requests
from bs4 import BeautifulSoup

def get_csrf_token(instance_url):
    # Récupérer la page d'accueil pour extraire le token CSRF
    response = requests.get(instance_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Le token CSRF est généralement dans une balise meta ou un champ caché
    csrf_token = soup.find('input', {'name': 'csrf_token'})['value']
    return csrf_token

def search_searxng(instance_url, query, format="json"):
    # Récupérer le token CSRF
    #csrf_token = get_csrf_token(instance_url)

    # Paramètres de la requête
    params = {
        'q': query,
        'format': format
    }

    # En-têtes avec le token CSRF
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    # Effectuer la requête
    response = requests.get(instance_url + "/search", params=params)
    response.raise_for_status()
    return response.json()

# Exemple d'utilisation :
instance_url = "http://127.0.0.1:5002"  # Remplacez par l'URL correcte
query = "python programming"
try:
    results = search_searxng(instance_url, query)
    for result in results["results"]:
        print(result)
except Exception as e:
    print(f"Erreur : {e}")