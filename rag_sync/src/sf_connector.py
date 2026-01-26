import logging
import os
import requests
from simple_salesforce import Salesforce
from .config import Config

logger = logging.getLogger("rag_sync.sf_connector")

class SalesforceConnector:
    def __init__(self):
        username = os.getenv("SF_USERNAME")
        password = os.getenv("SF_PASSWORD")
        token = os.getenv("SF_SECURITY_TOKEN")
        client_id = os.getenv("SF_CLIENT_ID")
        client_secret = os.getenv("SF_CLIENT_SECRET")
        domain_raw = os.getenv("SF_DOMAIN", "login")

        if not all([username, password, token, client_id, client_secret]):
            missing = []
            if not username: missing.append("SF_USERNAME")
            if not password: missing.append("SF_PASSWORD")
            if not token: missing.append("SF_SECURITY_TOKEN")
            if not client_id: missing.append("SF_CLIENT_ID")
            if not client_secret: missing.append("SF_CLIENT_SECRET")
            logger.error(f"Faltan variables para OAuth2 REST: {', '.join(missing)}")
            self.sf = None
            return

        # Limpieza del dominio para la URL de OAuth
        domain = domain_raw.replace('https://', '').replace('http://', '').split('.')[0]
        if not domain or domain == "login":
            base_url = "login.salesforce.com"
        elif domain == "test":
            base_url = "test.salesforce.com"
        else:
            base_url = f"{domain}.my.salesforce.com"

        try:
            logger.info(f"Autenticando vía REST OAuth2 ({base_url})...")
            
            # Paso 1: Obtener Access Token vía REST
            auth_url = f"https://{base_url}/services/oauth2/token"
            payload = {
                'grant_type': 'password',
                'client_id': client_id,
                'client_secret': client_secret,
                'username': username,
                'password': f"{password}{token}" # Combinación estándar
            }
            
            response = requests.post(auth_url, data=payload)
            response.raise_for_status()
            auth_data = response.json()
            
            access_token = auth_data['access_token']
            instance_url = auth_data['instance_url']
            
            # Paso 2: Inicializar Salesforce con el token ya obtenido
            self.sf = Salesforce(
                instance_url=instance_url,
                session_id=access_token
            )
            logger.info("¡Conexión REST OAuth2 exitosa con Salesforce!")
            
        except Exception as e:
            logger.error(f"Error en autenticación REST OAuth2: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Detalle error SF: {e.response.text}")
            self.sf = None

    def get_knowledge_articles(self):
        if not self.sf:
            return []

        # SOQL optimizado para REST
        query = "SELECT Id, Title, ArticleBody__c, LastModifiedDate FROM Knowledge__kav WHERE IsLatestVersion = true AND PublishStatus = 'Online' AND Language = 'es'"
        
        try:
            results = self.sf.query_all(query)
            articles = results.get('records', [])
            logger.info(f"Se han encontrado {len(articles)} artículos vía REST API.")
            return articles
        except Exception as e:
            logger.warning(f"Error query inicial (Language): {str(e)}")
            try:
                query_retry = "SELECT Id, Title, ArticleBody__c, LastModifiedDate FROM Knowledge__kav WHERE IsLatestVersion = true AND PublishStatus = 'Online'"
                results = self.sf.query_all(query_retry)
                return results.get('records', [])
            except Exception as e2:
                logger.error(f"Error total en REST query: {str(e2)}")
                return []

    def get_article_details(self, article_id: str):
        try:
            # Los campos ya vienen en la query principal para mayor eficiencia
            article = self.sf.Knowledge__kav.get(article_id)
            content_html = f"<h3>{article.get('Title')}</h3><div>{article.get('ArticleBody__c', '')}</div>"
            return {
                "id": article['Id'],
                "title": article['Title'],
                "html": content_html,
                "modified": article['LastModifiedDate']
            }
        except Exception as e:
            logger.error(f"Error en detalle REST para {article_id}: {e}")
            return None
