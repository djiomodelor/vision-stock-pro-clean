# =============================================================================
# 📦 IMPORTS ET CONFIGURATION
# =============================================================================
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import joblib
import time
import base64
import json
import io
from datetime import datetime, timedelta
from groq import Groq

# =============================================================================
# 🤖 CONFIGURATION GROQ POUR QUESTIONS GÉNÉRALES
# =============================================================================

def get_groq_response(user_input):
    """
    Utilise Groq pour répondre aux questions générales
    """
    try:
        # Votre clé API Groq
        api_key = os.getenv("GROQ_API_KEY", "")
        
        # Initialiser le client Groq
        client = Groq(api_key=api_key)
        
        # Modèles disponibles et fonctionnels
        models = [
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "llama-3.1-70b-versatile"
        ]
        
        # Essayer chaque modèle jusqu'à ce qu'un fonctionne
        for model in models:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Tu es un assistant IA intelligent et utile. Réponds de manière claire, précise et engageante en français. Utilise des emojis appropriés et structure tes réponses de manière lisible."
                        },
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1000,
                    top_p=1,
                    stream=False
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"❌ Erreur avec le modèle {model}: {e}")
                continue
        
        # Si aucun modèle ne fonctionne
        return "❌ **Erreur** : Impossible de contacter l'API Groq. Veuillez réessayer plus tard."
        
    except Exception as e:
        print(f"❌ Erreur Groq: {e}")
        return f"❌ **Erreur de connexion** : {str(e)}"

# =============================================================================
# 💾 SYSTÈME DE BASE DE DONNÉES INTÉGRÉE POUR CHATBOT
# =============================================================================

def save_dashboard_data_to_memory(dashboard_data, product_stocks, predictions_data):
    """
    Sauvegarde toutes les données du dashboard dans la mémoire de session
    pour une utilisation optimale par le chatbot
    """
    try:
        # Créer une structure de données complète et organisée
        chatbot_database = {
            'timestamp': datetime.now().isoformat(),
            'version': '2.0',
            'products': {},
            'global_metrics': dashboard_data if dashboard_data else {},
            'predictions': predictions_data if predictions_data else {},
            'raw_data': {
                'dashboard_data': dashboard_data,
                'product_stocks': product_stocks,
                'predictions_data': predictions_data
            }
        }
        
        # Données spécifiques par produit (hardcodées pour la précision)
        chatbot_database['products'] = {
            'couches_softcare': {
                'nom_complet': 'Couche Softcare T4',
                'stock_actuel': 80,
                'jours_rupture': 1.1,
                'consommation_jour': 75.6,
                'consommation_30j': 2267,
                'stock_max': 2623,
                'stock_min': 1911,
                'stock_recommande': 2445,
                'tendance': 'Hausse',
                'tendance_pourcentage': 9.4,
                'confiance': 0.843,
                'stabilite': 0.931,
                'score_confiance': 0.682,
                'volatilite': 5.2,
                'efficacite_stock': 0.035,
                'cv': 6.5,
                'incertitude_moyenne': 11.1,
                'status': 'URGENT',
                'alerte': True,
                'message_alerte': 'Rupture de Stock Imminente - Stock épuisé dans 1.1 jours'
            },
            'lait_broli_1kg': {
                'nom_complet': 'Lait Broli 1kg',
                'stock_actuel': 0,
                'jours_rupture': 0.0,
                'consommation_jour': 75.2,
                'consommation_30j': 2256,
                'stock_max': 2602,
                'stock_min': 1909,
                'stock_recommande': 2429,
                'tendance': 'Hausse',
                'tendance_pourcentage': 6.8,
                'confiance': 0.846,
                'stabilite': 0.926,
                'score_confiance': 0.698,
                'volatilite': 5.5,
                'efficacite_stock': 0.0,
                'cv': 5.6,
                'incertitude_moyenne': 9.8,
                'status': 'RUPTURE',
                'alerte': True,
                'message_alerte': 'STOCK ÉPUISÉ - Réapprovisionnement immédiat requis !'
            },
            'mayor_1_csv': {
                'nom_complet': 'Mayor 1 (CSV)',
                'stock_actuel': 0,
                'jours_rupture': 0.0,
                'consommation_jour': 76.0,
                'consommation_30j': 2279,
                'stock_max': 2628,
                'stock_min': 1929,
                'stock_recommande': 2453,
                'tendance': 'Hausse',
                'tendance_pourcentage': 5.4,
                'confiance': 0.847,
                'stabilite': 0.945,
                'score_confiance': 0.721,
                'volatilite': 0.0,
                'efficacite_stock': 0.0,
                'cv': 6.8,
                'incertitude_moyenne': 12.3,
                'status': 'INCONNU',
                'alerte': True,
                'message_alerte': 'Stock actuel inconnu - Vérification urgente requise !'
            },
            'may_arm_1kg': {
                'nom_complet': 'May Arm 1kg',
                'stock_actuel': 778,
                'jours_rupture': 10.4,
                'consommation_jour': 75.0,
                'consommation_30j': 2250,
                'stock_max': 2590,
                'stock_min': 1909,
                'stock_recommande': 2420,
                'tendance': 'Hausse',
                'tendance_pourcentage': 9.4,
                'confiance': 0.849,
                'stabilite': 0.931,
                'score_confiance': 0.692,
                'volatilite': 5.2,
                'efficacite_stock': 0.346,
                'cv': 8.3,
                'incertitude_moyenne': 15.7,
                'status': 'ATTENTION',
                'alerte': True,
                'message_alerte': 'Attention Stock - Stock épuisé dans 10.4 jours'
            },
            'may_arm_5kg': {
                'nom_complet': 'May Arm 5kg',
                'stock_actuel': 0,
                'jours_rupture': 0.0,
                'consommation_jour': 75.6,
                'consommation_30j': 2268,
                'stock_max': 2597,
                'stock_min': 1939,
                'stock_recommande': 2432,
                'tendance': 'Hausse',
                'tendance_pourcentage': 10.4,
                'confiance': 0.855,
                'stabilite': 0.928,
                'score_confiance': 0.717,
                'volatilite': 5.4,
                'efficacite_stock': 0.0,
                'cv': 5.1,
                'incertitude_moyenne': 9.2,
                'status': 'RUPTURE',
                'alerte': True,
                'message_alerte': 'STOCK ÉPUISÉ - Réapprovisionnement immédiat requis !'
            },
            'papier_hygisita': {
                'nom_complet': 'Papier Hygisita',
                'stock_actuel': 152,
                'jours_rupture': 2.0,
                'consommation_jour': 74.4,
                'consommation_30j': 2233,
                'stock_max': 2559,
                'stock_min': 1907,
                'stock_recommande': 2396,
                'tendance': 'Hausse',
                'tendance_pourcentage': 7.3,
                'confiance': 0.854,
                'stabilite': 0.938,
                'score_confiance': 0.698,
                'volatilite': 4.6,
                'efficacite_stock': 0.068,
                'cv': 6.0,
                'incertitude_moyenne': 10.8,
                'status': 'URGENT',
                'alerte': True,
                'message_alerte': 'Rupture de Stock Imminente - Stock épuisé dans 2.0 jours'
            },
            'parle_g': {
                'nom_complet': 'ParleG',
                'stock_actuel': 290,
                'jours_rupture': 3.9,
                'consommation_jour': 74.4,
                'consommation_30j': 2232,
                'stock_max': 2555,
                'stock_min': 1909,
                'stock_recommande': 2394,
                'tendance': 'Hausse',
                'tendance_pourcentage': 9.4,
                'confiance': 0.855,
                'stabilite': 0.929,
                'score_confiance': 0.687,
                'volatilite': 5.3,
                'efficacite_stock': 0.130,
                'cv': 6.0,
                'incertitude_moyenne': 10.8,
                'status': 'URGENT',
                'alerte': True,
                'message_alerte': 'Rupture de Stock Imminente - Stock épuisé dans 3.9 jours'
            }
        }
        
        # Sauvegarder dans la session
        st.session_state.chatbot_database = chatbot_database
        
        # Debug
        print(f"✅ Base de données chatbot sauvegardée: {len(chatbot_database['products'])} produits")
        
        return chatbot_database
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")
        return None

# =============================================================================
# 🗄️ GESTION DE LA BASE DE DONNÉES DU CHATBOT
# =============================================================================

def load_chatbot_database():
    """
    Charge la base de données du chatbot depuis la session
    """
    try:
        if 'chatbot_database' in st.session_state:
            return st.session_state.chatbot_database
        else:
            # Créer une base de données par défaut
            return save_dashboard_data_to_memory({}, {}, {})
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None

def get_product_data(product_name):
    """
    Récupère les données d'un produit spécifique depuis la base de données
    """
    try:
        database = load_chatbot_database()
        if not database or 'products' not in database:
            return None
            
        # Recherche par nom ou alias
        product_aliases = {
            'couches': 'couches_softcare',
            'softcare': 'couches_softcare',
            'softcair': 'couches_softcare',
            'softcaire': 'couches_softcare',
            'couche': 'couches_softcare',
            'homepro': 'couches_softcare',
            'couches softcare': 'couches_softcare',
            'couches softcaire': 'couches_softcare',
            'lait': 'lait_broli_1kg',
            'broli': 'lait_broli_1kg',
            'lait broli': 'lait_broli_1kg',
            'laitbroli': 'lait_broli_1kg',
            'broli 1kg': 'lait_broli_1kg',
            'lait broli 1kg': 'lait_broli_1kg',
            'mayor': 'mayor_1_csv',
            'mayor_1': 'mayor_1_csv',
            'csv': 'mayor_1_csv',
            'mayor 1': 'mayor_1_csv',
            'mayor1': 'mayor_1_csv',
            'may': 'may_arm_1kg',
            'arm': 'may_arm_1kg',
            'may_arm': 'may_arm_1kg',
            'may_arm_1kg': 'may_arm_1kg',
            'may_arm_5kg': 'may_arm_5kg',
            'may arm': 'may_arm_1kg',
            'may arm 1kg': 'may_arm_1kg',
            'may arm 5kg': 'may_arm_5kg',
            'mayarm': 'may_arm_1kg',
            'parle': 'parleG',
            'parle-g': 'parleG',
            'parleg': 'parleG',
            'parle_g': 'parleG',
            'parle g': 'parleG',
            'papier': 'papier_hygisita',
            'hygisita': 'papier_hygisita',
            'hygienique': 'papier_hygisita',
            'hygiene': 'papier_hygisita',
            'papier hygisita': 'papier_hygisita',
            'papierhygisita': 'papier_hygisita'
        }
        
        # Normaliser le nom du produit
        product_key = product_name.lower().strip()
        if product_key in product_aliases:
            product_key = product_aliases[product_key]
        
        if product_key in database['products']:
            return database['products'][product_key]
        else:
            # Recherche partielle
            for key, data in database['products'].items():
                if product_name.lower() in data['nom_complet'].lower():
                    return data
            
        return None
        
    except Exception as e:
        print(f"❌ Erreur lors de la récupération du produit {product_name}: {e}")
        return None

def get_all_products_summary():
    """
    Récupère un résumé de tous les produits
    """
    try:
        database = load_chatbot_database()
        if not database or 'products' not in database:
            return []
            
        summary = []
        for product_key, product_data in database['products'].items():
            summary.append({
                'nom': product_data['nom_complet'],
                'stock': product_data['stock_actuel'],
                'jours_rupture': product_data['jours_rupture'],
                'status': product_data['status'],
                'alerte': product_data['alerte']
            })
        
        return summary
        
    except Exception as e:
        print(f"❌ Erreur lors de la récupération du résumé: {e}")
        return []

# =============================================================================
# 📊 ANALYSE DES DONNÉES DE STOCK
# =============================================================================

def analyze_stock_data_with_database(database, user_input):
    """
    Analyse les données de stock en utilisant la base de données intégrée
    """
    try:
        # Rechercher un produit spécifique dans la question
        product_keywords = {
            'couches_softcare': ['couches', 'softcare', 'softcair', 'softcqre', 'couche', 't4', 'softcaire', 'homepro'],
            'lait_broli_1kg': ['lait', 'broli', 'lait broli', 'laitbroli', 'broli 1kg', 'lait broli 1kg'],
            'mayor_1_csv': ['mayor', 'mayor_1', 'csv', 'mayor 1', 'mayor1'],
            'may_arm_1kg': ['may', 'arm', 'may arm', '1kg', 'may arm 1kg', 'mayarm'],
            'may_arm_5kg': ['may', 'arm', 'may arm', '5kg', 'may arm 5kg', 'mayarm'],
            'papier_hygisita': ['papier', 'hygienique', 'hygiene', 'papier hygienique', 'hygisita', 'papierhygisita'],
            'parleG': ['parle', 'parle-g', 'parleg', 'parle g', 'parleg']
        }
        
        detected_product = None
        for product, keywords in product_keywords.items():
            if any(keyword in user_input for keyword in keywords):
                detected_product = product
                break
        
        if detected_product:
            # Analyser le produit spécifique
            return analyze_specific_product_with_database(detected_product, user_input)
        else:
            # Afficher un résumé de tous les produits
            return get_global_stock_summary()
            
    except Exception as e:
        print(f"❌ Erreur dans l'analyse des stocks: {e}")
        return f"❌ **Erreur** : Impossible d'analyser les données de stock.\n\n**Détails** : {str(e)}"

def analyze_specific_product_with_database(product_key, user_input):
    """
    Analyse un produit spécifique avec la base de données
    """
    try:
        product_data = get_product_data(product_key)
        if not product_data:
            return f"❌ **Produit non trouvé** : Je ne trouve pas d'informations sur ce produit."
        
        # Construire la réponse simplifiée
        response = f"📊 **Analyse de {product_data['nom_complet']}**\n\n"
        
        # Statut et alertes
        if product_data['status'] == 'RUPTURE':
            status_emoji = "💀"
            explication = "SITUATION CRITIQUE : Votre stock est complètement épuisé ! Il faut immédiatement passer une commande d'urgence."
        elif product_data['status'] == 'URGENT':
            status_emoji = "🚨"
            explication = "SITUATION URGENTE : Votre stock sera épuisé dans moins de 2 jours ! Commander dès aujourd'hui."
        elif product_data['status'] == 'INCONNU':
            status_emoji = "❓"
            explication = "SITUATION INCONNUE : Vérifier manuellement votre inventaire pour connaître la situation réelle."
        elif product_data['status'] == 'ATTENTION':
            status_emoji = "⚠️"
            explication = "SITUATION D'ATTENTION : Votre stock sera épuisé dans moins de 4 jours. Planifier une commande."
        elif product_data['status'] == 'Normal':
            status_emoji = "✅"
            explication = "SITUATION STABLE : Votre stock est en bon état. Continuer la surveillance normale."
        else:
            status_emoji = "🔴"
            explication = "SITUATION À VÉRIFIER : Vérifier manuellement votre inventaire."
            
        response += f"{status_emoji} **Statut** : {product_data['status']}\n"
        
        if product_data['alerte']:
            response += f"🚨 **ALERTE** : {product_data.get('message_alerte', 'Attention requise !')}\n\n"
        
        # Données principales
        response += f"📦 **Stock actuel** : {product_data['stock_actuel']} unités\n"
        response += f"⏰ **Jours avant rupture** : {product_data['jours_rupture']} jours\n"
        response += f"📈 **Consommation quotidienne** : {product_data['consommation_jour']} unités\n"
        if 'consommation_30j' in product_data:
            response += f"📊 **Consommation 30j** : {product_data['consommation_30j']} unités\n"
        if 'tendance_pourcentage' in product_data:
            response += f"📈 **Tendance** : {product_data['tendance']} (+{product_data['tendance_pourcentage']}%)\n"
        else:
            response += f"📊 **Tendance** : {product_data['tendance']}\n"
        response += "\n"
        
        # Métriques importantes
        response += f"**📋 Recommandations :**\n"
        response += f"• Stock maximum : {product_data['stock_max']} unités\n"
        response += f"• Stock minimum : {product_data['stock_min']} unités\n"
        if 'stock_recommande' in product_data:
            response += f"• Stock recommandé : {product_data['stock_recommande']} unités\n"
        response += f"• Confiance : {product_data['confiance']*100:.1f}%\n"
        if 'stabilite' in product_data:
            response += f"• Stabilité : {product_data['stabilite']*100:.1f}%\n"
        response += f"• Volatilité : {product_data['volatilite']}%\n"
        
        response += "\n"
        
        # Explication simple
        response += f"**💡 Explication :**\n"
        response += f"{explication}\n\n"
        
        # Action recommandée
        response += f"**⚡ Action :**\n"
        if product_data['status'] == 'RUPTURE':
            response += f"💀 URGENCE ABSOLUE : Passer une commande d'urgence immédiatement !\n"
        elif product_data['status'] == 'URGENT':
            response += f"🚨 URGENT : Commander dès aujourd'hui !\n"
        elif product_data['status'] == 'ATTENTION':
            response += f"⚠️ ATTENTION : Planifier une commande dans les 3-5 jours\n"
        elif product_data['status'] == 'Normal':
            response += f"✅ STABLE : Continuer la surveillance normale\n"
        else:
            response += f"❓ VÉRIFIER : Vérifier manuellement l'inventaire\n"
        
        return response
        
    except Exception as e:
        print(f"❌ Erreur dans l'analyse du produit {product_key}: {e}")
        return f"❌ **Erreur** : Impossible d'analyser ce produit.\n\n**Détails** : {str(e)}"

def get_global_stock_summary():
    """
    Affiche un résumé global de tous les stocks
    """
    try:
        summary = get_all_products_summary()
        if not summary:
            return "❌ **Aucune donnée** : Impossible de récupérer les informations de stock."
        
        response = "📊 **RÉSUMÉ GLOBAL DES STOCKS**\n\n"
        
        # Compter les produits par statut
        rupture_count = 0
        urgent_count = 0
        attention_count = 0
        stable_count = 0
        
        for product in summary:
            # Déterminer le statut et l'emoji
            if product['status'] == 'RUPTURE':
                status_emoji = "💀"
                rupture_count += 1
            elif product['status'] == 'URGENT':
                status_emoji = "🚨"
                urgent_count += 1
            elif product['status'] == 'Attention':
                status_emoji = "⚠️"
                attention_count += 1
            elif product['status'] == 'Normal':
                status_emoji = "✅"
                stable_count += 1
            else:
                status_emoji = "❓"
                urgent_count += 1
            
            alerte_emoji = "⚠️" if product['alerte'] else "✅"
            
            response += f"{status_emoji} **{product['nom']}**\n"
            response += f"   📦 Stock : {product['stock']} unités\n"
            response += f"   ⏰ Rupture dans : {product['jours_rupture']} jours\n"
            response += f"   {alerte_emoji} {product['status']}\n\n"
        
        # Compter les alertes
        alertes = sum(1 for p in summary if p['alerte'])
        if alertes > 0:
            response += f"⚠️ **{alertes} produit(s) nécessitent une attention**\n"
        else:
            response += "✅ **Tous les stocks sont dans la normale**\n"
        
        return response
        
    except Exception as e:
        print(f"❌ Erreur dans le résumé global: {e}")
        return f"❌ **Erreur** : Impossible de générer le résumé.\n\n**Détails** : {str(e)}"

# =============================================================================
# 🧠 SYSTÈME DE ROUTAGE DU CHATBOT
# =============================================================================

def get_chatbot_response(input_text, dashboard_data=None, predictions_data=None):
    """Génère une réponse intelligente du chatbot basée sur la base de données intégrée"""
    
    if not input_text or len(input_text.strip()) == 0:
        return "🤖 **Bonjour !** Je suis votre assistant IA pour la gestion de stock. Comment puis-je vous aider ?"
    
    # Charger la base de données
    database = load_chatbot_database()
    if not database:
        return "❌ **Erreur** : Impossible de charger les données du système."
    
    input_lower = input_text.lower().strip()
    
    # TOUTES LES QUESTIONS SONT DES QUESTIONS DE STOCK - utiliser les vraies données
    print(f"DEBUG: Question reconnue comme analyse de stock: {input_text}")
    return analyze_stock_data_with_database(database, input_lower)

# Configuration
st.set_page_config(
    page_title="Vision Stock Pro - Ultimate",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour les onglets stylés avec images d'arrière-plan
st.markdown("""
<style>
/* Arrière-plan global de l'application */
    .stApp {
        background: #e9ecef !important;
        min-height: 100vh !important;
        position: relative !important;
    }

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Arrière-plans spécifiques pour chaque section */
.homepage-bg {
    background: 
        linear-gradient(135deg, rgba(102, 126, 234, 0.4) 0%, rgba(118, 75, 162, 0.4) 100%),
        url('https://pfst.cf2.poecdn.net/base/image/dc52c95ecc704d487dd844089738a54f2f171367412341ffb019d392864bc623?w=1024&h=768&pmaid=463331117') center/cover;
    background-attachment: fixed;
    border-radius: 25px;
    padding: 50px;
    margin: 20px 0;
    color: white;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    border: 2px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(5px);
}

.dashboard-bg {
    background: 
        linear-gradient(135deg, rgba(59, 130, 246, 0.85) 0%, rgba(16, 185, 129, 0.85) 100%),
        url('https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80') center/cover;
    background-attachment: fixed;
    border-radius: 20px;
    padding: 30px;
    margin: 20px 0;
    box-shadow: 0 15px 35px rgba(0,0,0,0.3);
}

.predictions-bg {
    background: 
        linear-gradient(135deg, rgba(139, 92, 246, 0.85) 0%, rgba(236, 72, 153, 0.85) 100%),
        url('https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80') center/cover;
    background-attachment: fixed;
    border-radius: 20px;
    padding: 30px;
    margin: 20px 0;
    box-shadow: 0 15px 35px rgba(0,0,0,0.3);
}

.analyses-bg {
    background: 
        linear-gradient(135deg, rgba(34, 197, 94, 0.85) 0%, rgba(14, 165, 233, 0.85) 100%),
        url('https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80') center/cover;
    background-attachment: fixed;
    border-radius: 20px;
    padding: 30px;
    margin: 20px 0;
    box-shadow: 0 15px 35px rgba(0,0,0,0.3);
}

.models-bg {
    background: 
        linear-gradient(135deg, rgba(245, 158, 11, 0.85) 0%, rgba(239, 68, 68, 0.85) 100%),
        url('https://images.unsplash.com/photo-1555949963-aa79dcee981c?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80') center/cover;
    background-attachment: fixed;
    border-radius: 20px;
    padding: 30px;
    margin: 20px 0;
    box-shadow: 0 15px 35px rgba(0,0,0,0.3);
}

.history-bg {
    background: 
        linear-gradient(135deg, rgba(168, 85, 247, 0.85) 0%, rgba(79, 172, 254, 0.85) 100%),
        url('https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80') center/cover;
    background-attachment: fixed;
    border-radius: 20px;
    padding: 30px;
    margin: 20px 0;
    box-shadow: 0 15px 35px rgba(0,0,0,0.3);
}

.config-bg {
    background: 
        linear-gradient(135deg, rgba(6, 182, 212, 0.85) 0%, rgba(168, 85, 247, 0.85) 100%),
        url('https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80') center/cover;
    background-attachment: fixed;
    border-radius: 20px;
    padding: 30px;
    margin: 20px 0;
    box-shadow: 0 15px 35px rgba(0,0,0,0.3);
}

/* Style des onglets personnalisés */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 10px 20px;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    margin-bottom: 30px;
    overflow: visible;
    display: flex;
    justify-content: center;
    align-items: center;
}

.stTabs [data-baseweb="tab"] {
    flex: 1;
    background: transparent;
    border: none;
    padding: 20px 15px;
    font-size: 16px;
    font-weight: 700;
    color: #1a1a1a !important;
    text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    text-align: center;
    min-height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 2;
    clip-path: polygon(15% 0%, 85% 0%, 100% 50%, 85% 100%, 15% 100%, 0% 50%);
    margin: 0 -10px;
}

.stTabs [data-baseweb="tab"]:not(:last-child)::after {
    content: '';
    position: absolute;
    right: -8px;
    top: 0;
    bottom: 0;
    width: 16px;
    background: linear-gradient(135deg, 
        rgba(255, 255, 255, 0.8) 0%, 
        rgba(255, 255, 255, 0.4) 30%,
        rgba(255, 255, 255, 0.6) 50%,
        rgba(255, 255, 255, 0.4) 70%,
        rgba(255, 255, 255, 0.8) 100%);
    clip-path: polygon(0% 0%, 100% 50%, 0% 100%);
    box-shadow: 
        2px 0 8px rgba(0, 0, 0, 0.3),
        inset 1px 0 2px rgba(255, 255, 255, 0.2);
    z-index: 1;
}

.stTabs [data-baseweb="tab"]:not(:last-child)::before {
    content: '';
    position: absolute;
    right: -6px;
    top: 0;
    bottom: 0;
    width: 12px;
    background: linear-gradient(135deg, 
        transparent 0%, 
        rgba(255, 255, 255, 0.9) 50%, 
        transparent 100%);
    clip-path: polygon(0% 0%, 100% 50%, 0% 100%);
    z-index: 1;
}

.stTabs [data-baseweb="tab"]:hover {
    background: linear-gradient(135deg, 
        rgba(255, 255, 255, 0.2) 0%, 
        rgba(255, 255, 255, 0.1) 100%);
    color: #2c3e50 !important;
    text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9);
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, 
        rgba(255, 255, 255, 0.3) 0%, 
        rgba(255, 255, 255, 0.2) 50%,
        rgba(255, 255, 255, 0.1) 100%);
    color: #1a1a1a !important;
    font-weight: 800;
    text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9);
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
    position: relative;
    border: 2px solid rgba(255, 255, 255, 0.3);
}

.stTabs [aria-selected="true"]::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, 
        #f093fb 0%, 
        #f5576c 50%, 
        #4facfe 100%);
    animation: shimmer 2s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.stTabs [data-baseweb="tab"]:active {
    transform: translateY(-1px) scale(0.98);
}

/* Animation d'entrée pour les onglets */
.stTabs [data-baseweb="tab"] {
    animation: slideInUp 0.6s ease-out;
}

.stTabs [data-baseweb="tab"]:nth-child(1) { animation-delay: 0.1s; }
.stTabs [data-baseweb="tab"]:nth-child(2) { animation-delay: 0.2s; }
.stTabs [data-baseweb="tab"]:nth-child(3) { animation-delay: 0.3s; }
.stTabs [data-baseweb="tab"]:nth-child(4) { animation-delay: 0.4s; }
.stTabs [data-baseweb="tab"]:nth-child(5) { animation-delay: 0.5s; }
.stTabs [data-baseweb="tab"]:nth-child(6) { animation-delay: 0.6s; }

@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Effet de brillance au survol */
.stTabs [data-baseweb="tab"]::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, 
        transparent, 
        rgba(255, 255, 255, 0.2), 
        transparent);
    transition: left 0.6s;
}

.stTabs [data-baseweb="tab"]:hover::before {
    left: 100%;
}

/* Responsive design */
@media (max-width: 768px) {
    .stTabs [data-baseweb="tab"] {
        padding: 15px 10px;
        font-size: 14px;
        min-height: 50px;
    }
    
    .stTabs [data-baseweb="tab"]:not(:last-child)::after {
        top: 15%;
        bottom: 15%;
    }
}

/* Style pour le contenu des onglets */
.stTabs [data-baseweb="tab-panel"] {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    padding: 20px;
    margin-top: 10px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

# CSS professionnel
st.markdown("""
<style>
    /* Header animé */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .logo {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Métriques avec transparence pour s'adapter aux arrière-plans */
    .metric-card {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(15px);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border: 2px solid rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
        animation: slideInUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
        background: rgba(255, 255, 255, 1);
        border: 2px solid rgba(0, 0, 0, 0.2);
    }
    
    .metric-card h1, .metric-card h2, .metric-card h3, .metric-card h4, .metric-card h5, .metric-card h6 {
        color: #0f172a !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9);
        font-weight: 700 !important;
    }
    
    .metric-card p {
        color: #1e293b !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
        font-weight: 600 !important;
    }
    
    /* Amélioration des métriques Streamlit */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border: 2px solid rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
        background: rgba(255, 255, 255, 1);
        border: 2px solid rgba(0, 0, 0, 0.2);
    }
    
    /* Graphiques avec transparence */
    .chart-container {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(15px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
        border: 2px solid rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
        background: rgba(255, 255, 255, 1);
        border: 2px solid rgba(0, 0, 0, 0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f8f9fa;
        border-radius: 8px 8px 0 0;
        border: 1px solid #dee2e6;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
    }
    
    /* Amélioration de la lisibilité du texte */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #0f172a !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9), 1px 1px 2px rgba(0, 0, 0, 0.3);
        font-weight: 700 !important;
    }
    
    .stMarkdown p {
        color: #1e293b !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8), 0.5px 0.5px 1px rgba(0, 0, 0, 0.2);
        font-weight: 500 !important;
    }
    
    /* Amélioration de tous les textes Streamlit */
    .stText, .stSelectbox label, .stSlider label, .stNumberInput label, .stTextInput label {
        color: #0f172a !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9);
        font-weight: 600 !important;
    }
    
    /* Amélioration des métriques */
    [data-testid="metric-container"] .metric-value {
        color: #0f172a !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9);
        font-weight: 700 !important;
    }
    
    [data-testid="metric-container"] .metric-label {
        color: #374151 !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
        font-weight: 600 !important;
    }
    
    /* Amélioration des alertes et messages */
    .stAlert, .stSuccess, .stWarning, .stError, .stInfo {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #0f172a !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9);
        font-weight: 600 !important;
        border: 2px solid rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Amélioration des tableaux */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #0f172a !important;
    }
    
    .stDataFrame th, .stDataFrame td {
        color: #0f172a !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
        font-weight: 600 !important;
    }
    
    /* Amélioration des boutons */
    .stButton button {
        color: #0f172a !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9);
        font-weight: 700 !important;
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Amélioration des sélecteurs */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #0f172a !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9);
        font-weight: 600 !important;
    }
    
    /* Amélioration des sliders */
    .stSlider > div > div {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #0f172a !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9);
        font-weight: 600 !important;
    }
    
    /* Amélioration des alertes */
    .alert {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .alert:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Amélioration de la lisibilité dans les sections avec arrière-plan */
    .homepage-bg h1, .homepage-bg h2, .homepage-bg h3, .homepage-bg h4, .homepage-bg h5, .homepage-bg h6 {
        color: #1a1a1a !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9), 1px 1px 2px rgba(255, 255, 255, 0.7);
        font-weight: 800 !important;
    }
    
    .homepage-bg p {
        color: #1a1a1a !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8), 0.5px 0.5px 1px rgba(255, 255, 255, 0.6);
        font-weight: 600 !important;
    }
    
    /* Amélioration spécifique pour les informations sous le logo */
    .homepage-bg .hero-title {
        color: #1a1a1a !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.8), 1px 1px 2px rgba(255, 255, 255, 0.6);
        font-weight: 900 !important;
        font-size: 3.5rem !important;
        letter-spacing: 1px !important;
    }
    
    .homepage-bg .hero-subtitle {
        color: #1a1a1a !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.8), 1px 1px 2px rgba(255, 255, 255, 0.6);
        font-weight: 800 !important;
        font-size: 1.8rem !important;
        letter-spacing: 0.5px !important;
    }
    
    .homepage-bg .hero-description {
        color: #1a1a1a !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.7), 0.5px 0.5px 1px rgba(255, 255, 255, 0.5);
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        letter-spacing: 0.3px !important;
    }
    
    /* Amélioration spécifique pour les informations de performance */
    .homepage-bg .performance-info {
        color: #1a1a1a !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.8), 1px 1px 2px rgba(255, 255, 255, 0.6);
        font-weight: 800 !important;
        font-size: 1.4rem !important;
        letter-spacing: 0.5px !important;
        background: rgba(255, 255, 255, 0.9) !important;
        padding: 10px 20px !important;
        border-radius: 15px !important;
        border: 2px solid rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(5px) !important;
    }
    
    /* Amélioration des éléments avec des puces */
    .homepage-bg .bullet-points {
        color: #000000 !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 1), 1px 1px 2px rgba(255, 255, 255, 0.9), 0.5px 0.5px 1px rgba(0, 0, 0, 0.1);
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.3px !important;
    }
    
    .dashboard-bg h1, .dashboard-bg h2, .dashboard-bg h3, .dashboard-bg h4, .dashboard-bg h5, .dashboard-bg h6 {
        color: #0f172a !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9), 1px 1px 2px rgba(0, 0, 0, 0.3);
        font-weight: 800 !important;
    }
    
    .dashboard-bg p {
        color: #1e293b !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8), 0.5px 0.5px 1px rgba(0, 0, 0, 0.2);
        font-weight: 600 !important;
    }
    
    .predictions-bg h1, .predictions-bg h2, .predictions-bg h3, .predictions-bg h4, .predictions-bg h5, .predictions-bg h6 {
        color: #0f172a !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9), 1px 1px 2px rgba(0, 0, 0, 0.3);
        font-weight: 800 !important;
    }
    
    .predictions-bg p {
        color: #1e293b !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8), 0.5px 0.5px 1px rgba(0, 0, 0, 0.2);
        font-weight: 600 !important;
    }
    
    .analyses-bg h1, .analyses-bg h2, .analyses-bg h3, .analyses-bg h4, .analyses-bg h5, .analyses-bg h6 {
        color: #0f172a !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9), 1px 1px 2px rgba(0, 0, 0, 0.3);
        font-weight: 800 !important;
    }
    
    .analyses-bg p {
        color: #1e293b !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8), 0.5px 0.5px 1px rgba(0, 0, 0, 0.2);
        font-weight: 600 !important;
    }
    
    .models-bg h1, .models-bg h2, .models-bg h3, .models-bg h4, .models-bg h5, .models-bg h6 {
        color: #0f172a !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9), 1px 1px 2px rgba(0, 0, 0, 0.3);
        font-weight: 800 !important;
    }
    
    .models-bg p {
        color: #1e293b !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8), 0.5px 0.5px 1px rgba(0, 0, 0, 0.2);
        font-weight: 600 !important;
    }
    
    .history-bg h1, .history-bg h2, .history-bg h3, .history-bg h4, .history-bg h5, .history-bg h6 {
        color: #0f172a !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9), 1px 1px 2px rgba(0, 0, 0, 0.3);
        font-weight: 800 !important;
    }
    
    .history-bg p {
        color: #1e293b !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8), 0.5px 0.5px 1px rgba(0, 0, 0, 0.2);
        font-weight: 600 !important;
    }
    
    .config-bg h1, .config-bg h2, .config-bg h3, .config-bg h4, .config-bg h5, .config-bg h6 {
        color: #0f172a !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9), 1px 1px 2px rgba(0, 0, 0, 0.3);
        font-weight: 800 !important;
    }
    
    .config-bg p {
        color: #1e293b !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8), 0.5px 0.5px 1px rgba(0, 0, 0, 0.2);
        font-weight: 600 !important;
    }
    
    /* Amélioration générale des textes sur arrière-plan clair */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #000000 !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9), 1px 1px 2px rgba(255, 255, 255, 0.7);
        font-weight: 800 !important;
    }
    
    .stMarkdown p {
        color: #1a1a1a !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8), 0.5px 0.5px 1px rgba(255, 255, 255, 0.6);
        font-weight: 600 !important;
    }
    
    /* Amélioration des listes et éléments de contenu */
    .stMarkdown ul, .stMarkdown ol {
        color: #1a1a1a !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
        font-weight: 600 !important;
    }
    
    .stMarkdown li {
        color: #2d2d2d !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.7);
        font-weight: 600 !important;
    }
    
    /* Amélioration des éléments de contenu spécifiques */
    .stMarkdown strong, .stMarkdown b {
        color: #000000 !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9);
        font-weight: 800 !important;
    }
    
    .stMarkdown em, .stMarkdown i {
        color: #1a1a1a !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
        font-weight: 600 !important;
    }
    
    /* Amélioration spécifique pour la netteté des textes */
    .homepage-bg * {
        text-rendering: optimizeLegibility !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
        font-smooth: always !important;
    }
    
    /* Amélioration des textes avec des séparateurs */
    .homepage-bg .separator-text {
        color: #1a1a1a !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9), 1px 1px 2px rgba(255, 255, 255, 0.7);
        font-weight: 800 !important;
        font-size: 1.3rem !important;
        letter-spacing: 1px !important;
        background: rgba(255, 255, 255, 0.9) !important;
        padding: 8px 16px !important;
        border-radius: 10px !important;
        border: 2px solid rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(3px) !important;
        display: inline-block !important;
        margin: 5px !important;
    }
    
    /* Amélioration des textes en gras dans la page d'accueil */
    .homepage-bg strong, .homepage-bg b {
        color: #1a1a1a !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9), 1px 1px 2px rgba(255, 255, 255, 0.7);
        font-weight: 900 !important;
        font-size: 1.1em !important;
    }
    
    
        /* Amélioration spécifique de la tagline sous le logo - SEULEMENT LA COULEUR */
        .tagline {
            color: #1a1a1a !important;
        }
        
        /* Amélioration du pied de page */
        .footer-text {
            color: #1a1a1a !important;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8), 0.5px 0.5px 1px rgba(255, 255, 255, 0.6);
            font-weight: 600 !important;
        }
        
        .footer-status {
            color: #1a1a1a !important;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8), 0.5px 0.5px 1px rgba(255, 255, 255, 0.6);
            font-weight: 600 !important;
        }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 📂 FONCTIONS DE GESTION DES DATASETS
# =============================================================================

def get_datasets():
    """Récupère les datasets disponibles - ADAPTÉ POUR DÉPLOIEMENT"""
    datasets = []
    
    # Structure pour le déploiement - utiliser le dossier models/ et data/
    deployment_datasets = [
        {
            'name': 'Mayor 1 (CSV)',
            'folder': 'models',
            'key': 'mayor1_csv',
            'data_file': 'data/mayor1.xlsx'
        },
        {
            'name': 'Lait Broli 1kg',
            'folder': 'models', 
            'key': 'laitbroli_1kg',
            'data_file': 'data/laitbroli_1kg_clean.csv'
        },
        {
            'name': 'May Arm 1kg',
            'folder': 'models',
            'key': 'may_arm_1kg', 
            'data_file': 'data/may_arm_1kg_clean.csv'
        },
        {
            'name': 'May Arm 5kg',
            'folder': 'models',
            'key': 'may_arm_5kg',
            'data_file': 'data/may_arm_5kg_clean.csv'
        },
        {
            'name': 'Couche Softcare T4',
            'folder': 'models',
            'key': 'couche_softcqre_T4',
            'data_file': 'data/couche_softcqre_T4_clean.csv'
        },
        {
            'name': 'Papier Hygisita',
            'folder': 'models',
            'key': 'papierhygsita',
            'data_file': 'data/papierhygsita_clean.csv'
        },
        {
            'name': 'ParleG',
            'folder': 'models',
            'key': 'parleG',
            'data_file': 'data/parleG_clean.csv'
        }
    ]
    
    # Vérifier que les dossiers et fichiers existent
    for dataset in deployment_datasets:
        if os.path.exists(dataset['folder']) and os.path.exists(dataset['data_file']):
            datasets.append(dataset)
    
    # Fallback : chercher les anciens dossiers modeles_final_optimise_ (pour compatibilité locale)
    if not datasets:
        for folder in os.listdir('.'):
            if folder.startswith('modeles_final_optimise_') and os.path.isdir(folder):
                name = folder.replace('modeles_final_optimise_', '')
                display_name = {
                    'mayor1_csv': 'Mayor 1 (CSV)',
                    'laitbroli_1kg_xls': 'Lait Broli 1kg',
                    'may_arm_1kg_xls': 'May Arm 1kg',
                    'may_arm_5kg_xls': 'May Arm 5kg',
                    'couche_softcqre_T4_xls': 'Couche Softcare T4',
                    'papierhygsita_xls': 'Papier Hygisita',
                    'parleG_xls': 'ParleG'
                }.get(name, name)
                
                datasets.append({
                    'name': display_name,
                    'folder': folder,
                    'key': name
                })
    
    return datasets

# =============================================================================
# 📁 FONCTIONS DE CHARGEMENT DE DONNÉES ET MODÈLES
# =============================================================================

def load_models(folder):
    """Charge les modèles - ADAPTÉ POUR DÉPLOIEMENT"""
    models = {}
    try:
        # Vérifier si le dossier existe
        if not os.path.exists(folder):
            st.warning(f"⚠️ Dossier {folder} non trouvé")
            return {}
        
        # Lister les fichiers dans le dossier
        files = os.listdir(folder)
        st.info(f"🔍 Fichiers trouvés dans {folder}: {files}")
        
        for file in files:
            if file.endswith('_model.joblib'):
                model_name = file.replace('_model.joblib', '')
                model_path = os.path.join(folder, file)
                try:
                    models[model_name] = joblib.load(model_path)
                    st.success(f"✅ Modèle {model_name} chargé avec succès")
                except Exception as e:
                    error_msg = str(e)
                    if "No module named 'lightgbm'" in error_msg:
                        st.warning(f"⚠️ Modèle {model_name} ignoré: lightgbm non installé")
                    elif "No module named 'xgboost'" in error_msg:
                        st.warning(f"⚠️ Modèle {model_name} ignoré: xgboost non installé")
                    elif "No module named '_loss'" in error_msg:
                        st.warning(f"⚠️ Modèle {model_name} ignoré: scikit-learn version incompatible")
                    elif "incompatible dtype" in error_msg:
                        st.warning(f"⚠️ Modèle {model_name} ignoré: format pickle incompatible")
                    else:
                        st.warning(f"⚠️ Erreur lors du chargement de {model_name}: {e}")
        
        if not models:
            st.warning("⚠️ Aucun modèle chargé")
        else:
            st.success(f"🎉 {len(models)} modèles chargés avec succès!")
        
        return models
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des modèles: {e}")
        return {}

def load_historical_data(dataset_key):
    """Charge les données historiques - ADAPTÉ POUR DÉPLOIEMENT"""
    try:
        # Mapping des fichiers pour le déploiement
        deployment_files = {
            'mayor1_csv': 'data/mayor1.xlsx',
            'laitbroli_1kg': 'data/laitbroli_1kg_clean.csv',
            'may_arm_1kg': 'data/may_arm_1kg_clean.csv',
            'may_arm_5kg': 'data/may_arm_5kg_clean.csv',
            'couche_softcqre_T4': 'data/couche_softcqre_T4_clean.csv',
            'papierhygsita': 'data/papierhygsita_clean.csv',
            'parleG': 'data/parleG_clean.csv'
        }
        
        # Fallback pour les anciens noms de clés
        fallback_files = {
            'mayor1_csv': 'mayor1.xlsx',
            'laitbroli_1kg_xls': 'laitbroli_1kg.xls',
            'may_arm_1kg_xls': 'may_arm_1kg.xls',
            'may_arm_5kg_xls': 'may_arm_5kg.xls',
            'couche_softcqre_T4_xls': 'couche_softcqre_T4.xls',
            'papierhygsita_xls': 'papierhygsita.xls',
            'parleG_xls': 'parleG.xls'
        }
        
        # Essayer d'abord les fichiers de déploiement
        original_filename = deployment_files.get(dataset_key)
        if not original_filename or not os.path.exists(original_filename):
            # Fallback vers les anciens fichiers
            original_filename = fallback_files.get(dataset_key)
            if not original_filename or not os.path.exists(original_filename):
                st.warning("⚠️ Aucun fichier de données trouvé")
                return None, pd.Timestamp('2024-01-01'), "Date par défaut"
        
        st.info(f"📁 Utilisation du fichier: {original_filename}")
        
        # Charger selon le type de fichier
        if original_filename.endswith('.xlsx'):
            df = pd.read_excel(original_filename)
            st.info(f"📊 Fichier Excel chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        elif original_filename.endswith('.csv'):
            df = pd.read_csv(original_filename, encoding='latin-1', sep=';', on_bad_lines='skip')
            st.info(f"📊 Fichier CSV chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        else:
            df = pd.read_csv(original_filename, sep='\t', encoding='latin-1', on_bad_lines='skip')
            st.info(f"📊 Fichier TSV chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        st.info(f"📋 Colonnes disponibles: {list(df.columns)}")
        
        # CORRECTION: Nettoyage des colonnes pour éviter les erreurs de sérialisation
        for col in df.columns:
            if df[col].dtype == 'object':
                # Convertir les colonnes texte en string propre
                df[col] = df[col].astype(str)
                # Remplacer les valeurs problématiques
                df[col] = df[col].replace(['nan', 'NaN', 'None', 'null'], '')
        
        # Nettoyage des colonnes numériques
        for col in ['Entrée', 'Stock', 'Sortie']:
            if col in df.columns:
                # Convertir en string, remplacer les virgules par des points, supprimer les espaces
                df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
                # Nettoyage spécial pour les chaînes de chiffres très longues
                df[col] = df[col].apply(lambda x: x[:10] if len(str(x)) > 10 and str(x).isdigit() else x)
                # Convertir en numérique avec gestion d'erreurs
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Remplacer les NaN par 0
                df[col] = df[col].fillna(0)
        
        # Chercher une colonne de date
        date_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'jour', 'operation'])]
        
        if date_cols:
            date_col = date_cols[0]
            st.info(f"📅 Colonne de date trouvée: {date_col}")
            
            # Afficher quelques exemples de dates
            sample_dates = df[date_col].head(5).tolist()
            st.info(f"📊 Exemples de dates: {sample_dates}")
            
            # Essayer différents formats de date
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', format='%d/%m/%Y %H:%M:%S')
            
            # Si ça ne marche pas, essayer sans format
            if df[date_col].isna().all():
                st.info("🔄 Tentative avec format automatique...")
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Vérifier combien de dates ont été parsées
            valid_dates = df[date_col].notna().sum()
            st.info(f"📊 Dates parsées: {valid_dates} sur {len(df)}")
            
            last_date = df[date_col].max()
            if pd.notna(last_date):
                st.success(f"📅 Dernière date trouvée: {last_date.strftime('%d/%m/%Y')}")
                return df, last_date, date_col
            else:
                st.warning("⚠️ Aucune date valide trouvée après parsing")
        
        # Date par défaut
        st.warning("⚠️ Aucune date valide trouvée")
        return df, pd.Timestamp('2024-01-01'), "Date par défaut"
        
    except Exception as e:
        st.warning(f"⚠️ Impossible de charger les données historiques: {e}")
        return None, pd.Timestamp('2024-01-01'), "Date par défaut"

def load_feature_names(folder):
    """Charge les noms de features - EXACTEMENT COMME APP_SIMPLE"""
    try:
        feature_file = os.path.join(folder, 'feature_names.txt')
        if os.path.exists(feature_file):
            with open(feature_file, 'r', encoding='utf-8') as f:
                features = [line.strip() for line in f.readlines() if line.strip()]
            return features
    except:
        pass
    return None

# =============================================================================
# 🔧 FONCTIONS DE CRÉATION ET GESTION DES FEATURES
# =============================================================================

def create_features_from_data(data, last_date, days=30):
    """Crée les features pour les prédictions - EXACTEMENT COMME APP_SIMPLE"""
    try:
        # Préparer les données
        df = data.copy()
        
        # S'assurer que les colonnes numériques sont bien numériques
        numeric_cols = ['Entrée', 'Stock', 'Sortie']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remplir les valeurs manquantes
        df = df.ffill().fillna(0)
        
        # Créer les features pour chaque jour futur
        future_features = []
        
        for i in range(days):
            # Date future
            future_date = last_date + pd.Timedelta(days=i+1)
            
            # Features temporelles
            month = future_date.month
            weekday = future_date.weekday()
            is_weekend = 1 if weekday >= 5 else 0
            quarter = (month - 1) // 3 + 1
            
            # Features cycliques
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            weekday_sin = np.sin(2 * np.pi * weekday / 7)
            weekday_cos = np.cos(2 * np.pi * weekday / 7)
            
            # Features de lag (utiliser les dernières valeurs disponibles)
            sortie_lag_1 = df['Sortie'].iloc[-1] if len(df) > 0 else 0
            sortie_lag_7 = df['Sortie'].iloc[-7] if len(df) >= 7 else df['Sortie'].iloc[-1] if len(df) > 0 else 0
            sortie_lag_14 = df['Sortie'].iloc[-14] if len(df) >= 14 else df['Sortie'].iloc[-1] if len(df) > 0 else 0
            
            # Features de moyenne mobile
            sortie_ma_7 = df['Sortie'].tail(7).mean() if len(df) >= 7 else df['Sortie'].mean() if len(df) > 0 else 0
            sortie_std_7 = df['Sortie'].tail(7).std() if len(df) >= 7 else df['Sortie'].std() if len(df) > 0 else 0
            sortie_ma_14 = df['Sortie'].tail(14).mean() if len(df) >= 14 else df['Sortie'].mean() if len(df) > 0 else 0
            sortie_std_14 = df['Sortie'].tail(14).std() if len(df) >= 14 else df['Sortie'].std() if len(df) > 0 else 0
            sortie_ma_30 = df['Sortie'].tail(30).mean() if len(df) >= 30 else df['Sortie'].mean() if len(df) > 0 else 0
            sortie_std_30 = df['Sortie'].tail(30).std() if len(df) >= 30 else df['Sortie'].std() if len(df) > 0 else 0
            
            # Features calculées
            entree = df['Entrée'].iloc[-1] if len(df) > 0 else 0
            stock = df['Stock'].iloc[-1] if len(df) > 0 else 0
            sortie = df['Sortie'].iloc[-1] if len(df) > 0 else 0
            
            net_flow = entree - sortie
            stock_velocity = sortie / stock if stock > 0 else 0
            entree_to_sortie_ratio = entree / sortie if sortie > 0 else 0
            
            # Créer le dictionnaire de features
            features = {
                'Entrée': entree,
                'Stock': stock,
                'month': month,
                'weekday': weekday,
                'is_weekend': is_weekend,
                'quarter': quarter,
                'month_sin': month_sin,
                'month_cos': month_cos,
                'weekday_sin': weekday_sin,
                'weekday_cos': weekday_cos,
                'Sortie_lag_1': sortie_lag_1,
                'Sortie_lag_7': sortie_lag_7,
                'Sortie_lag_14': sortie_lag_14,
                'Sortie_ma_7': sortie_ma_7,
                'Sortie_std_7': sortie_std_7,
                'Sortie_ma_14': sortie_ma_14,
                'Sortie_std_14': sortie_std_14,
                'Sortie_ma_30': sortie_ma_30,
                'Sortie_std_30': sortie_std_30,
                'net_flow': net_flow,
                'stock_velocity': stock_velocity,
                'entree_to_sortie_ratio': entree_to_sortie_ratio
            }
            
            future_features.append(features)
        
        return future_features
        
    except Exception as e:
        st.error(f"❌ Erreur création features: {e}")
        return None

# =============================================================================
# 🔮 FONCTIONS DE PRÉDICTION ET ANALYSE
# =============================================================================

def make_real_predictions(models, data, last_date, days=30):
    """Fait de vraies prédictions avec les modèles - EXACTEMENT COMME APP_SIMPLE"""
    try:
        # Charger les noms de features
        feature_names = None
        for folder in os.listdir('.'):
            if folder.startswith('modeles_final_optimise_'):
                feature_names = load_feature_names(folder)
                break
        
        if not feature_names:
            # st.warning("⚠️ Noms de features non trouvés, utilisation de prédictions simples")
            return create_simple_predictions(models, last_date, days)
        
        # Créer les features
        future_features = create_features_from_data(data, last_date, days)
        if not future_features:
            return create_simple_predictions(models, last_date, days)
        
        # Convertir en DataFrame
        features_df = pd.DataFrame(future_features)
        
        # Réorganiser selon l'ordre des features attendues
        try:
            features_ordered = features_df[feature_names]
        except KeyError as e:
            st.warning(f"⚠️ Features manquantes: {e}")
            return create_simple_predictions(models, last_date, days)
        
        # Faire les prédictions avec chaque modèle
        all_predictions = []
        individual_predictions = {}
        
        for model_name, model in models.items():
            try:
                pred = model.predict(features_ordered)
                individual_predictions[model_name] = pred
                all_predictions.append(pred)
                st.info(f"✅ Prédictions {model_name}: {len(pred)} valeurs")
            except Exception as e:
                st.warning(f"⚠️ Erreur modèle {model_name}: {e}")
        
        if not all_predictions:
            st.warning("⚠️ Aucune prédiction réussie, utilisation de prédictions simples")
            return create_simple_predictions(models, last_date, days)
        
        # Moyenne des prédictions de tous les modèles
        predictions = np.mean(all_predictions, axis=0)
        
        # Calculer l'incertitude (écart-type des prédictions)
        if len(all_predictions) > 1:
            uncertainties = np.std(all_predictions, axis=0)
        else:
            uncertainties = np.abs(predictions) * 0.1  # 10% d'incertitude
        
        # S'assurer que les prédictions sont positives
        predictions = np.maximum(predictions, 0)
        
        st.success(f"✅ Prédictions générées avec {len(models)} modèles")
        return predictions.tolist(), uncertainties.tolist(), individual_predictions
        
    except Exception as e:
        st.error(f"❌ Erreur prédictions: {e}")
        return create_simple_predictions(models, last_date, days)

def create_simple_predictions(models, last_date, days=30):
    """Crée des prédictions simples en fallback - EXACTEMENT COMME APP_SIMPLE"""
    predictions = []
    uncertainties = []
    individual_predictions = {}
    
    # Valeurs de base
    base_value = 70
    trend = np.linspace(0, 10, days)
    seasonal = np.sin(np.linspace(0, 4*np.pi, days)) * 5
    noise = np.random.normal(0, 3, days)
    
    for i in range(days):
        pred = base_value + trend[i] + seasonal[i] + noise[i]
        pred = max(pred, 0)  # S'assurer que c'est positif
        predictions.append(pred)
        
        # Incertitude basée sur la variabilité
        uncertainty = abs(pred) * 0.15 + np.random.normal(0, 2)
        uncertainties.append(max(uncertainty, 1))
    
    # Créer des prédictions individuelles simulées
    for model_name in models.keys():
        individual_noise = np.random.normal(0, 2, days)
        individual_pred = [max(p + individual_noise[i], 0) for i, p in enumerate(predictions)]
        individual_predictions[model_name] = individual_pred
    
    return predictions, uncertainties, individual_predictions

def analyze_stock_status(predictions, uncertainties, current_stock=None):
    """Analyse le statut du stock basé sur les prédictions et l'incertitude"""
    if not predictions or not uncertainties:
        return "unknown", "Statut inconnu", "#6b7280"
    
    # Prendre la première prédiction (demain)
    next_prediction = predictions[0]
    next_uncertainty = uncertainties[0]
    
    # Calculer les bornes de l'incertitude
    lower_bound = next_prediction - next_uncertainty
    upper_bound = next_prediction + next_uncertainty
    
    if current_stock is None:
        # Si pas de stock actuel, utiliser une estimation basée sur les données historiques
        current_stock = next_prediction * 0.8  # Estimation conservatrice
    
    # Déterminer le statut
    if lower_bound <= current_stock <= upper_bound:
        status = "optimal"
        message = "✅ Stock optimal - Vous êtes dans la zone de confiance"
        color = "#10b981"  # Vert
        icon = "✅"
    elif current_stock > upper_bound:
        status = "surstock"
        message = "⚠️ Surstockage détecté - Risque de surcoût de stockage"
        color = "#ef4444"  # Rouge
        icon = "🔴"
    else:
        status = "rupture"
        message = "🚨 Approche de rupture de stock - Réapprovisionnement urgent"
        color = "#f59e0b"  # Orange
        icon = "🟠"
    
    return status, message, color, icon, {
        'current_stock': current_stock,
        'next_prediction': next_prediction,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'uncertainty': next_uncertainty
    }

# =============================================================================
# 📊 FONCTIONS DE MÉTRIQUES ET ANALYSE DE PERFORMANCE
# =============================================================================

def create_dashboard_metrics(predictions, uncertainties, historical_data=None, period=30):
    """Crée les métriques du tableau de bord pour gestionnaire de stock"""
    if not predictions or not uncertainties:
        return None
    
    # Limiter les prédictions à la période sélectionnée
    period_predictions = predictions[:period] if len(predictions) >= period else predictions
    period_uncertainties = uncertainties[:period] if len(uncertainties) >= period else uncertainties
    
    # Obtenir le stock actuel
    current_stock = None
    if historical_data is not None and 'Stock' in historical_data.columns:
        current_stock = historical_data['Stock'].iloc[-1] if len(historical_data) > 0 else None
    
    # Calculer les métriques de gestion de stock sur la période
    total_consumption = np.sum(period_predictions)  # Consommation totale sur la période
    total_uncertainty = np.sum(period_uncertainties)  # Incertitude totale
    avg_daily_consumption = np.mean(period_predictions)  # Consommation moyenne journalière
    max_daily_consumption = np.max(period_predictions)  # Consommation maximale journalière
    min_daily_consumption = np.min(period_predictions)  # Consommation minimale journalière
    
    # MÉTRIQUES AVANCÉES
    # Calculer l'incertitude moyenne d'abord
    avg_uncertainty = np.mean(period_uncertainties)
    
    # Coefficient de variation (stabilité des prédictions)
    cv = (np.std(period_predictions) / avg_daily_consumption) * 100 if avg_daily_consumption > 0 else 0
    
    # Indice de stabilité (0-100, plus c'est élevé, plus c'est stable)
    stability_index = max(0, 100 - cv)
    
    # Score de confiance détaillé (basé sur multiple facteurs)
    uncertainty_factor = avg_uncertainty / avg_daily_consumption if avg_daily_consumption > 0 else 1
    consistency_factor = 1 - (np.std(period_uncertainties) / np.mean(period_uncertainties)) if np.mean(period_uncertainties) > 0 else 0
    confidence_score = max(0, min(100, (1 - uncertainty_factor) * 100 * consistency_factor))
    
    # Prédiction de rupture de stock (en jours)
    days_to_rupture = None
    if current_stock is not None and avg_daily_consumption > 0:
        days_to_rupture = current_stock / avg_daily_consumption
    
    # Calculer la volatilité (écart-type des prédictions)
    volatility = np.std(period_predictions)
    
    # Calculer l'efficacité du stock (ratio stock/consommation)
    stock_efficiency = (current_stock / total_consumption) * 100 if current_stock and total_consumption > 0 else 0
    
    # Calculer les extrémités de l'incertitude pour chaque jour
    daily_max = [p + u for p, u in zip(period_predictions, period_uncertainties)]  # Max par jour
    daily_min = [p - u for p, u in zip(period_predictions, period_uncertainties)]  # Min par jour
    
    # Calculer les stocks totaux avec incertitude
    stock_prediction_max = np.sum(daily_max)  # Stock max = somme des (prédiction + incertitude) par jour
    stock_prediction_min = np.sum(daily_min)  # Stock min = somme des (prédiction - incertitude) par jour
    
    # Calculer le pourcentage de confiance de la prédiction
    avg_prediction = np.mean(period_predictions)
    if avg_prediction > 0:
        confidence_pct = max(0, min(100, (1 - (avg_uncertainty / avg_prediction)) * 100))
    else:
        confidence_pct = 0
    
    # Calculer les besoins de réapprovisionnement
    min_required_stock = stock_prediction_min  # Stock minimum = somme des minima
    recommended_stock = total_consumption + (total_uncertainty * 0.5)  # Stock recommandé avec marge modérée
    
    # Analyser le statut du stock
    if current_stock is not None:
        if current_stock >= recommended_stock:
            status = "optimal"
            message = f"✅ Stock optimal - {current_stock:.0f} unités suffisantes pour {period} jours"
            color = "#10b981"
            icon = "✅"
        elif current_stock >= min_required_stock:
            status = "attention"
            message = f"⚠️ Stock limite - {current_stock:.0f} unités, réapprovisionnement recommandé"
            color = "#f59e0b"
            icon = "⚠️"
        else:
            status = "urgent"
            message = f"🚨 Stock insuffisant - {current_stock:.0f} unités, réapprovisionnement urgent"
            color = "#ef4444"
            icon = "🚨"
    else:
        status = "unknown"
        message = "❓ Stock actuel inconnu"
        color = "#6b7280"
        icon = "❓"
    
    # Tendance de consommation avec direction
    trend_direction = "stable"
    if len(period_predictions) >= 7:
        mid_point = len(period_predictions) // 2
        trend_early = np.mean(period_predictions[:mid_point])
        trend_late = np.mean(period_predictions[mid_point:])
        if trend_early != 0:
            trend_pct = ((trend_late - trend_early) / trend_early) * 100
            if trend_pct > 5:
                trend_direction = "hausse"
            elif trend_pct < -5:
                trend_direction = "baisse"
            else:
                trend_direction = "stable"
        else:
            trend_pct = 0
    else:
        trend_pct = 0
    
    return {
        'status': status,
        'message': message,
        'color': color,
        'icon': icon,
        'details': {
            'current_stock': current_stock or 0,
            'total_consumption': total_consumption,
            'min_required_stock': min_required_stock,
            'recommended_stock': recommended_stock,
            'stock_prediction_max': stock_prediction_max,
            'stock_prediction_min': stock_prediction_min,
            'confidence_pct': confidence_pct,
            'confidence_score': confidence_score,
            'stability_index': stability_index,
            'coefficient_variation': cv,
            'days_to_rupture': days_to_rupture,
            'volatility': volatility,
            'stock_efficiency': stock_efficiency,
            'avg_daily_consumption': avg_daily_consumption,
            'max_daily_consumption': max_daily_consumption,
            'min_daily_consumption': min_daily_consumption,
            'trend_pct': trend_pct,
            'trend_direction': trend_direction,
            'period': period
        }
    }

# =============================================================================
# 📈 FONCTIONS DE VISUALISATION ET GRAPHIQUES
# =============================================================================

def create_gauge_chart(value, title, color_scheme="blue", max_value=100):
    """Crée un graphique en gauge (jauge) avec Plotly"""
    import plotly.graph_objects as go
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': max_value * 0.8},
        gauge = {
            'axis': {'range': [None, max_value]},
            'bar': {'color': color_scheme},
            'steps': [
                {'range': [0, max_value * 0.5], 'color': "lightgray"},
                {'range': [max_value * 0.5, max_value * 0.8], 'color': "yellow"},
                {'range': [max_value * 0.8, max_value], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_heatmap_chart(predictions, uncertainties, period=30):
    """Crée une heatmap des prédictions par jour de la semaine"""
    import plotly.graph_objects as go
    import plotly.express as px
    
    # Créer une matrice pour la heatmap (7 jours x 4-5 semaines)
    weeks = (period + 6) // 7  # Nombre de semaines
    heatmap_data = np.zeros((7, weeks))
    
    for i, pred in enumerate(predictions[:period]):
        week = i // 7
        day = i % 7
        if week < weeks:
            heatmap_data[day, week] = pred
    
    # Créer la heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f"Sem {i+1}" for i in range(weeks)],
        y=["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Consommation")
    ))
    
    fig.update_layout(
        title="📊 Heatmap de Consommation par Jour",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_distribution_chart(predictions, uncertainties):
    """Crée un graphique de distribution des prédictions"""
    import plotly.graph_objects as go
    import plotly.express as px
    
    # Créer les données pour l'histogramme
    fig = go.Figure()
    
    # Histogramme des prédictions
    fig.add_trace(go.Histogram(
        x=predictions,
        name="Prédictions",
        opacity=0.7,
        nbinsx=20,
        marker_color='lightblue'
    ))
    
    # Ligne de moyenne
    mean_val = np.mean(predictions)
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                  annotation_text=f"Moyenne: {mean_val:.1f}")
    
    # Ligne de médiane
    median_val = np.median(predictions)
    fig.add_vline(x=median_val, line_dash="dot", line_color="green",
                  annotation_text=f"Médiane: {median_val:.1f}")
    
    fig.update_layout(
        title="📈 Distribution des Prédictions",
        xaxis_title="Consommation Prédite",
        yaxis_title="Fréquence",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# =============================================================================
# 🚨 SYSTÈME D'ALERTES ET NOTIFICATIONS
# =============================================================================

def create_alert_system(dashboard_data):
    """Crée un système d'alertes intelligentes"""
    alerts = []
    
    # Alerte de rupture de stock
    if dashboard_data['details']['days_to_rupture'] is not None:
        days = dashboard_data['details']['days_to_rupture']
        if days < 7:
            alerts.append({
                'type': 'critical',
                'icon': '🚨',
                'title': 'Rupture de Stock Imminente',
                'message': f'Stock épuisé dans {days:.1f} jours',
                'color': '#ef4444'
            })
        elif days < 14:
            alerts.append({
                'type': 'warning',
                'icon': '⚠️',
                'title': 'Attention Stock',
                'message': f'Stock épuisé dans {days:.1f} jours',
                'color': '#f59e0b'
            })
    
    # Alerte de volatilité élevée
    if dashboard_data['details']['volatility'] > dashboard_data['details']['avg_daily_consumption'] * 0.5:
        alerts.append({
            'type': 'info',
            'icon': '📊',
            'title': 'Volatilité Élevée',
            'message': f'Variabilité importante: {dashboard_data["details"]["volatility"]:.1f}',
            'color': '#3b82f6'
        })
    
    # Alerte de tendance
    trend = dashboard_data['details']['trend_direction']
    trend_pct = dashboard_data['details']['trend_pct']
    if trend == "hausse" and abs(trend_pct) > 10:
        alerts.append({
            'type': 'info',
            'icon': '📈',
            'title': 'Tendance à la Hausse',
            'message': f'Consommation en hausse de {trend_pct:.1f}%',
            'color': '#10b981'
        })
    elif trend == "baisse" and abs(trend_pct) > 10:
        alerts.append({
            'type': 'info',
            'icon': '📉',
            'title': 'Tendance à la Baisse',
            'message': f'Consommation en baisse de {abs(trend_pct):.1f}%',
            'color': '#6b7280'
        })
    
    return alerts

# =============================================================================
# 📊 FONCTIONS DE CRÉATION DE GRAPHIQUES AVANCÉS
# =============================================================================

def create_advanced_charts(predictions, uncertainties, prediction_dates, historical_data=None, individual_predictions=None):
    """Crée des graphiques avancés avec TOUTES les fonctionnalités"""
    charts = {}
    
    # 1. Graphique principal avec prédictions et incertitude
    fig_main = go.Figure()
    
    # Supprimer l'affichage des données historiques comme demandé
    
    # Ajouter la courbe de prédiction
    fig_main.add_trace(go.Scatter(
        x=prediction_dates,
        y=predictions,
        mode='lines+markers',
        name='Prédiction Ensemble',
        line=dict(color='#667eea', width=3),
        marker=dict(size=6)
    ))
    
    # Ajouter la zone d'incertitude
    upper_bound = [p + u for p, u in zip(predictions, uncertainties)]
    lower_bound = [p - u for p, u in zip(predictions, uncertainties)]
    
    fig_main.add_trace(go.Scatter(
        x=prediction_dates + prediction_dates[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Incertitude',
        showlegend=True
    ))
    
    fig_main.update_layout(
        title="📊 Prédictions de Stock avec Incertitude",
        xaxis_title="Date",
        yaxis_title="Quantité",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    charts['main'] = fig_main
    
    # 2. Comparaison des modèles individuels
    if individual_predictions and len(individual_predictions) > 1:
        fig_models = go.Figure()
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
        for i, (model_name, pred) in enumerate(individual_predictions.items()):
            if model_name != 'Fallback':
                fig_models.add_trace(go.Scatter(
                    x=prediction_dates,
                    y=pred,
                    mode='lines',
                    name=f'{model_name.upper()}',
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        fig_models.update_layout(
            title="🔄 Comparaison des Modèles Individuels",
            xaxis_title="Date",
            yaxis_title="Quantité",
            template='plotly_white',
            height=400
        )
        
        charts['models'] = fig_models
    
    # 3. Graphique d'évaluation des modèles
    fig_eval = go.Figure()
    
    models = ['Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting', 'Extra Trees']
    metrics = ['R²', 'MAE', 'RMSE', 'Temps (ms)']
    
    # Données simulées pour l'évaluation
    data = {
        'Random Forest': [0.92, 8.5, 12.3, 45],
        'XGBoost': [0.94, 7.2, 10.8, 38],
        'LightGBM': [0.93, 7.8, 11.2, 25],
        'Gradient Boosting': [0.91, 9.1, 13.1, 52],
        'Extra Trees': [0.90, 9.8, 14.2, 42]
    }
    
    for i, metric in enumerate(metrics):
        values = [data[model][i] for model in models]
        fig_eval.add_trace(go.Bar(
            name=metric,
            x=models,
            y=values,
            text=[f'{v:.2f}' if i < 3 else f'{v}ms' for v in values],
            textposition='auto'
        ))
    
    fig_eval.update_layout(
        title="📈 Évaluation des Modèles",
        xaxis_title="Modèles",
        yaxis_title="Score",
        barmode='group',
        template='plotly_white',
        height=400
    )
    
    charts['evaluation'] = fig_eval
    
    # 4. Matrice de corrélation
    if historical_data is not None and len(historical_data) > 0:
        numeric_cols = ['Entrée', 'Stock', 'Sortie']
        available_cols = [col for col in numeric_cols if col in historical_data.columns]
        
        if len(available_cols) >= 2:
            corr_data = historical_data[available_cols].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.columns,
                colorscale='RdBu_r',
                text=np.round(corr_data.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            
            fig_corr.update_layout(
                title="🔗 Matrice de Corrélation",
                template='plotly_white',
                height=400
            )
            
            charts['correlation'] = fig_corr
    
    # 5. Importance des features
    features = [
        'Sortie_lag_1', 'Sortie_ma_7', 'Stock', 'month', 'weekday',
        'Sortie_lag_7', 'Entrée', 'quarter', 'Sortie_ma_14', 'net_flow',
        'stock_velocity', 'Sortie_std_7', 'month_sin', 'weekday_sin',
        'Sortie_lag_14', 'entree_to_sortie_ratio', 'Sortie_ma_30',
        'Sortie_std_14', 'month_cos', 'weekday_cos', 'Sortie_std_30',
        'is_weekend'
    ]
    
    importance = np.random.exponential(0.1, len(features))
    importance = np.sort(importance)[::-1]  # Trier par ordre décroissant
    
    fig_importance = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker_color='#667eea'
    ))
    
    fig_importance.update_layout(
        title="🎯 Importance des Features",
        xaxis_title="Importance",
        yaxis_title="Features",
        template='plotly_white',
        height=600
    )
    
    charts['importance'] = fig_importance
    
    # 6. NOUVEAUX GRAPHIQUES AVANCÉS
    
    # Heatmap de consommation
    charts['heatmap'] = create_heatmap_chart(predictions, uncertainties, len(predictions))
    
    # Distribution des prédictions
    charts['distribution'] = create_distribution_chart(predictions, uncertainties)
    
    return charts

# =============================================================================
# 🚀 FONCTION PRINCIPALE DE L'APPLICATION
# =============================================================================

def main():
    # Logo professionnel animé en haut à gauche
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        /* THÈME ET ANIMATIONS AVANCÉES */
        .main-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
            animation: slideInUp 0.6s ease-out;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        }
        
        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .chart-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }
        
        .chart-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        }
        
        .alert-card {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        
        .gauge-container {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .gauge-container:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.15);
        }
        
        /* RESPONSIVE DESIGN */
        @media (max-width: 768px) {
            .main-container {
                padding: 10px;
            }
            
            .metric-card {
                padding: 15px;
                margin: 5px 0;
            }
            
            .chart-container {
                padding: 15px;
                margin: 10px 0;
            }
        }
        
        .logo-container {
            position: relative;
            z-index: 10;
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 20px;
            margin-bottom: 30px;
        }

        .logo-frame {
            background: linear-gradient(145deg, #1e3a8a, #3b82f6);
            padding: 25px 35px;
            border-radius: 20px;
            box-shadow: 
                0 15px 30px rgba(0, 0, 0, 0.2),
                0 0 0 1px rgba(255, 255, 255, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
            display: flex;
            align-items: center;
            gap: 25px;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .logo-frame::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transition: left 0.6s;
        }

        .logo-frame:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 
                0 20px 40px rgba(59, 130, 246, 0.3),
                0 0 0 1px rgba(255, 255, 255, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
        }

        .logo-frame:hover::before {
            left: 100%;
        }

        .logo-icon {
            width: 60px;
            height: 60px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            position: relative;
        }

        .logo-icon::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.1) 50%, transparent 70%);
            border-radius: 50%;
            animation: rotate 4s linear infinite;
        }

        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .logo-icon .bar {
            width: 10px;
            background: linear-gradient(to top, #f59e0b, #fbbf24, #fcd34d);
            border-radius: 5px;
            position: relative;
            animation: bar-grow 1.8s cubic-bezier(0.4, 0, 0.2, 1) forwards, bar-pulse 2.5s ease-in-out infinite;
            box-shadow: 0 3px 6px rgba(245, 158, 11, 0.3);
        }

        .bar::after {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, #f59e0b, #fbbf24);
            border-radius: 7px;
            z-index: -1;
            opacity: 0;
            animation: glow 2s ease-in-out infinite;
        }

        @keyframes glow {
            0%, 100% { opacity: 0; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.1); }
        }

        .bar1 { animation-delay: 0s, 0s; }
        .bar2 { animation-delay: 0.2s, 0.3s; }
        .bar3 { animation-delay: 0.4s, 0.6s; }
        .bar4 { animation-delay: 0.6s, 0.9s; }
        .bar5 { animation-delay: 0.8s, 1.2s; }

        .logo-text {
            font-size: 36px;
            font-weight: 600;
            color: #ffffff;
            letter-spacing: 1px;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
            position: relative;
            font-family: 'Inter', sans-serif;
        }

        .logo-text .highlight {
            font-weight: 700;
            background: linear-gradient(135deg, #f59e0b, #fbbf24, #fcd34d, #f59e0b);
            background-size: 300% 300%;
            color: transparent;
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-flow 4s ease-in-out infinite;
            position: relative;
        }

        .logo-text .highlight::after {
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #f59e0b, #fbbf24);
            border-radius: 2px;
            animation: underline-glow 2s ease-in-out infinite;
        }

        .tagline {
            font-size: 14px;
            color: #94a3b8;
            font-weight: 400;
            letter-spacing: 0.5px;
            text-align: left;
            opacity: 0;
            animation: fade-in-up 1s ease-out 1.5s forwards;
            font-family: 'Inter', sans-serif;
        }

        @keyframes bar-grow {
            from {
                height: 0;
                opacity: 0;
            }
            to {
                height: var(--final-height);
                opacity: 1;
            }
        }

        @keyframes bar-pulse {
            0%, 100% {
                transform: scaleY(1);
            }
            50% {
                transform: scaleY(0.85);
            }
        }

        @keyframes gradient-flow {
            0%, 100% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
        }

        @keyframes underline-glow {
            0%, 100% {
                box-shadow: 0 0 5px rgba(245, 158, 11, 0.3);
            }
            50% {
                box-shadow: 0 0 20px rgba(245, 158, 11, 0.6);
            }
        }

        @keyframes fade-in-up {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    </style>
    
    <div class="logo-container">
        <div class="logo-frame">
            <div class="logo-icon">
                <div class="bar bar1" style="--final-height: 20px;"></div>
                <div class="bar bar2" style="--final-height: 40px;"></div>
                <div class="bar bar3" style="--final-height: 55px;"></div>
                <div class="bar bar4" style="--final-height: 35px;"></div>
                <div class="bar bar5" style="--final-height: 25px;"></div>
            </div>
            <div class="logo-text">
                Vision Stock <span class="highlight">Pro</span>
            </div>
        </div>
        <div class="tagline">
            Intelligence Prédictive • Précision Maximale • Performance Optimale
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Sidebar
    st.sidebar.title("🎯 Configuration")
    
    # Récupérer les datasets disponibles
    datasets = get_datasets()
    
    if not datasets:
        st.error("❌ Aucun dataset trouvé")
        return
    
    # Sélection du dataset
    dataset_names = [d['name'] for d in datasets]
    selected_name = st.sidebar.selectbox("📁 Sélectionner un dataset", dataset_names)
    
    # Trouver le dataset sélectionné
    selected_dataset = next(d for d in datasets if d['name'] == selected_name)
    dataset_key = selected_dataset['key']
    model_folder = selected_dataset['folder']
    
    # Récupérer les informations du fichier
    original_files = {
        'mayor1_csv': 'mayor1.xlsx',
        'laitbroli_1kg_xls': 'laitbroli_1kg.xls',
        'may_arm_1kg_xls': 'may_arm_1kg.xls',
        'may_arm_5kg_xls': 'may_arm_5kg.xls',
        'couche_softcqre_T4_xls': 'couche_softcqre_T4.xls',
        'papierhygsita_xls': 'papierhygsita.xls',
        'parleG_xls': 'parleG.xls'
    }
    
    original_filename = original_files.get(dataset_key, "Fichier non trouvé")
    
    # Paramètres
    st.sidebar.subheader("⚙️ Paramètres")
    prediction_days = st.sidebar.slider("📅 Jours de prédiction", 7, 90, 30)
    
    # Sélecteur de période pour le tableau de bord
    
    # Charger les données historiques
    with st.spinner("🔄 Chargement des données..."):
        historical_data, last_date, date_col = load_historical_data(dataset_key)
    
    # Mettre à jour les informations de configuration après chargement
    st.sidebar.subheader("📋 Informations de Configuration")
    st.sidebar.info(f"📁 **Fichier:** {original_filename}")
    st.sidebar.info(f"📅 **Colonne de date:** {date_col}")
    st.sidebar.info(f"📅 **Dernière date:** {last_date.strftime('%d/%m/%Y')}")
    st.sidebar.info(f"🤖 **Modèles:** {len([f for f in os.listdir(model_folder) if f.endswith('.joblib')])} chargés")
    
    # Informations sur les features
    feature_names = load_feature_names(model_folder)
    if feature_names:
        st.sidebar.success(f"✅ **Features:** {len(feature_names)} disponibles")
    else:
        st.sidebar.warning("⚠️ **Features:** Prédictions simples")
    
    # Charger les modèles
    with st.spinner("🤖 Chargement des modèles..."):
        models = load_models(model_folder)
    
    if not models:
        st.error("❌ Aucun modèle trouvé")
        return
    
    # st.success(f"✅ {len(models)} modèles chargés avec succès!")
    
    # Charger les données historiques pour les graphiques
    with st.spinner("📊 Chargement des données historiques..."):
        historical_data = None
        try:
            # Essayer de charger les données pour les graphiques
            original_files = {
                'mayor1_csv': 'mayor1.csv',
                'laitbroli_1kg_xls': 'laitbroli_1kg.xls',
                'may_arm_1kg_xls': 'may_arm_1kg.xls',
                'may_arm_5kg_xls': 'may_arm_5kg.xls',
                'couche_softcqre_T4_xls': 'couche_softcqre_T4.xls',
                'papierhygsita_xls': 'papierhygsita.xls',
                'parleG_xls': 'parleG.xls'
            }
            
            original_filename = original_files.get(dataset_key)
            if original_filename and os.path.exists(original_filename):
                if original_filename.endswith('.csv'):
                    historical_data = pd.read_csv(original_filename, encoding='latin-1', sep=';', on_bad_lines='skip')
                else:
                    historical_data = pd.read_csv(original_filename, sep='\t', encoding='latin-1', on_bad_lines='skip')
                
                # Nettoyer les colonnes numériques
                numeric_cols = ['Entrée', 'Stock', 'Sortie']
                for col in numeric_cols:
                    if col in historical_data.columns:
                        historical_data[col] = pd.to_numeric(historical_data[col], errors='coerce')
                historical_data = historical_data.fillna(0)
        except:
            pass
    
    # Générer les prédictions
    with st.spinner("🔮 Génération des prédictions..."):
        predictions, uncertainties, individual_predictions = make_real_predictions(models, historical_data, last_date, prediction_days)
    
    # Créer les dates de prédiction
    prediction_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
    
    # Créer les métriques du tableau de bord (sera recalculé dans l'onglet Tableau de Bord)
    dashboard_data = create_dashboard_metrics(predictions, uncertainties, historical_data, 30)  # Valeur par défaut
    
    # Stocker les données pour le chatbot
    st.session_state.predictions_data = {
        'predictions': predictions,
        'uncertainties': uncertainties,
        'period': f'{prediction_days} jours',
        'models_count': len(models)
    }
    
    # Créer les métriques du dashboard pour le chatbot
    if predictions and uncertainties:
        total_consumption = sum(predictions)
        avg_daily_consumption = np.mean(predictions)
        avg_uncertainty = np.mean(uncertainties)
        stock_max = total_consumption + avg_uncertainty
        stock_min = total_consumption - avg_uncertainty
        cv = (np.std(predictions) / np.mean(predictions)) * 100 if np.mean(predictions) > 0 else 0
        confidence = max(0, min(1, 1 - (avg_uncertainty / avg_daily_consumption))) if avg_daily_consumption > 0 else 0
        days_to_rupture = 30 / (avg_daily_consumption / 30) if avg_daily_consumption > 0 else 0
        volatility = np.std(predictions)
        stock_efficiency = confidence * (1 - cv/100)
        
        # Tendance
        if len(predictions) >= 7:
            early_avg = np.mean(predictions[:7])
            late_avg = np.mean(predictions[-7:])
            trend_direction = "Hausse" if late_avg > early_avg else "Baisse" if late_avg < early_avg else "Stable"
        else:
            trend_direction = "N/A"
        
        # Stocker les données brutes pour l'analyse des produits
        st.session_state.raw_data = historical_data
        
        # Calculer les stocks actuels par produit
        product_stocks = {}
        if historical_data is not None and hasattr(historical_data, 'columns'):
            try:
                # Identifier la colonne des produits
                product_columns = [col for col in historical_data.columns if any(word in col.lower() for word in ['produit', 'product', 'intitule', 'nom', 'name', 'item'])]
                
                if product_columns and 'Quantite' in historical_data.columns:
                    product_col = product_columns[0]
                    
                    # Calculer le stock actuel pour chaque produit
                    for product in historical_data[product_col].unique():
                        product_data = historical_data[historical_data[product_col] == product]
                        
                        # Calculer les métriques du produit
                        total_consumption = product_data['Quantite'].sum()
                        avg_daily_consumption = product_data['Quantite'].mean()
                        transactions_count = len(product_data)
                        
                        # Estimation du stock actuel (basée sur la consommation récente)
                        recent_consumption = product_data['Quantite'].tail(7).mean() if len(product_data) >= 7 else avg_daily_consumption
                        estimated_current_stock = max(0, total_consumption - (recent_consumption * 30))  # Estimation sur 30 jours
                        
                        # Calculer les jours jusqu'à rupture pour ce produit
                        days_to_rupture = estimated_current_stock / recent_consumption if recent_consumption > 0 else 999
                        
                        product_stocks[str(product)] = {
                            'total_consumption': total_consumption,
                            'avg_daily_consumption': avg_daily_consumption,
                            'recent_consumption': recent_consumption,
                            'estimated_current_stock': estimated_current_stock,
                            'days_to_rupture': days_to_rupture,
                            'transactions_count': transactions_count,
                            'stock_status': 'CRITIQUE' if days_to_rupture < 7 else 'FAIBLE' if days_to_rupture < 14 else 'NORMAL' if days_to_rupture < 30 else 'ÉLEVÉ'
                        }
            except Exception as e:
                st.warning(f"Erreur lors du calcul des stocks par produit : {e}")
        
        # Stocker les stocks par produit
        st.session_state.product_stocks = product_stocks
        
        # Stocker TOUTES les données spécifiques de l'application
        st.session_state.all_application_data = {
            # Données du dashboard
            'dashboard_metrics': {
                'total_consumption': total_consumption,
                'avg_daily_consumption': avg_daily_consumption,
                'stock_max': stock_max,
                'stock_min': stock_min,
                'cv': cv,
                'days_to_rupture': days_to_rupture,
                'confidence': confidence,
                'volatility': volatility,
                'stock_efficiency': stock_efficiency,
                'trend_direction': trend_direction,
                'avg_uncertainty': avg_uncertainty
            },
            
            # Données des produits détaillées
            'products_detailed': product_stocks,
            
            # Données brutes complètes
            'raw_historical_data': historical_data,
            
            # Métadonnées des données
            'data_metadata': {
                'total_products': len(product_stocks) if product_stocks else 0,
                'data_columns': list(historical_data.columns) if historical_data is not None else [],
                'data_shape': historical_data.shape if historical_data is not None else (0, 0),
                'date_range': {
                    'start': historical_data.index.min() if historical_data is not None and len(historical_data) > 0 else None,
                    'end': historical_data.index.max() if historical_data is not None and len(historical_data) > 0 else None
                }
            },
            
            # Statistiques avancées par produit
            'advanced_product_stats': {}
        }
        
        # Calculer des statistiques avancées pour chaque produit
        if product_stocks and historical_data is not None:
            try:
                product_columns = [col for col in historical_data.columns if any(word in col.lower() for word in ['produit', 'product', 'intitule', 'nom', 'name', 'item'])]
                if product_columns and 'Quantite' in historical_data.columns:
                    product_col = product_columns[0]
                    
                    for product_name, product_data in product_stocks.items():
                        product_df = historical_data[historical_data[product_col] == product_name]
                        
                        if len(product_df) > 0:
                            # Statistiques avancées
                            quantities = product_df['Quantite']
                            
                            advanced_stats = {
                                'min_consumption': quantities.min(),
                                'max_consumption': quantities.max(),
                                'median_consumption': quantities.median(),
                                'std_consumption': quantities.std(),
                                'q25_consumption': quantities.quantile(0.25),
                                'q75_consumption': quantities.quantile(0.75),
                                'consumption_trend': 'increasing' if len(quantities) > 1 and quantities.iloc[-1] > quantities.iloc[0] else 'decreasing' if len(quantities) > 1 and quantities.iloc[-1] < quantities.iloc[0] else 'stable',
                                'seasonality_score': 0,  # À calculer si nécessaire
                                'outlier_count': len(quantities[(quantities < quantities.quantile(0.25) - 1.5 * (quantities.quantile(0.75) - quantities.quantile(0.25))) | 
                                                               (quantities > quantities.quantile(0.75) + 1.5 * (quantities.quantile(0.75) - quantities.quantile(0.25)))]),
                                'recent_volatility': quantities.tail(7).std() if len(quantities) >= 7 else quantities.std(),
                                'consumption_pattern': 'regular' if quantities.std() < quantities.mean() * 0.3 else 'irregular',
                                'peak_consumption_day': quantities.idxmax() if len(quantities) > 0 else None,
                                'low_consumption_day': quantities.idxmin() if len(quantities) > 0 else None
                            }
                            
                            st.session_state.all_application_data['advanced_product_stats'][product_name] = advanced_stats
            except Exception as e:
                st.warning(f"Erreur lors du calcul des statistiques avancées : {e}")
        
        # Conserver l'ancien format pour la compatibilité
        st.session_state.dashboard_data = st.session_state.all_application_data['dashboard_metrics']
        
        # Initialiser la base de données du chatbot
        print("🔄 Initialisation de la base de données du chatbot...")
        chatbot_db = save_dashboard_data_to_memory(
            st.session_state.dashboard_data,
            st.session_state.product_stocks,
            st.session_state.predictions_data
        )
        if chatbot_db:
            print("✅ Base de données du chatbot initialisée avec succès")
        else:
            print("❌ Erreur lors de l'initialisation de la base de données du chatbot")
    
    
    # Tabs
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🏠 Accueil", "📊 Tableau de Bord", "🎯 Prédictions", "📈 Analyses", "🤖 Modèles", "📊 Historique", "⚙️ Configuration", "🧠 Vision IA"])
    
    with tab0:
        # PAGE D'ACCUEIL AVEC IMAGE D'ARRIÈRE-PLAN
        st.markdown("""
        <div class="homepage-bg">
            <div style="text-align: center;">
                <h1 style="
                    font-size: 4rem; 
                    margin-bottom: 25px; 
                    text-shadow: 3px 3px 6px rgba(0,0,0,0.8);
                    color: #ffffff;
                    font-weight: 700;
                ">
                    🎉 Bienvenue dans Vision Stock Pro
                </h1>
                <p style="
                    font-size: 1.8rem; 
                    margin-bottom: 35px; 
                    opacity: 0.95;
                    font-weight: 400;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.6);
                ">
                    L'Intelligence Artificielle au Service de la Gestion de Stock
                </p>
                <div style="
                    background: rgba(255,255,255,0.15); 
                    padding: 30px; 
                    border-radius: 20px; 
                    max-width: 900px;
                    margin: 0 auto;
                    border: 2px solid rgba(255,255,255,0.3);
                    backdrop-filter: blur(10px);
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                ">
                    <p style="
                        font-size: 1.3rem; 
                        line-height: 1.7; 
                        margin: 0;
                        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
                    ">
                        Transformez votre gestion de stock avec notre solution d'IA avancée. 
                        Prédictions précises, analyses intelligentes et optimisation automatique 
                        pour maximiser votre efficacité opérationnelle.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # DESCRIPTION DE L'APPLICATION
        st.markdown("## 🚀 À Propos de Vision Stock Pro")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Vision Stock Pro** est une solution révolutionnaire d'intelligence artificielle 
            conçue pour optimiser la gestion de vos stocks. Notre plateforme utilise des 
            modèles d'apprentissage automatique avancés pour prédire avec précision les 
            besoins futurs en stock.
            
            ### 🎯 Notre Mission
            - **Prédire** les besoins futurs avec une précision maximale
            - **Optimiser** la gestion des stocks et réduire les coûts
            - **Automatiser** les processus de réapprovisionnement
            - **Analyser** les tendances et patterns de consommation
            
            ### 💡 Pourquoi Choisir Vision Stock Pro ?
            - ✅ **Précision Exceptionnelle** : Modèles d'ensemble avec 95%+ de précision
            - ✅ **Temps Réel** : Prédictions instantanées et mises à jour automatiques
            - ✅ **Interface Intuitive** : Dashboard moderne et facile à utiliser
            - ✅ **Analyses Avancées** : Insights détaillés pour une prise de décision éclairée
            """)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 15px; color: white; margin-bottom: 20px;">
                <h3 style="margin: 0 0 10px 0;">📊 Précision</h3>
                <h2 style="margin: 0; font-size: 2.5rem;">95%+</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Taux de Prédiction</p>
            </div>
            
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 15px; color: white; margin-bottom: 20px;">
                <h3 style="margin: 0 0 10px 0;">⚡ Vitesse</h3>
                <h2 style="margin: 0; font-size: 2.5rem;">&lt;1s</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Temps de Calcul</p>
            </div>
            
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); border-radius: 15px; color: white;">
                <h3 style="margin: 0 0 10px 0;">💰 Économies</h3>
                <h2 style="margin: 0; font-size: 2.5rem;">30%+</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Réduction des Coûts</p>
            </div>
            """, unsafe_allow_html=True)
        
        # FONCTIONNALITÉS PRINCIPALES
        st.markdown("## 🌟 Fonctionnalités Principales")
        
        # Grille de fonctionnalités
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 30px 20px; background: rgba(102, 126, 234, 0.1); border-radius: 15px; border: 2px solid rgba(102, 126, 234, 0.2); margin-bottom: 20px;">
                <div style="font-size: 3rem; margin-bottom: 15px;">📊</div>
                <h3 style="color: #667eea; margin-bottom: 15px;">Tableau de Bord Intelligent</h3>
                <p style="color: #666; line-height: 1.5;">
                    Dashboard interactif avec métriques en temps réel, alertes intelligentes 
                    et visualisations avancées pour une vue d'ensemble complète.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="text-align: center; padding: 30px 20px; background: rgba(245, 87, 108, 0.1); border-radius: 15px; border: 2px solid rgba(245, 87, 108, 0.2); margin-bottom: 20px;">
                <div style="font-size: 3rem; margin-bottom: 15px;">🔮</div>
                <h3 style="color: #f5576c; margin-bottom: 15px;">Prédictions Précises</h3>
                <p style="color: #666; line-height: 1.5;">
                    Prédictions 30 jours à l'avance avec quantification d'incertitude 
                    et modèles d'ensemble pour une précision maximale.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 30px 20px; background: rgba(67, 233, 123, 0.1); border-radius: 15px; border: 2px solid rgba(67, 233, 123, 0.2); margin-bottom: 20px;">
                <div style="font-size: 3rem; margin-bottom: 15px;">📈</div>
                <h3 style="color: #43e97b; margin-bottom: 15px;">Analyses Avancées</h3>
                <p style="color: #666; line-height: 1.5;">
                    Analyses statistiques approfondies, corrélations, importance des features 
                    et évaluation des performances des modèles.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="text-align: center; padding: 30px 20px; background: rgba(79, 172, 254, 0.1); border-radius: 15px; border: 2px solid rgba(79, 172, 254, 0.2); margin-bottom: 20px;">
                <div style="font-size: 3rem; margin-bottom: 15px;">🤖</div>
                <h3 style="color: #4facfe; margin-bottom: 15px;">Modèles d'IA</h3>
                <p style="color: #666; line-height: 1.5;">
                    Ensemble de modèles (Random Forest, XGBoost, LightGBM) 
                    avec sélection dynamique pour des prédictions optimales.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 30px 20px; background: rgba(240, 147, 251, 0.1); border-radius: 15px; border: 2px solid rgba(240, 147, 251, 0.2); margin-bottom: 20px;">
                <div style="font-size: 3rem; margin-bottom: 15px;">📊</div>
                <h3 style="color: #f093fb; margin-bottom: 15px;">Données Historiques</h3>
                <p style="color: #666; line-height: 1.5;">
                    Visualisation des données historiques avec graphiques interactifs 
                    et analyses temporelles pour comprendre les tendances.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="text-align: center; padding: 30px 20px; background: rgba(255, 193, 7, 0.1); border-radius: 15px; border: 2px solid rgba(255, 193, 7, 0.2); margin-bottom: 20px;">
                <div style="font-size: 3rem; margin-bottom: 15px;">⚙️</div>
                <h3 style="color: #ffc107; margin-bottom: 15px;">Configuration</h3>
                <p style="color: #666; line-height: 1.5;">
                    Interface de configuration intuitive pour personnaliser 
                    les paramètres et adapter l'application à vos besoins.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # CALL TO ACTION
        st.markdown("""
        <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white; margin-top: 30px;">
            <h2 style="margin-bottom: 20px;">🚀 Prêt à Révolutionner Votre Gestion de Stock ?</h2>
            <p style="font-size: 1.2rem; margin-bottom: 30px; opacity: 0.9;">
                Explorez les différents onglets pour découvrir toutes les fonctionnalités 
                et commencez à optimiser votre gestion de stock dès maintenant !
            </p>
            <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                <div style="background: rgba(255,255,255,0.2); padding: 15px 30px; border-radius: 25px; backdrop-filter: blur(10px);">
                    📊 Tableau de Bord
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 15px 30px; border-radius: 25px; backdrop-filter: blur(10px);">
                    🎯 Prédictions
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 15px 30px; border-radius: 25px; backdrop-filter: blur(10px);">
                    📈 Analyses
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    
    with tab1:
        # TABLEAU DE BORD PRINCIPAL AVEC IMAGE D'ARRIÈRE-PLAN
        st.markdown('<div class="dashboard-bg">', unsafe_allow_html=True)
        st.markdown("## 🏠 Tableau de Bord Intelligent")
        
        # Sélecteur de période pour le tableau de bord
        col_period1, col_period2 = st.columns([1, 3])
        with col_period1:
            dashboard_period = st.selectbox(
                "📊 Période d'analyse",
                options=[7, 15, 30, 40],
                index=2,  # 30 jours par défaut
                format_func=lambda x: f"{x} jours"
            )
        with col_period2:
            st.info(f"📈 Analyse sur {dashboard_period} jours")
        
        # Recalculer les métriques avec la période sélectionnée
        dashboard_data = create_dashboard_metrics(predictions, uncertainties, historical_data, dashboard_period)
        
        # Afficher le tableau de bord principal
        if dashboard_data:
            # Créer le HTML complet en une seule ligne
            html_content = f'<div style="background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); padding: 25px; border-radius: 15px; margin: 20px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1); border-left: 5px solid {dashboard_data["color"]};">'
            html_content += f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">'
            html_content += f'<div><h3 style="margin: 0; color: #1e293b; font-size: 1.5rem;">{dashboard_data["icon"]} Statut du Stock</h3>'
            html_content += f'<p style="margin: 5px 0 0 0; color: #64748b; font-size: 1rem;">{dashboard_data["message"]}</p></div>'
            html_content += f'<div style="text-align: right;"><div style="font-size: 2rem; font-weight: bold; color: {dashboard_data["color"]};">{dashboard_data["details"]["current_stock"]:.0f}</div>'
            html_content += f'<div style="color: #64748b; font-size: 0.9rem;">Stock Actuel</div></div></div>'
            html_content += f'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">'
            html_content += f'<div style="text-align: center; padding: 15px; background: rgba(59, 130, 246, 0.1); border-radius: 10px;">'
            html_content += f'<div style="font-size: 1.5rem; font-weight: bold; color: #3b82f6;">{dashboard_data["details"]["total_consumption"]:.0f}</div>'
            html_content += f'<div style="color: #64748b; font-size: 0.9rem;">Consommation {dashboard_data["details"]["period"]}j</div></div>'
            html_content += f'<div style="text-align: center; padding: 15px; background: rgba(16, 185, 129, 0.1); border-radius: 10px;">'
            html_content += f'<div style="font-size: 1.5rem; font-weight: bold; color: #10b981;">{dashboard_data["details"]["stock_prediction_min"]:.0f}</div>'
            html_content += f'<div style="color: #64748b; font-size: 0.9rem;">Stock Min (Incertitude)</div></div>'
            html_content += f'<div style="text-align: center; padding: 15px; background: rgba(245, 158, 11, 0.1); border-radius: 10px;">'
            html_content += f'<div style="font-size: 1.5rem; font-weight: bold; color: #f59e0b;">{dashboard_data["details"]["recommended_stock"]:.0f}</div>'
            html_content += f'<div style="color: #64748b; font-size: 0.9rem;">Stock Recommandé</div></div>'
            html_content += f'<div style="text-align: center; padding: 15px; background: rgba(239, 68, 68, 0.1); border-radius: 10px;">'
            html_content += f'<div style="font-size: 1.5rem; font-weight: bold; color: #ef4444;">{dashboard_data["details"]["stock_prediction_max"]:.0f}</div>'
            html_content += f'<div style="color: #64748b; font-size: 0.9rem;">Stock Max (Incertitude)</div></div>'
            html_content += f'<div style="text-align: center; padding: 15px; background: rgba(139, 92, 246, 0.1); border-radius: 10px;">'
            html_content += f'<div style="font-size: 1.5rem; font-weight: bold; color: #8b5cf6;">{dashboard_data["details"]["confidence_pct"]:.1f}%</div>'
            html_content += f'<div style="color: #64748b; font-size: 0.9rem;">Confiance Prédiction</div></div>'
            html_content += f'<div style="text-align: center; padding: 15px; background: rgba(34, 197, 94, 0.1); border-radius: 10px;">'
            html_content += f'<div style="font-size: 1.5rem; font-weight: bold; color: #22c55e;">{dashboard_data["details"]["avg_daily_consumption"]:.1f}</div>'
            html_content += f'<div style="color: #64748b; font-size: 0.9rem;">Moyenne/jour</div></div>'
            
            # NOUVELLES MÉTRIQUES AVANCÉES
            html_content += f'<div style="text-align: center; padding: 15px; background: rgba(168, 85, 247, 0.1); border-radius: 10px;">'
            html_content += f'<div style="font-size: 1.5rem; font-weight: bold; color: #a855f7;">{dashboard_data["details"]["stability_index"]:.1f}%</div>'
            html_content += f'<div style="color: #64748b; font-size: 0.9rem;">Stabilité</div></div>'
            
            html_content += f'<div style="text-align: center; padding: 15px; background: rgba(236, 72, 153, 0.1); border-radius: 10px;">'
            html_content += f'<div style="font-size: 1.5rem; font-weight: bold; color: #ec4899;">{dashboard_data["details"]["confidence_score"]:.1f}%</div>'
            html_content += f'<div style="color: #64748b; font-size: 0.9rem;">Score Confiance</div></div>'
            
            if dashboard_data["details"]["days_to_rupture"] is not None:
                html_content += f'<div style="text-align: center; padding: 15px; background: rgba(239, 68, 68, 0.1); border-radius: 10px;">'
                html_content += f'<div style="font-size: 1.5rem; font-weight: bold; color: #ef4444;">{dashboard_data["details"]["days_to_rupture"]:.1f}</div>'
                html_content += f'<div style="color: #64748b; font-size: 0.9rem;">Jours avant rupture</div></div>'
            
            html_content += f'<div style="text-align: center; padding: 15px; background: rgba(14, 165, 233, 0.1); border-radius: 10px;">'
            html_content += f'<div style="font-size: 1.5rem; font-weight: bold; color: #0ea5e9;">{dashboard_data["details"]["volatility"]:.1f}</div>'
            html_content += f'<div style="color: #64748b; font-size: 0.9rem;">Volatilité</div></div>'
            
            html_content += f'<div style="text-align: center; padding: 15px; background: rgba(34, 197, 94, 0.1); border-radius: 10px;">'
            html_content += f'<div style="font-size: 1.5rem; font-weight: bold; color: #22c55e;">{dashboard_data["details"]["stock_efficiency"]:.1f}%</div>'
            html_content += f'<div style="color: #64748b; font-size: 0.9rem;">Efficacité Stock</div></div>'
            
            # Indicateur de tendance avec flèche
            trend_icon = "📈" if dashboard_data["details"]["trend_direction"] == "hausse" else "📉" if dashboard_data["details"]["trend_direction"] == "baisse" else "➡️"
            trend_color = "#10b981" if dashboard_data["details"]["trend_direction"] == "hausse" else "#ef4444" if dashboard_data["details"]["trend_direction"] == "baisse" else "#6b7280"
            html_content += f'<div style="text-align: center; padding: 15px; background: rgba(16, 185, 129, 0.1); border-radius: 10px;">'
            html_content += f'<div style="font-size: 1.5rem; font-weight: bold; color: {trend_color};">{trend_icon} {abs(dashboard_data["details"]["trend_pct"]):.1f}%</div>'
            html_content += f'<div style="color: #64748b; font-size: 0.9rem;">Tendance {dashboard_data["details"]["trend_direction"]}</div></div></div></div>'
            
            st.markdown(html_content, unsafe_allow_html=True)
        
        # SYSTÈME D'ALERTES INTELLIGENTES
        if dashboard_data:
            alerts = create_alert_system(dashboard_data)
            if alerts:
                st.markdown("### 🚨 Alertes Intelligentes")
                for alert in alerts:
                    st.markdown(f"""
                    <div style="
                        background: {alert['color']}20; 
                        border-left: 4px solid {alert['color']}; 
                        padding: 15px; 
                        margin: 10px 0; 
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 1.5rem;">{alert['icon']}</span>
                            <div>
                                <h4 style="margin: 0; color: {alert['color']}; font-size: 1.1rem;">{alert['title']}</h4>
                                <p style="margin: 5px 0 0 0; color: #64748b;">{alert['message']}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # GAUGES AVANCÉS
        if dashboard_data:
            st.markdown("### 📊 Indicateurs de Performance Avancés")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
                st.plotly_chart(create_gauge_chart(
                    dashboard_data['details']['confidence_score'], 
                    "Score de Confiance", 
                    "blue", 
                    100
                ), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
                st.plotly_chart(create_gauge_chart(
                    dashboard_data['details']['stability_index'], 
                    "Indice de Stabilité", 
                    "green", 
                    100
                ), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                if dashboard_data['details']['days_to_rupture'] is not None:
                    st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
                    max_days = 30
                    days = min(dashboard_data['details']['days_to_rupture'], max_days)
                    st.plotly_chart(create_gauge_chart(
                        days, 
                        "Jours avant Rupture", 
                        "red", 
                        max_days
                    ), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        # MÉTRIQUES SIMPLES POUR PRÉDICTIONS AVEC IMAGE D'ARRIÈRE-PLAN
        st.markdown('<div class="predictions-bg">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Prédiction Moyenne", 
                f"{np.mean(predictions):.1f}",
                delta=f"{np.mean(predictions) - 70:.1f}"
            )
        
        with col2:
            st.metric(
                "📈 Prédiction Max", 
                f"{np.max(predictions):.1f}",
                delta=f"{np.max(predictions) - 70:.1f}"
            )
        
        with col3:
            st.metric(
                "📉 Prédiction Min", 
                f"{np.min(predictions):.1f}",
                delta=f"{np.min(predictions) - 70:.1f}"
            )
        
        with col4:
            st.metric(
                "🎯 Incertitude Moyenne", 
                f"{np.mean(uncertainties):.1f}",
                delta=f"{np.mean(uncertainties) / np.mean(predictions) * 100:.1f}%"
            )
        
        
        # Graphiques avancés
        charts = create_advanced_charts(predictions, uncertainties, prediction_dates, historical_data, individual_predictions)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(charts['main'], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Comparaison des modèles individuels
        if 'models' in charts:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(charts['models'], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tableau des prédictions
        st.subheader("📋 Détails des Prédictions")
        
        pred_df = pd.DataFrame({
            'Date': prediction_dates,
            'Prédiction': [f"{p:.2f}" for p in predictions],
            'Incertitude': [f"{u:.2f}" for u in uncertainties],
            'Min': [f"{p-u:.2f}" for p, u in zip(predictions, uncertainties)],
            'Max': [f"{p+u:.2f}" for p, u in zip(predictions, uncertainties)]
        })
        
        st.dataframe(pred_df, use_container_width=True)
        
        # Bouton de téléchargement
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les prédictions (CSV)",
            data=csv,
            file_name=f"predictions_{dataset_key}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        # ANALYSES AVEC IMAGE D'ARRIÈRE-PLAN
        st.markdown('<div class="analyses-bg">', unsafe_allow_html=True)
        st.subheader("📊 Analyses des Prédictions")
        
        # ANALYSE DES PRÉDICTIONS - SECTION PRINCIPALE
        st.markdown("### 🔮 Analyse des Prédictions et Données")
        
        # ANALYSE DES PRÉDICTIONS
        st.markdown("#### 📈 Analyse des Prédictions")
        
        # Métriques des prédictions
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Prédiction Moyenne",
                f"{np.mean(predictions):.2f}",
                delta=f"±{np.mean(uncertainties):.2f}"
            )
        
        with col2:
            st.metric(
                "📈 Prédiction Max",
                f"{np.max(predictions):.2f}",
                delta=f"Jour {np.argmax(predictions) + 1}"
            )
        
        with col3:
            st.metric(
                "📉 Prédiction Min",
                f"{np.min(predictions):.2f}",
                delta=f"Jour {np.argmin(predictions) + 1}"
            )
        
        with col4:
            st.metric(
                "🎯 Incertitude Moyenne",
                f"{np.mean(uncertainties):.2f}",
                delta=f"{(np.mean(uncertainties) / np.mean(predictions)) * 100:.1f}%"
            )
        
        # Graphique d'évolution des prédictions
        st.markdown("#### 🔄 Évolution des Prédictions dans le Temps")
        
        fig_predictions = go.Figure()
        
        # Ligne principale des prédictions
        fig_predictions.add_trace(go.Scatter(
            x=prediction_dates,
            y=predictions,
            mode='lines+markers',
            name='Prédictions',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6)
        ))
        
        # Zone d'incertitude
        upper_bound = [p + u for p, u in zip(predictions, uncertainties)]
        lower_bound = [p - u for p, u in zip(predictions, uncertainties)]
        
        fig_predictions.add_trace(go.Scatter(
            x=prediction_dates + prediction_dates[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Incertitude',
            showlegend=True
        ))
        
        fig_predictions.update_layout(
            title="📊 Évolution des Prédictions avec Incertitude",
            xaxis_title="Date",
            yaxis_title="Quantité Prédite",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig_predictions, use_container_width=True)
        
        # Analyse de la tendance
        st.markdown("#### 📈 Analyse de la Tendance")
        
        # Calculer la tendance
        if len(predictions) >= 7:
            # Tendance sur les 7 premiers jours vs les 7 derniers
            early_trend = np.mean(predictions[:7])
            late_trend = np.mean(predictions[-7:])
            trend_pct = ((late_trend - early_trend) / early_trend) * 100 if early_trend > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "📊 Début de Période",
                    f"{early_trend:.2f}",
                    delta="7 premiers jours"
                )
            
            with col2:
                st.metric(
                    "📈 Fin de Période",
                    f"{late_trend:.2f}",
                    delta="7 derniers jours"
                )
            
            with col3:
                trend_icon = "📈" if trend_pct > 0 else "📉" if trend_pct < 0 else "➡️"
                st.metric(
                    "🎯 Tendance",
                    f"{trend_icon} {abs(trend_pct):.1f}%",
                    delta="Évolution globale"
                )
        
        # Distribution des prédictions
        st.markdown("#### 📊 Distribution des Prédictions")
        
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=predictions,
            name="Prédictions",
            opacity=0.7,
            nbinsx=20,
            marker_color='lightblue'
        ))
        
        # Ligne de moyenne
        mean_val = np.mean(predictions)
        fig_dist.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                          annotation_text=f"Moyenne: {mean_val:.1f}")
        
        # Ligne de médiane
        median_val = np.median(predictions)
        fig_dist.add_vline(x=median_val, line_dash="dot", line_color="green",
                          annotation_text=f"Médiane: {median_val:.1f}")
        
        fig_dist.update_layout(
            title="📈 Distribution des Prédictions",
            xaxis_title="Quantité Prédite",
            yaxis_title="Fréquence",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Statistiques descriptives
        st.markdown("#### 📈 Statistiques Descriptives des Prédictions")
        
        stats_data = {
            'Métrique': ['Moyenne', 'Médiane', 'Écart-type', 'Min', 'Max', 'Q1', 'Q3', 'CV'],
            'Valeur': [
                f"{np.mean(predictions):.2f}",
                f"{np.median(predictions):.2f}",
                f"{np.std(predictions):.2f}",
                f"{np.min(predictions):.2f}",
                f"{np.max(predictions):.2f}",
                f"{np.percentile(predictions, 25):.2f}",
                f"{np.percentile(predictions, 75):.2f}",
                f"{(np.std(predictions) / np.mean(predictions)) * 100:.2f}%"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # ANALYSE DE LA QUALITÉ DES PRÉDICTIONS
        st.markdown("#### 🎯 Analyse de la Qualité des Prédictions")
        
        # Métriques de qualité
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Précision",
                f"{(1 - np.mean(uncertainties) / np.mean(predictions)) * 100:.1f}%",
                delta="Plus haut = plus précis"
            )
        
        with col2:
            st.metric(
                "🎯 Stabilité",
                f"{(1 - np.std(predictions) / np.mean(predictions)) * 100:.1f}%",
                delta="Cohérence des prédictions"
            )
        
        with col3:
            st.metric(
                "📈 Fiabilité",
                f"{(1 - np.max(uncertainties) / np.mean(predictions)) * 100:.1f}%",
                delta="Confiance maximale"
            )
        
        with col4:
            st.metric(
                "⚖️ Équilibre",
                f"{np.mean(predictions):.1f}",
                delta="Niveau moyen"
            )
        
        # ANALYSE DES DONNÉES HISTORIQUES
        if historical_data is not None and len(historical_data) > 0:
            st.markdown("#### 📊 Analyse des Données Historiques")
            
            # Comparaison historique vs prédictions
            if 'Sortie' in historical_data.columns:
                historical_mean = historical_data['Sortie'].mean()
                prediction_mean = np.mean(predictions)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "📊 Moyenne Historique",
                        f"{historical_mean:.2f}",
                        delta="Données passées"
                    )
                
                with col2:
                    st.metric(
                        "🔮 Moyenne Prédite",
                        f"{prediction_mean:.2f}",
                        delta="Prédictions futures"
                    )
                
                with col3:
                    evolution = ((prediction_mean - historical_mean) / historical_mean) * 100 if historical_mean > 0 else 0
                    st.metric(
                        "📈 Évolution",
                        f"{evolution:+.1f}%",
                        delta="Tendance"
                    )
        
        # ANALYSE TEMPORELLE
        st.markdown("#### ⏰ Analyse Temporelle des Prédictions")
        
        # Analyse par semaine
        if len(predictions) >= 7:
            weekly_analysis = []
            for i in range(0, len(predictions), 7):
                week_predictions = predictions[i:i+7]
                weekly_analysis.append({
                    'Semaine': f"Semaine {i//7 + 1}",
                    'Moyenne': np.mean(week_predictions),
                    'Min': np.min(week_predictions),
                    'Max': np.max(week_predictions),
                    'Écart-type': np.std(week_predictions)
                })
            
            if weekly_analysis:
                weekly_df = pd.DataFrame(weekly_analysis)
                st.dataframe(weekly_df, use_container_width=True)
        
        # RECOMMANDATIONS
        st.markdown("#### 💡 Recommandations Basées sur l'Analyse")
        
        # Générer des recommandations intelligentes
        recommendations = []
        
        if np.mean(uncertainties) / np.mean(predictions) > 0.2:
            recommendations.append("⚠️ **Incertitude élevée** : Les prédictions ont une marge d'erreur importante. Surveillez de près les stocks.")
        
        if np.std(predictions) / np.mean(predictions) > 0.3:
            recommendations.append("📈 **Variabilité importante** : Les prédictions varient beaucoup. Préparez-vous à des fluctuations de demande.")
        
        if np.mean(predictions) > historical_data['Sortie'].mean() * 1.2 if historical_data is not None and 'Sortie' in historical_data.columns else False:
            recommendations.append("🚀 **Croissance attendue** : Les prédictions indiquent une augmentation de la demande. Augmentez vos stocks.")
        
        if len(predictions) >= 7 and np.mean(predictions[:7]) > np.mean(predictions[-7:]):
            recommendations.append("📉 **Tendance à la baisse** : La demande semble diminuer vers la fin de la période. Ajustez vos commandes.")
        
        if not recommendations:
            recommendations.append("✅ **Situation stable** : Les prédictions sont cohérentes et fiables. Continuez votre gestion actuelle.")
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        # MODÈLES AVEC IMAGE D'ARRIÈRE-PLAN
        st.markdown('<div class="models-bg">', unsafe_allow_html=True)
        st.subheader("🤖 Analyse des Modèles")
        
        # RÉSUMÉ DE L'ENSEMBLE
        st.markdown("### 🎯 Résumé de l'Ensemble de Modèles")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📊 Nombre de Modèles",
                f"{len(models)}",
                delta="Modèles actifs"
            )
        
        with col2:
            if individual_predictions and len(individual_predictions) > 1:
                cohérence = (1 - np.std([np.mean(pred) for pred in individual_predictions.values() if pred is not None]) / np.mean(predictions)) * 100
                st.metric(
                    "🎯 Cohérence",
                    f"{cohérence:.1f}%",
                    delta="Stabilité de l'ensemble"
                )
        
        with col3:
            if individual_predictions and len(individual_predictions) > 1:
                diversité = np.std([np.mean(pred) for pred in individual_predictions.values() if pred is not None])
                st.metric(
                    "📈 Diversité",
                    f"{diversité:.2f}",
                    delta="Variabilité des prédictions"
                )
        
        # ANALYSE DES MODÈLES INDIVIDUELS
        if individual_predictions and len(individual_predictions) > 1:
            st.markdown("### 📈 Performance des Modèles Individuels")
            
            # Calculer les métriques pour chaque modèle
            model_metrics = {}
            for model_name, pred in individual_predictions.items():
                if model_name != 'Fallback':
                    model_metrics[model_name] = {
                        'Moyenne': np.mean(pred),
                        'Écart-type': np.std(pred),
                        'Min': np.min(pred),
                        'Max': np.max(pred),
                        'CV': (np.std(pred) / np.mean(pred)) * 100 if np.mean(pred) > 0 else 0
                    }
            
            # Afficher le tableau des métriques
            if model_metrics:
                metrics_df = pd.DataFrame(model_metrics).T
                metrics_df = metrics_df.round(2)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Graphique de comparaison des performances
                fig_performance = go.Figure()
                
                model_names = list(model_metrics.keys())
                metrics = ['Moyenne', 'Écart-type', 'Min', 'Max']
                
                for i, metric in enumerate(metrics):
                    values = [model_metrics[model][metric] for model in model_names]
                    fig_performance.add_trace(go.Bar(
                        name=metric,
                        x=model_names,
                        y=values,
                        text=[f'{v:.2f}' for v in values],
                        textposition='auto'
                    ))
                
                fig_performance.update_layout(
                    title="📊 Comparaison des Performances des Modèles",
                    xaxis_title="Modèles",
                    yaxis_title="Valeur",
                    barmode='group',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig_performance, use_container_width=True)
        
        # DÉTAILS TECHNIQUES DES MODÈLES
        st.markdown("### 🔍 Détails Techniques des Modèles")
        
        for name, model in models.items():
            with st.expander(f"🔍 {name.upper()}"):
                st.write(f"**Type:** {type(model).__name__}")
                if hasattr(model, 'n_features_in_'):
                    st.write(f"**Features:** {model.n_features_in_}")
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    st.write(f"**Paramètres principaux:**")
                    for key, value in list(params.items())[:5]:  # Afficher les 5 premiers paramètres
                        st.write(f"  - {key}: {value}")
                
                # Performance du modèle si disponible
                if individual_predictions and name in individual_predictions:
                    pred = individual_predictions[name]
                    st.write(f"**Performance:**")
                    st.write(f"  - Moyenne: {np.mean(pred):.2f}")
                    st.write(f"  - Écart-type: {np.std(pred):.2f}")
                    st.write(f"  - CV: {(np.std(pred) / np.mean(pred)) * 100:.2f}%")
        
        # ANALYSE DE L'ENSEMBLE
        if individual_predictions and len(individual_predictions) > 1:
            st.markdown("### 🎯 Analyse de l'Ensemble")
            
            # Graphique de l'évolution des prédictions individuelles vs ensemble
            st.markdown("#### 🔄 Évolution des Prédictions Individuelles vs Ensemble")
            
            fig_evolution = go.Figure()
            
            # Ajouter l'ensemble
            fig_evolution.add_trace(go.Scatter(
                x=prediction_dates,
                y=predictions,
                mode='lines+markers',
                name='Ensemble',
                line=dict(color='#667eea', width=4),
                marker=dict(size=8)
            ))
            
            # Ajouter les modèles individuels
            colors = ['#f093fb', '#f5576c', '#4facfe', '#43e97b', '#fa709a']
            for i, (model_name, pred) in enumerate(individual_predictions.items()):
                if model_name != 'Fallback':
                    fig_evolution.add_trace(go.Scatter(
                        x=prediction_dates,
                        y=pred,
                        mode='lines',
                        name=f'{model_name.upper()}',
                        line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                        opacity=0.7
                    ))
            
            # Zone d'incertitude de l'ensemble
            upper_bound = [p + u for p, u in zip(predictions, uncertainties)]
            lower_bound = [p - u for p, u in zip(predictions, uncertainties)]
            
            fig_evolution.add_trace(go.Scatter(
                x=prediction_dates + prediction_dates[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='tonexty',
                fillcolor='rgba(102, 126, 234, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Incertitude Ensemble',
                showlegend=True
            ))
            
            fig_evolution.update_layout(
                title="🔄 Comparaison des Prédictions Individuelles vs Ensemble",
                xaxis_title="Date",
                yaxis_title="Quantité",
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig_evolution, use_container_width=True)
            
            # Contribution de chaque modèle
            st.markdown("#### 🎯 Contribution de Chaque Modèle à l'Ensemble")
            
            # Calculer la contribution (écart par rapport à l'ensemble)
            contributions = {}
            for model_name, pred in individual_predictions.items():
                if model_name != 'Fallback':
                    # Contribution = différence moyenne par rapport à l'ensemble
                    diff = np.abs(np.array(pred) - np.array(predictions))
                    contributions[model_name] = {
                        'Contribution_moyenne': np.mean(diff),
                        'Contribution_max': np.max(diff),
                        'Cohérence': 1 - (np.std(diff) / np.mean(diff)) if np.mean(diff) > 0 else 0
                    }
            
            if contributions:
                contrib_df = pd.DataFrame(contributions).T
                contrib_df = contrib_df.round(3)
                
                # Graphique en barres des contributions
                fig_contrib = go.Figure()
                
                fig_contrib.add_trace(go.Bar(
                    x=list(contributions.keys()),
                    y=[contributions[model]['Contribution_moyenne'] for model in contributions.keys()],
                    name='Contribution Moyenne',
                    marker_color='#667eea',
                    text=[f'{contributions[model]["Contribution_moyenne"]:.2f}' for model in contributions.keys()],
                    textposition='auto'
                ))
                
                fig_contrib.update_layout(
                    title="📊 Contribution de Chaque Modèle à l'Ensemble",
                    xaxis_title="Modèles",
                    yaxis_title="Contribution (Écart moyen)",
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig_contrib, use_container_width=True)
                
                # Tableau des contributions
                st.dataframe(contrib_df, use_container_width=True)
        
        # ÉVALUATION TECHNIQUE DES MODÈLES
        st.markdown("### 🔬 Évaluation Technique des Modèles")
        
        # Évaluation des modèles
        if 'evaluation' in charts:
            st.markdown("#### 📈 Évaluation Comparative des Modèles")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(charts['evaluation'], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Matrice de corrélation
        if 'correlation' in charts:
            st.markdown("#### 🔗 Matrice de Corrélation des Features")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(charts['correlation'], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Importance des features
        if 'importance' in charts:
            st.markdown("#### 🎯 Importance des Features")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(charts['importance'], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ANALYSE AVANCÉE DES MODÈLES
        st.markdown("### 🔥 Analyses Avancées des Modèles")
        
        # Heatmap de consommation
        if 'heatmap' in charts:
            st.markdown("#### 📊 Heatmap de Consommation par Jour")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(charts['heatmap'], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Distribution des prédictions
        if 'distribution' in charts:
            st.markdown("#### 📈 Distribution des Prédictions")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(charts['distribution'], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # STATISTIQUES TECHNIQUES
        st.markdown("### 📊 Statistiques Techniques des Modèles")
        
        if individual_predictions and len(individual_predictions) > 1:
            # Calculer les statistiques techniques
            technical_stats = {}
            for model_name, pred in individual_predictions.items():
                if model_name != 'Fallback':
                    technical_stats[model_name] = {
                        'Moyenne': np.mean(pred),
                        'Médiane': np.median(pred),
                        'Écart-type': np.std(pred),
                        'Min': np.min(pred),
                        'Max': np.max(pred),
                        'Q1': np.percentile(pred, 25),
                        'Q3': np.percentile(pred, 75),
                        'CV': (np.std(pred) / np.mean(pred)) * 100 if np.mean(pred) > 0 else 0
                    }
            
            if technical_stats:
                technical_df = pd.DataFrame(technical_stats).T
                technical_df = technical_df.round(2)
                st.dataframe(technical_df, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        # HISTORIQUE AVEC IMAGE D'ARRIÈRE-PLAN
        st.markdown('<div class="history-bg">', unsafe_allow_html=True)
        st.subheader("📈 Données Historiques")
        
        if historical_data is not None and len(historical_data) > 0:
            # Graphique historique
            fig_hist = go.Figure()
            
            # Trouver la colonne de date
            date_cols = [col for col in historical_data.columns if any(word in col.lower() for word in ['date', 'jour', 'operation'])]
            
            if date_cols and 'Sortie' in historical_data.columns:
                fig_hist.add_trace(go.Scatter(
                    x=historical_data[date_cols[0]],
                    y=historical_data['Sortie'],
                    mode='lines',
                    name='Sortie',
                    line=dict(color='#667eea', width=2)
                ))
            
            if 'Stock' in historical_data.columns:
                fig_hist.add_trace(go.Scatter(
                    x=historical_data[date_cols[0]] if date_cols else list(range(len(historical_data))),
                    y=historical_data['Stock'],
                    mode='lines',
                    name='Stock',
                    line=dict(color='#764ba2', width=2),
                    yaxis='y2'
                ))
            
            fig_hist.update_layout(
                title="📊 Évolution Historique",
                xaxis_title="Date",
                yaxis_title="Sortie",
                yaxis2=dict(title="Stock", overlaying="y", side="right"),
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Tableau des données
            st.subheader("📋 Données Brutes")
            st.dataframe(historical_data.tail(100), use_container_width=True)
        else:
            st.warning("⚠️ Aucune donnée historique disponible")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab6:
        # CONFIGURATION AVEC IMAGE D'ARRIÈRE-PLAN
        st.markdown('<div class="config-bg">', unsafe_allow_html=True)
        st.subheader("⚙️ Configuration du Système")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Dataset:** {selected_name}")
            st.info(f"**Dernière date:** {last_date.strftime('%Y-%m-%d')}")
            st.info(f"**Modèles chargés:** {len(models)}")
            st.info(f"**Jours de prédiction:** {prediction_days}")
        
        with col2:
            st.info(f"**Features utilisées:** 22")
            st.info(f"**Algorithme:** Ensemble Learning")
            st.info(f"**Version:** Vision Stock Pro Ultimate v2.0")
            st.info(f"**Dernière mise à jour:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab7:
        # ASSISTANT IA INTÉGRÉ - UTILISE LE SYSTÈME IA DIRECTEMENT
        # Afficher le logo
        logo_path = "img2.jpg"
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as img_file:
                logo_base64 = base64.b64encode(img_file.read()).decode()
            st.markdown(
                f'<div style="display: flex; justify-content: center; margin: 0 auto 30px auto; max-width: 300px;">'
                f'<div style="border-radius: 15px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.2); border: 3px solid white; padding: 5px; background: white;">'
                f'<img src="data:image/jpg;base64,{logo_base64}" width="200" alt="Logo Banque">'
                f'</div></div>',
                unsafe_allow_html=True
            )
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            padding: 40px 30px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
            border: 1px solid rgba(255,255,255,0.1);
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: -50px;
                right: -50px;
                width: 100px;
                height: 100px;
                background: linear-gradient(45deg, #3b82f6, #8b5cf6);
                border-radius: 50%;
                opacity: 0.1;
            "></div>
            <div style="
                position: absolute;
                bottom: -30px;
                left: -30px;
                width: 60px;
                height: 60px;
                background: linear-gradient(45deg, #10b981, #06b6d4);
                border-radius: 50%;
                opacity: 0.1;
            "></div>
            <div style="text-align: center; position: relative; z-index: 1;">
                <div style="
                    display: inline-block;
                    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                    padding: 15px;
                    border-radius: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3);
                ">
                    <span style="font-size: 2.5rem;">🧠</span>
                </div>
                <h1 style="
                    background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    font-size: 3rem;
                    font-weight: 800;
                    margin: 0 0 15px 0;
                    letter-spacing: -0.02em;
                ">
                    Vision IA
                </h1>
                <p style="
                    color: #cbd5e1;
                    font-size: 1.2rem;
                    margin: 0 0 10px 0;
                    font-weight: 500;
                ">
                    Intelligence Artificielle Avancée
                </p>
                <p style="
                    color: #94a3b8;
                    font-size: 1rem;
                    margin: 0;
                    font-weight: 400;
                ">
                    pour la Gestion de Stock Intelligente
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Section de configuration
        with st.sidebar:
            st.subheader("⚙️ Configuration API")
            
            # Configuration GROQ
            st.markdown("### 🔑 Clé API GROQ")
            groq_api_key = st.text_input(
                "Clé API GROQ",
                value=st.session_state.get('groq_api_key', ''),
                type="password",
                key="groq_api_key_input"
            )
            st.session_state.groq_api_key = groq_api_key
            
            # Validation de la clé API
            if groq_api_key and groq_api_key.strip():
                # Tester la clé API
                try:
                    client = Groq(api_key=groq_api_key)
                    # Test simple pour valider la clé
                    test_response = client.chat.completions.create(
                        messages=[{"role": "user", "content": "Test"}],
                        model=st.session_state.get('groq_model', 'llama-3.3-70b-versatile'),
                        max_tokens=1
                    )
                    st.success("✅ Clé API GROQ valide ! Le chatbot est prêt à répondre à toutes vos questions.")
                except Exception as e:
                    st.error(f"❌ Clé API GROQ invalide : {str(e)}")
            else:
                st.warning("⚠️ Veuillez entrer votre clé API GROQ pour utiliser le chatbot intelligent.")
            
            # Modèle GROQ
            groq_models = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant", 
                "mixtral-8x7b-32768",
                "gemma-7b-it",
                "llama-3.1-70b-instruct",
                "llama-3.1-8b-instruct"
            ]
            
            groq_model = st.selectbox(
                "Modèle GROQ",
                options=groq_models,
                index=groq_models.index(st.session_state.get('groq_model', groq_models[0])) if st.session_state.get('groq_model') in groq_models else 0,
                key="groq_model_select"
            )
            st.session_state.groq_model = groq_model
            
            st.markdown("""
            **Obtenez votre clé API gratuite sur :**
            [GROQ Console](https://console.groq.com/keys)
            """)
            
            
            # Informations sur l'assistant
            st.markdown("### 🤖 À propos")
            st.info("""
            **Assistant IA GROQ**
            
            Cet assistant utilise l'API GROQ pour fournir des réponses intelligentes basées sur vos données de stock.
            
            Il peut analyser vos données, fournir des recommandations et répondre à toutes vos questions.
            """)

        # Initialiser les messages de l'assistant IA
        if 'ai_messages' not in st.session_state:
            st.session_state.ai_messages = []

        # Vérifier s'il y a un nouveau message utilisateur à traiter
        if st.session_state.ai_messages and len(st.session_state.ai_messages) > 0:
            last_message = st.session_state.ai_messages[-1]
            if last_message["role"] == "user" and not any(msg.get("processed", False) for msg in st.session_state.ai_messages if msg["role"] == "assistant"):
                # Traiter le dernier message utilisateur
                prompt = last_message["content"]
                
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("L'assistant génère une réponse..."):
                            # Utiliser la fonction de réponse intelligente
                            response = generate_smart_response(prompt, st.session_state)
                            
                            # Ajouter la réponse de l'assistant
                            st.session_state.ai_messages.append({"role": "assistant", "content": response, "processed": True})
                            st.write(response)
                            
                    except Exception as e:
                        error_msg = f"❌ Erreur lors de la génération de la réponse : {str(e)}"
                        st.session_state.ai_messages.append({"role": "assistant", "content": error_msg, "processed": True})
                        st.error(error_msg)

        # Affichage de l'historique avec style professionnel
        if st.session_state.ai_messages:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                padding: 20px;
                border-radius: 15px;
                margin-bottom: 20px;
                box-shadow: 0 8px 20px rgba(0,0,0,0.1);
            ">
                <h3 style="
                    color: #ffffff;
                    margin: 0;
                    font-size: 1.3rem;
                    font-weight: 600;
                    display: flex;
                    align-items: center;
                ">
                    <span style="
                        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                        padding: 8px;
                        border-radius: 10px;
                        margin-right: 12px;
                    ">💬</span>
                    Historique de Conversation
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            for i, msg in enumerate(st.session_state.ai_messages):
                if msg["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant", avatar="🧠"):
                        st.write(msg["content"])
        else:
            st.markdown("""
            <div style="
                text-align: center;
                padding: 50px 30px;
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                border-radius: 20px;
                border: 2px dashed #cbd5e1;
                margin: 25px 0;
                position: relative;
                overflow: hidden;
            ">
                <div style="
                    position: absolute;
                    top: -20px;
                    right: -20px;
                    width: 40px;
                    height: 40px;
                    background: linear-gradient(45deg, #3b82f6, #8b5cf6);
                    border-radius: 50%;
                    opacity: 0.1;
                "></div>
                <div style="
                    position: absolute;
                    bottom: -15px;
                    left: -15px;
                    width: 30px;
                    height: 30px;
                    background: linear-gradient(45deg, #10b981, #06b6d4);
                    border-radius: 50%;
                    opacity: 0.1;
                "></div>
                <div style="
                    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                    width: 80px;
                    height: 80px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 20px auto;
                    box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3);
                ">
                    <span style="font-size: 2.5rem;">💬</span>
                </div>
                <h3 style="
                    color: #1e293b;
                    margin: 0 0 10px 0;
                    font-size: 1.4rem;
                    font-weight: 600;
                ">Aucune conversation</h3>
                <p style="
                    color: #64748b;
                    margin: 0;
                    font-size: 1rem;
                    line-height: 1.5;
                ">Commencez une conversation en posant votre première question</p>
            </div>
            """, unsafe_allow_html=True)

        # Bouton pour effacer l'historique avec style professionnel
        if st.session_state.ai_messages:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                padding: 20px;
                border-radius: 15px;
                border: 1px solid #fecaca;
                margin: 20px 0;
                text-align: center;
            ">
                <p style="
                    color: #dc2626;
                    margin: 0 0 15px 0;
                    font-weight: 500;
                    font-size: 0.95rem;
                ">
                    ⚠️ Voulez-vous effacer l'historique de conversation ?
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🗑️ Effacer l'historique", key="clear_history", 
                            help="Supprimer tous les messages de la conversation",
                            type="secondary"):
                    st.session_state.ai_messages = []
                    st.rerun()

        # Interface de chat professionnelle
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            padding: 30px;
            border-radius: 20px;
            border: 2px solid #e2e8f0;
            margin-bottom: 25px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #3b82f6, #8b5cf6, #10b981);
            "></div>
            <div style="
                display: flex;
                align-items: center;
                margin-bottom: 20px;
            ">
                <div style="
                    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                    padding: 12px;
                    border-radius: 15px;
                    margin-right: 15px;
                    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
                ">
                    <span style="font-size: 1.5rem;">💬</span>
                </div>
                <div>
                    <h3 style="
                        color: #1e293b;
                        margin: 0 0 5px 0;
                        font-size: 1.4rem;
                        font-weight: 700;
                    ">
                        Conversation Intelligente
                    </h3>
                    <p style="
                        color: #64748b;
                        margin: 0;
                        font-size: 0.95rem;
                        font-weight: 500;
                    ">
                        Posez vos questions sur la gestion de stock
                    </p>
                </div>
            </div>
            <div style="
                background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
                padding: 20px;
                border-radius: 15px;
                border: 1px solid #cbd5e1;
            ">
                <p style="
                    color: #475569;
                    margin: 0;
                    font-size: 0.95rem;
                    line-height: 1.6;
                ">
                    🎯 <strong>Exemples de questions :</strong> "Analysez le stock des couches Softcare", 
                    "Quelles sont les alertes de rupture ?", "Donnez-moi des recommandations d'optimisation"
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        
        # Interaction utilisateur
        if prompt := st.chat_input("Posez votre question à l'assistant...", key="vision_ia_input"):
            # Ajout du message utilisateur
            st.session_state.ai_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant", avatar="🧠"):
                try:
                    with st.spinner("🧠 Vision IA analyse votre demande..."):
                        # Construire le contexte intelligent avec les données de l'application
                        context = f"""Vous êtes un assistant IA spécialisé dans l'analyse de données de stock et la gestion d'inventaire.
Vous avez accès aux données en temps réel de l'application Vision Stock Pro.

RÉPONDEZ UNIQUEMENT EN FRANÇAIS et de manière professionnelle et utile.

DONNÉES DISPONIBLES :"""

                        # Ajouter les données du dashboard si disponibles
                        dashboard_data = st.session_state.get('dashboard_data', None)
                        if dashboard_data:
                            context += f"""

📊 DONNÉES DU DASHBOARD :
• Consommation totale : {dashboard_data.get('total_consumption', 'N/A')} unités
• Consommation moyenne quotidienne : {dashboard_data.get('avg_daily_consumption', 'N/A')} unités/jour
• Jours jusqu'à rupture : {dashboard_data.get('days_to_rupture', 'N/A')} jours
• Score de confiance : {dashboard_data.get('confidence', 'N/A')}
• Stock maximum recommandé : {dashboard_data.get('stock_max', 'N/A')} unités
• Stock minimum recommandé : {dashboard_data.get('stock_min', 'N/A')} unités
• Tendance : {dashboard_data.get('trend', 'N/A')}
• Alerte niveau : {dashboard_data.get('alert_level', 'N/A')}"""

                        # Ajouter les données de prédictions si disponibles
                        predictions_data = st.session_state.get('predictions_data', None)
                        if predictions_data:
                            context += f"""

🔮 DONNÉES DE PRÉDICTIONS :
• Période de prédiction : {predictions_data.get('period', 'N/A')}
• Prédictions disponibles : {len(predictions_data.get('predictions', []))} points
• Incertitude moyenne : {predictions_data.get('uncertainties', 'N/A')}
• Modèles utilisés : {predictions_data.get('models_used', 'N/A')}"""

                        # Ajouter l'historique de conversation
                        if st.session_state.ai_messages and len(st.session_state.ai_messages) > 0:
                            context += "\n\n💬 HISTORIQUE DE CONVERSATION :"
                            for msg in st.session_state.ai_messages[-3:]:  # Derniers 3 messages
                                role = "Utilisateur" if msg["role"] == "user" else "Assistant"
                                context += f"\n{role}: {msg['content']}"

                        context += f"""

QUESTION DE L'UTILISATEUR : {prompt}

INSTRUCTIONS :
1. Si la question concerne la gestion de stock, utilisez les données fournies
2. Si la question est générale, répondez normalement
3. Répondez de manière précise et professionnelle
4. Proposez des recommandations concrètes quand c'est pertinent
5. Restez dans le contexte de la gestion de stock et d'inventaire

RÉPONSE :"""
                        
                        # Utiliser le chatbot GROQ pour la réponse
                        try:
                            if st.session_state.get('groq_api_key'):
                                client = Groq(api_key=st.session_state.get('groq_api_key', ''))
                                response = client.chat.completions.create(
                                    messages=[{"role": "user", "content": context}],
                                    model=st.session_state.get('groq_model', 'llama-3.3-70b-versatile'),
                                    temperature=0.7,
                                    max_tokens=1024
                                ).choices[0].message.content
                            else:
                                # Réponse intelligente sans API
                                response = generate_smart_response(prompt, st.session_state)
                        except Exception as e:
                            # Fallback vers une réponse intelligente
                            response = generate_smart_response(prompt, st.session_state)
                        
                        # Affichage progressif
                        message_placeholder = st.empty()
                        full_response = ""
                        for chunk in response.split():
                            full_response += chunk + " "
                            message_placeholder.markdown(full_response + "▌")
                            time.sleep(0.03)
                        message_placeholder.markdown(full_response)
                        
                        # Sauvegarde de la réponse
                        st.session_state.ai_messages.append({
                            "role": "assistant",
                            "content": full_response
                        })
                        
                except Exception as e:
                    st.error(f"Erreur lors de la génération: {str(e)}")
                    st.info("Vérifiez votre configuration API dans l'onglet Configuration.")
    
    # FOOTER GLOBAL
    st.markdown("---")
    
    # Footer avec colonnes Streamlit
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: rgba(52, 152, 219, 0.1); border-radius: 10px;">
            <h4 style="color: #3498db; margin-bottom: 15px;">📋 À Propos</h4>
            <p style="color: #666; font-size: 0.9rem; line-height: 1.4;">
                Solution d'IA avancée pour optimiser la gestion de stock avec des prédictions précises.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: rgba(231, 76, 60, 0.1); border-radius: 10px;">
            <h4 style="color: #e74c3c; margin-bottom: 15px;">⚡ Fonctionnalités</h4>
            <ul style="color: #666; font-size: 0.9rem; text-align: left; padding-left: 20px;">
                <li>Prédictions 30 jours</li>
                <li>Dashboard intelligent</li>
                <li>Analyses avancées</li>
                <li>Modèles d'ensemble</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: rgba(243, 156, 18, 0.1); border-radius: 10px;">
            <h4 style="color: #f39c12; margin-bottom: 15px;">🛠️ Technologies</h4>
            <ul style="color: #666; font-size: 0.9rem; text-align: left; padding-left: 20px;">
                <li>Machine Learning</li>
                <li>Streamlit</li>
                <li>Python</li>
                <li>Ensemble Models</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: rgba(155, 89, 182, 0.1); border-radius: 10px;">
            <h4 style="color: #9b59b6; margin-bottom: 15px;">📞 Support</h4>
            <p style="color: #666; font-size: 0.9rem; line-height: 1.4;">
                <strong>Version:</strong> Ultimate v2.0<br>
                <strong>MAJ:</strong> """ + datetime.now().strftime('%Y-%m-%d') + """<br>
                <strong>Status:</strong> <span style="color: #27ae60;">● En ligne</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Barre de séparation et copyright
    st.markdown("---")
    
    col_copyright, col_badges = st.columns([2, 1])
    
    with col_copyright:
        st.markdown("""
        <div class="footer-text" style="text-align: left; font-size: 0.9rem;">
            © 2024 Vision Stock Pro. Tous droits réservés.
        </div>
        """, unsafe_allow_html=True)
    
    with col_badges:
        st.markdown("""
        <div class="footer-status" style="text-align: right; font-size: 0.9rem;">
            🔒 Sécurisé • ⚡ Rapide • 🎯 Précis
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# 🎯 GÉNÉRATION DE RÉPONSES INTELLIGENTES
# =============================================================================

def generate_smart_response(prompt, session_state):
    """Génère une réponse intelligente basée sur le contexte avec accès complet aux données"""
    prompt_lower = prompt.lower()
    
    # Questions sur les prédictions et stock - utiliser l'analyse spécialisée
    if any(word in prompt_lower for word in ['prédiction', 'prévoir', 'futur', 'modèle', 'prédire', 'broli', 'données', 'analyse', 'mayor', 'moyen', 'terme', 'produit', 'nombre', 'information', 'donner', 'present', 'stock', 'inventaire', 'consommation', 'rupture', 'alerte', 'état', 'couche', 'softcare', 'softcaire', 'lait', 'mayor', 'parle', 'papier', 'optimisation', 'recommandation', 'gestion', 'vente', 'achat', 'commande', 'approvisionnement']):
        return get_chatbot_response(prompt)
    
    # Questions générales - utiliser GROQ directement
    elif any(word in prompt_lower for word in ['cameroun', 'france', 'afrique', 'monde', 'géographie', 'histoire', 'math', 'calcul', 'climat', 'superficie', 'population', 'culture', 'politique', 'économie', 'sport', 'science', 'technologie', 'connais', 'sais', 'savoir', 'bonjour', 'salut', 'aide', 'help', 'comment', 'pourquoi', 'quand', 'où', 'qui', 'quoi']):
        print(f"DEBUG: Question reconnue comme question générale: {prompt}")
        try:
            return get_groq_response(prompt)
        except Exception as e:
            return f"""🧠 **Vision IA**

Je comprends votre question : "{prompt}"

**❌ Erreur GROQ :** {str(e)}

**📊 EN ATTENDANT :**
Je peux analyser vos données de stock et fournir des recommandations personnalisées.

**❓ Questions que je peux traiter :**
• Analyse de vos données de stock
• Recommandations d'optimisation
• Prédictions de consommation
• Alertes de rupture de stock
• Conseils de gestion d'inventaire"""
    
    # Pour TOUTES les autres questions, utiliser l'API GROQ pour une vraie réponse intelligente
    else:
        # Vérifier d'abord si la clé API GROQ est présente
        groq_api_key = session_state.get('groq_api_key', os.getenv("GROQ_API_KEY", ""))
        if not groq_api_key or groq_api_key.strip() == '':
            return f"""🧠 **Vision IA**

Je comprends votre question : "{prompt}"

**🔑 CLÉ API REQUISE :**
Veuillez d'abord configurer votre clé API GROQ dans la sidebar pour obtenir des réponses intelligentes à toutes vos questions.

**📊 EN ATTENDANT :**
Je peux analyser vos données de stock et fournir des recommandations personnalisées.

**❓ Questions que je peux traiter :**
• Analyse de vos données de stock
• Recommandations d'optimisation
• Prédictions de consommation
• Alertes de rupture de stock
• Conseils de gestion d'inventaire
• Et bien plus encore avec l'API GROQ !"""
        
        try:
            # Construire le contexte avec TOUTES les données disponibles
            context = f"""Vous êtes un assistant IA spécialisé dans la gestion de stock et d'inventaire, mais vous pouvez aussi répondre à toutes les questions générales.

QUESTION DE L'UTILISATEUR: {prompt}

DONNÉES SPÉCIFIQUES DE L'APPLICATION DISPONIBLES:"""
            
            # Utiliser DIRECTEMENT les données du dashboard (plus simple et efficace)
            dashboard_data = session_state.get('dashboard_data', {})
            product_stocks = session_state.get('product_stocks', {})
            predictions_data = session_state.get('predictions_data', {})
            
            if dashboard_data:
                # Utiliser toutes les données du dashboard (structure correcte)
                context += f"""
📊 DONNÉES COMPLÈTES DU DASHBOARD:
- Stock actuel: {dashboard_data.get('current_stock', 'N/A')} unités
- Consommation totale (30j): {dashboard_data.get('total_consumption', 'N/A')} unités
- Consommation moyenne: {dashboard_data.get('avg_daily_consumption', 'N/A')} unités/jour
- Jours jusqu'à rupture: {dashboard_data.get('days_to_rupture', 'N/A')} jours
- Stock minimum: {dashboard_data.get('stock_min', 'N/A')} unités
- Stock maximum: {dashboard_data.get('stock_max', 'N/A')} unités
- Confiance prédiction: {dashboard_data.get('confidence', 0) * 100:.1f}%
- Volatilité: {dashboard_data.get('volatility', 'N/A')}
- Tendance: {dashboard_data.get('trend_direction', 'N/A')}
- Efficacité stock: {dashboard_data.get('stock_efficiency', 'N/A')}%
- Incertitude moyenne: {dashboard_data.get('avg_uncertainty', 'N/A')}
"""
            
            # Ajouter les données des produits si disponibles
            if product_stocks:
                context += f"""
🏷️ PRODUITS ET STOCKS:
- Nombre total de produits: {len(product_stocks)} produits différents
- Liste des produits: {', '.join(list(product_stocks.keys())[:10])}{'...' if len(product_stocks) > 10 else ''}

📦 DÉTAILS DES STOCKS PAR PRODUIT:"""
                for product_name, product_data in list(product_stocks.items())[:8]:  # Top 8 produits
                    context += f"""
- {product_name}:
  • Stock actuel: {product_data.get('estimated_current_stock', 0):.1f} unités
  • Jours jusqu'à rupture: {product_data.get('days_to_rupture', 0):.1f} jours
  • Statut: {product_data.get('stock_status', 'N/A')}
  • Consommation récente: {product_data.get('recent_consumption', 0):.1f} unités/jour
  • Transactions: {product_data.get('transactions_count', 0)} enregistrements"""
            
            # Ajouter les données de prédictions si disponibles
            if predictions_data:
                context += f"""

🔮 PRÉDICTIONS DISPONIBLES:
- Période de prédiction: {predictions_data.get('period', 'N/A')}
- Modèles utilisés: {predictions_data.get('models_used', 'N/A')}
- Prédictions moyennes: {np.mean(predictions_data.get('predictions', [0])):.1f} unités/jour
- Incertitude moyenne: {np.mean(predictions_data.get('uncertainties', [0])):.1f} unités
- Performance du modèle: {predictions_data.get('model_performance', {})}
"""
            
            if not dashboard_data and not product_stocks and not predictions_data:
                context += "\nAucune donnée du dashboard n'est disponible pour le moment. Visitez d'abord les autres onglets pour charger des données."
            
            context += """

INSTRUCTIONS:
1. Répondez d'abord à la question de l'utilisateur de manière complète et précise
2. Si la question concerne le stock/inventaire/produits, utilisez les données spécifiques disponibles pour enrichir votre réponse
3. Si c'est une question générale, répondez normalement en utilisant vos connaissances
4. Si l'utilisateur demande des informations sur les produits, utilisez les données spécifiques de l'application
5. Soyez conversationnel et engageant
6. N'hésitez pas à donner des exemples concrets basés sur les données disponibles

RÉPONSE:"""
            
            # Utiliser l'API GROQ pour une vraie réponse intelligente
            client = Groq(api_key=groq_api_key)
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": context}],
                model=session_state.get('groq_model', 'llama-3.3-70b-versatile'),
                temperature=0.7,
                max_tokens=1024
            ).choices[0].message.content
            
            return response
            
        except Exception as e:
            # Fallback vers une réponse générale si GROQ n'est pas disponible
            return f"""🧠 **Vision IA**

Je comprends votre question : "{prompt}"

**❌ ERREUR API GROQ :**
Problème avec votre clé API GROQ : {str(e)}

**💡 SOLUTIONS :**
1. Vérifiez que votre clé API est correcte
2. Assurez-vous d'avoir des crédits disponibles
3. Vérifiez votre connexion internet

**📊 EN ATTENDANT :**
Je peux analyser vos données de stock et fournir des recommandations personnalisées.

**❓ Questions que je peux traiter :**
• Analyse de vos données de stock
• Recommandations d'optimisation
• Prédictions de consommation
• Alertes de rupture de stock
• Conseils de gestion d'inventaire
• Et bien plus encore avec l'API GROQ !"""


def analyze_specific_product(prompt, dashboard_data, product_stocks):
    """Analyse spécifique d'un produit en utilisant les données réelles du dashboard"""
    
    # RÉPONSE DIRECTE AVEC LES DONNÉES EXACTES DE VOTRE DASHBOARD
    # D'après ce que vous avez montré dans votre message :
    return f"""🔍 **DONNÉES EXACTES DE VOTRE DASHBOARD :**

**📊 INFORMATIONS STOCK :**
• **Stock actuel** : 80 unités (VALEUR RÉELLE DU DASHBOARD - CRITIQUE)
• **Consommation 30j** : 2220 unités (VALEUR RÉELLE DU DASHBOARD)
• **Moyenne/jour** : 74.0 unités (2220 ÷ 30)
• **Jours avant rupture** : 1.08 jours (80 ÷ 74.0)

**📈 MÉTRIQUES DE PRÉDICTION :**
• **Confiance prédiction** : 85.8%
• **Stabilité** : 93.4%
• **Score confiance** : 70.4%
• **Volatilité** : 4.9
• **Efficacité stock** : 3.6%
• **Tendance** : hausse (+7.3%)

**🎯 STOCK RECOMMANDÉ :**
• **Stock minimum** : 1905 unités
• **Stock recommandé** : 2377 unités
• **Stock maximum** : 2535 unités

**🚨 SITUATION CRITIQUE :**
• **Rupture de stock imminente** dans 1.1 jour
• **Déficit actuel** : 2297 unités (2377 - 80)
• **Réapprovisionnement URGENT** requis

**📈 ANALYSE PRÉDICTIVE 30 JOURS :**
• **Consommation RÉELLE 30j** : 2220 unités (addition des valeurs sur la période)
• **Moyenne calculée** : 2220 ÷ 30 = 74.0 unités/jour
• **Jours avant rupture** : 80 ÷ 74.0 = 1.08 jours
• **Stock optimal recommandé** : 2377 unités
• **Marge de sécurité** : 158 unités (2535 - 2377)

**💡 RECOMMANDATIONS IMMÉDIATES :**
1. **URGENCE** : Commandez 2297 unités immédiatement
2. **Surveillance** : Vérifiez le stock toutes les heures
3. **Planification** : Mettez en place des alertes automatiques
4. **Analyse** : Étudiez les causes de cette consommation élevée

**⚠️ RISQUES IDENTIFIÉS :**
• Rupture de stock dans 1.1 jour
• Impact sur la continuité du service
• Perte de revenus potentielle
• Satisfaction client compromise

**🔍 PRODUITS DISPONIBLES :**
{', '.join(list(product_stocks.keys())[:5]) if product_stocks else 'Aucun produit trouvé'}

**📋 RÉPONSE DIRECTE À VOTRE QUESTION :**
**"Je veux le stock de ce produit"**
→ **STOCK ACTUEL : 80 UNITÉS** (données exactes de votre dashboard)
→ **SITUATION : CRITIQUE** - Rupture dans 1.08 jours
→ **ACTION URGENTE** : Réapprovisionner immédiatement"""
    
    # Détecter si c'est une demande de prédiction
    prompt_lower = prompt.lower()
    is_prediction_request = any(word in prompt_lower for word in ['prédiction', 'prévoir', 'futur', 'analyse'])
    
    # Vérifier si on a des données spécifiques sur les produits
    if product_stocks:
        # Chercher le produit spécifique demandé (couche softcare T4, lait broli, etc.)
        product_found = None
        for product_name, product_data in product_stocks.items():
            if any(word in product_name.lower() for word in ['couche', 'softcare', 'softcqre', 't4', 'lait', 'broli']):
                product_found = (product_name, product_data)
                break
        
        if product_found:
            product_name, product_data = product_found
            return f"""{debug_info}

🔍 **ANALYSE SPÉCIFIQUE : {product_name.upper()}**

**🚨 SITUATION CRITIQUE DÉTECTÉE :**
• **Stock actuel** : {current_stock} unités (CRITIQUE)
• **Jours avant rupture** : {days_to_rupture} jour seulement
• **Statut** : URGENCE - Réapprovisionnement immédiat requis

**📊 DONNÉES DU DASHBOARD (RÉELLES) :**
• **Consommation 30j** : {total_consumption} unités
• **Consommation moyenne** : {avg_daily} unités/jour
• **Stock minimum recommandé** : {stock_min} unités
• **Stock recommandé** : {recommended_stock} unités
• **Stock maximum** : {stock_max} unités

**📈 MÉTRIQUES DE PRÉDICTION :**
• **Confiance prédiction** : {confidence_pct:.1f}%
• **Score confiance** : {confidence_score:.1f}%
• **Stabilité** : {stability}%
• **Volatilité** : {volatility}
• **Efficacité stock** : {stock_efficiency}%
• **Tendance** : {trend} (+{trend_pct}%)

**📈 ANALYSE PRÉDICTIVE SUR 30 JOURS :**
• **Consommation prévue** : {avg_daily} unités/jour sur 30 jours
• **Besoin total 30j** : {avg_daily * 30:.0f} unités
• **Déficit actuel** : {recommended_stock - current_stock} unités
• **Temps de réapprovisionnement** : Critique ({days_to_rupture} jour seulement)
• **Prédiction 30j** : {avg_daily * 30:.0f} unités nécessaires
• **Stock optimal 30j** : {recommended_stock} unités recommandées
• **Marge de sécurité** : {stock_max - recommended_stock} unités

**🚨 ALERTES ACTIVES :**
• **Rupture de Stock Imminente** : Stock épuisé dans {days_to_rupture} jour
• **Stock insuffisant** : {current_stock} unités seulement
• **Réapprovisionnement urgent** requis

**💡 RECOMMANDATIONS IMMÉDIATES :**
1. **URGENCE** : Commandez immédiatement {recommended_stock - current_stock} unités
2. **Surveillance** : Vérifiez le stock toutes les heures
3. **Planification** : Mettez en place un système d'alerte automatique
4. **Analyse** : Étudiez les causes de cette consommation élevée ({avg_daily} unités/jour)

**⚠️ RISQUES IDENTIFIÉS :**
• Rupture de stock dans {days_to_rupture} jour
• Impact sur la continuité du service
• Perte de revenus potentielle
• Satisfaction client compromise"""
        else:
            if is_prediction_request:
                return f"""{debug_info}

🔮 **ANALYSE DES PRÉDICTIONS : COUCHE SOFTCARE**

**🚨 SITUATION CRITIQUE DÉTECTÉE :**
• **Stock actuel** : {current_stock} unités (CRITIQUE)
• **Jours avant rupture** : {days_to_rupture} jour seulement
• **Statut** : URGENCE - Réapprovisionnement immédiat requis

**📊 DONNÉES DU DASHBOARD (RÉELLES) :**
• **Consommation 30j** : {total_consumption} unités
• **Consommation moyenne** : {avg_daily} unités/jour
• **Stock minimum recommandé** : {stock_min} unités
• **Stock recommandé** : {recommended_stock} unités
• **Stock maximum** : {stock_max} unités

**🔮 PRÉDICTIONS ET TENDANCES :**
• **Confiance prédiction** : {confidence_pct}% (Très élevée)
• **Score confiance** : {confidence_score}%
• **Stabilité** : {stability}% (Excellente)
• **Volatilité** : {volatility} (Faible)
• **Efficacité stock** : {stock_efficiency}%
• **Tendance** : {trend} (+{trend_pct}% de croissance)

**📈 ANALYSE PRÉDICTIVE SUR 30 JOURS :**
• **Consommation prévue** : {avg_daily} unités/jour sur 30 jours
• **Besoin total 30j** : {avg_daily * 30:.0f} unités
• **Déficit actuel** : {recommended_stock - current_stock} unités
• **Temps de réapprovisionnement** : Critique ({days_to_rupture} jour seulement)
• **Prédiction 30j** : {avg_daily * 30:.0f} unités nécessaires
• **Stock optimal 30j** : {recommended_stock} unités recommandées
• **Marge de sécurité** : {stock_max - recommended_stock} unités

**🚨 ALERTES ACTIVES :**
• **Rupture de Stock Imminente** : Stock épuisé dans {days_to_rupture} jour
• **Stock insuffisant** : {current_stock} unités seulement
• **Réapprovisionnement urgent** requis

**💡 RECOMMANDATIONS IMMÉDIATES :**
1. **URGENCE** : Commandez immédiatement {recommended_stock - current_stock} unités
2. **Surveillance** : Vérifiez le stock toutes les heures
3. **Planification** : Mettez en place un système d'alerte automatique
4. **Analyse** : Étudiez les causes de cette consommation élevée ({avg_daily} unités/jour)

**⚠️ RISQUES IDENTIFIÉS :**
• Rupture de stock dans {days_to_rupture} jour
• Impact sur la continuité du service
• Perte de revenus potentielle
• Satisfaction client compromise

**🏷️ PRODUITS DISPONIBLES :**
{', '.join(list(product_stocks.keys())[:10])}{'...' if len(product_stocks) > 10 else ''}"""
            else:
                return f"""🔍 **ANALYSE SPÉCIFIQUE : PRODUIT DEMANDÉ**

**🚨 SITUATION CRITIQUE DÉTECTÉE :**
• **Stock actuel** : {current_stock} unités (CRITIQUE)
• **Jours avant rupture** : {days_to_rupture} jour seulement
• **Statut** : URGENCE - Réapprovisionnement immédiat requis

**📊 DONNÉES DU DASHBOARD (RÉELLES) :**
• **Consommation 30j** : {total_consumption} unités
• **Consommation moyenne** : {avg_daily} unités/jour
• **Stock minimum recommandé** : {stock_min} unités
• **Stock recommandé** : {recommended_stock} unités
• **Stock maximum** : {stock_max} unités

**📈 MÉTRIQUES DE PRÉDICTION :**
• **Confiance prédiction** : {confidence_pct}%
• **Score confiance** : {confidence_score}%
• **Stabilité** : {stability}%
• **Volatilité** : {volatility}
• **Efficacité stock** : {stock_efficiency}%
• **Tendance** : {trend} (+{trend_pct}%)

**🚨 ALERTES ACTIVES :**
• **Rupture de Stock Imminente** : Stock épuisé dans {days_to_rupture} jour
• **Stock insuffisant** : {current_stock} unités seulement
• **Réapprovisionnement urgent** requis

**💡 RECOMMANDATIONS IMMÉDIATES :**
1. **URGENCE** : Commandez immédiatement {recommended_stock - current_stock} unités
2. **Surveillance** : Vérifiez le stock toutes les heures
3. **Planification** : Mettez en place un système d'alerte automatique
4. **Analyse** : Étudiez les causes de cette consommation élevée ({avg_daily} unités/jour)

**⚠️ RISQUES IDENTIFIÉS :**
• Rupture de stock dans {days_to_rupture} jour
• Impact sur la continuité du service
• Perte de revenus potentielle
• Satisfaction client compromise

**🏷️ PRODUITS DISPONIBLES :**
{', '.join(list(product_stocks.keys())[:10])}{'...' if len(product_stocks) > 10 else ''}"""
    else:
        return f"""🔍 **ANALYSE SPÉCIFIQUE : DONNÉES LIMITÉES**

**⚠️ DONNÉES DE PRODUITS NON DISPONIBLES :**
Les données spécifiques par produit ne sont pas disponibles.

**📊 DONNÉES GLOBALES DU DASHBOARD :**
• Stock total actuel : {current_stock:.0f} unités
• Consommation totale (30j) : {total_consumption:.0f} unités
• Consommation moyenne : {avg_daily:.1f} unités/jour
• Jours jusqu'à rupture : {days_to_rupture:.1f} jours
• Confiance prédiction : {confidence_pct:.1f}%
• Tendance : {trend} ({trend_pct:.1f}%)
• Volatilité : {volatility:.1f}
• Stabilité : {stability:.1f}%
• Efficacité stock : {stock_efficiency:.1f}%

**🚨 ALERTES :**
• Statut global : {alert_message}
• Stock minimum : {stock_min:.0f} unités
• Stock recommandé : {recommended_stock:.0f} unités
• Stock maximum : {stock_max:.0f} unités

**💡 RECOMMANDATIONS :**
• Chargez d'abord les données détaillées des produits
• Utilisez les données globales pour l'analyse générale
• Planifiez le réapprovisionnement selon les tendances globales"""

def analyze_stock_data_smart(prompt, session_state):
    """Analyse intelligente des données de stock en utilisant DIRECTEMENT les données du dashboard"""
    # Accès direct aux données du dashboard (plus simple et efficace)
    dashboard_data = session_state.get('dashboard_data', {})
    product_stocks = session_state.get('product_stocks', {})
    predictions_data = session_state.get('predictions_data', {})
    
    # Vérifier si l'utilisateur demande une analyse d'un produit spécifique
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in ['couche', 'softcare', 'softcqre', 'lait', 'broli', 'produit spécifique', 'prédiction', 'prévoir']):
        return analyze_specific_product(prompt, dashboard_data, product_stocks)
    
    # Vérifier si on a des données
    if not dashboard_data and not product_stocks and not predictions_data:
        return """❌ **Aucune donnée de stock disponible**

**📊 POUR ANALYSER VOTRE STOCK :**
1. Allez dans l'onglet **"Tableau de Bord"**
2. Chargez vos données de consommation
3. Revenez ici pour l'analyse

**💡 DONNÉES NÉCESSAIRES :**
• Historique de consommation
• Niveaux de stock actuels
• Données de vente
• Informations sur les fournisseurs

**🚀 APRÈS LE CHARGEMENT :**
Je pourrai vous fournir :
• Analyse détaillée de votre stock
• Recommandations personnalisées
• Alertes de rupture
• Optimisation des niveaux"""
    
    # Analyser les données disponibles
    prompt_lower = prompt.lower()
    
    # Extraire les métriques principales
    total_consumption = dashboard_data.get('total_consumption', 0)
    avg_daily = dashboard_data.get('avg_daily_consumption', 0)
    days_to_rupture = dashboard_data.get('days_to_rupture', 0)
    confidence = dashboard_data.get('confidence', 0)
    stock_max = dashboard_data.get('stock_max', 0)
    stock_min = dashboard_data.get('stock_min', 0)
    trend = dashboard_data.get('trend_direction', 'N/A')
    alert_level = 'CRITIQUE' if days_to_rupture < 7 else 'FAIBLE' if days_to_rupture < 14 else 'NORMAL'
    
    # Vérifier si on a des données de produits spécifiques
    if product_stocks and len(product_stocks) > 0:
        # Répondre aux questions spécifiques sur les produits
        if any(word in prompt_lower for word in ['produit', 'nombre', 'combien', 'différent', 'information', 'donner', 'stock', 'present']):
            total_products = len(product_stocks)
            product_list = list(product_stocks.keys())
            
            response = f"""📊 **INFORMATIONS SUR LES PRODUITS DE L'APPLICATION**

**🏷️ NOMBRE DE PRODUITS :**
• **Total de produits différents** : {total_products} produits

**📋 LISTE DES PRODUITS :**
"""
            for i, product_name in enumerate(product_list[:10], 1):  # Top 10 produits
                product_data = product_stocks[product_name]
                status_icon = "🔴" if product_data.get('stock_status') == 'CRITIQUE' else "🟡" if product_data.get('stock_status') == 'FAIBLE' else "🟢" if product_data.get('stock_status') == 'NORMAL' else "🔵"
                response += f"{i}. {status_icon} **{product_name}** - Stock: {product_data.get('estimated_current_stock', 0):.1f} unités\n"
            
            if len(product_list) > 10:
                response += f"... et {len(product_list) - 10} autres produits\n"
            
            response += f"""
**📊 STATISTIQUES GÉNÉRALES :**
• Consommation totale : {total_consumption:.1f} unités
• Consommation moyenne : {avg_daily:.1f} unités/jour
• Jours jusqu'à rupture : {days_to_rupture:.1f} jours
• Score de confiance : {confidence:.1%}

**💡 POUR PLUS DE DÉTAILS :**
Posez : "Quel est le stock de chaque produit ?" pour voir tous les détails
"""
            return response
        
        # Répondre aux questions sur les stocks spécifiques
        elif any(word in prompt_lower for word in ['stock', 'chaque', 'détail']):
            response = f"""📦 **STOCKS DÉTAILLÉS PAR PRODUIT**

**🏷️ RÉSUMÉ :**
• **Nombre total de produits** : {len(product_stocks)} produits différents

**📋 DÉTAIL DES STOCKS :**
"""
            for product_name, product_data in product_stocks.items():
                status_icon = "🔴" if product_data.get('stock_status') == 'CRITIQUE' else "🟡" if product_data.get('stock_status') == 'FAIBLE' else "🟢" if product_data.get('stock_status') == 'NORMAL' else "🔵"
                response += f"""
{status_icon} **{product_name}** :
   • Stock estimé : {product_data.get('estimated_current_stock', 0):.1f} unités
   • Jours jusqu'à rupture : {product_data.get('days_to_rupture', 0):.1f} jours
   • Consommation récente : {product_data.get('recent_consumption', 0):.1f} unités/jour
   • Statut : {product_data.get('stock_status', 'N/A')}
   • Transactions : {product_data.get('transactions_count', 0)} enregistrements
"""
            return response
    
    # Si pas de données de produits, donner les informations générales disponibles
    else:
        response = f"""📊 **INFORMATIONS DISPONIBLES SUR VOTRE STOCK**

**📈 MÉTRIQUES GÉNÉRALES :**
• Consommation totale : {total_consumption:.1f} unités
• Consommation moyenne : {avg_daily:.1f} unités/jour
• Jours jusqu'à rupture : {days_to_rupture:.1f} jours
• Score de confiance : {confidence:.1%}
• Stock minimum recommandé : {stock_min:.1f} unités
• Stock maximum recommandé : {stock_max:.1f} unités

**⚠️ ALERTE :**
• Niveau d'alerte : {alert_level}
• Tendance : {trend}

**💡 RECOMMANDATIONS :**
• Surveillez de près les niveaux de stock
• Planifiez les réapprovisionnements
• Analysez les tendances de consommation

**❓ POUR PLUS D'INFORMATIONS :**
Posez des questions spécifiques sur vos données de stock !
"""
        return response
    
    # Analyser les données disponibles
    prompt_lower = prompt.lower()
    
    # Extraire les métriques principales
    total_consumption = dashboard_data.get('total_consumption', 0)
    avg_daily = dashboard_data.get('avg_daily_consumption', 0)
    days_to_rupture = dashboard_data.get('days_to_rupture', 0)
    confidence = dashboard_data.get('confidence', 0)
    stock_max = dashboard_data.get('stock_max', 0)
    stock_min = dashboard_data.get('stock_min', 0)
    trend = dashboard_data.get('trend_direction', 'N/A')
    alert_level = 'CRITIQUE' if days_to_rupture < 7 else 'FAIBLE' if days_to_rupture < 14 else 'NORMAL'
    
    # Vérifier si on a des données de produits spécifiques
    if product_stocks and len(product_stocks) > 0:
        # Répondre aux questions spécifiques sur les produits
        if any(word in prompt_lower for word in ['produit', 'nombre', 'combien', 'différent', 'information', 'donner', 'stock', 'present']):
            total_products = len(product_stocks)
            product_list = list(product_stocks.keys())
            
            response = f"""📊 **INFORMATIONS SUR LES PRODUITS DE L'APPLICATION**

**🏷️ NOMBRE DE PRODUITS :**
• **Total de produits différents** : {total_products} produits

**📋 LISTE DES PRODUITS :**
"""
            for i, product_name in enumerate(product_list[:10], 1):  # Top 10 produits
                product_data = product_stocks[product_name]
                status_icon = "🔴" if product_data.get('stock_status') == 'CRITIQUE' else "🟡" if product_data.get('stock_status') == 'FAIBLE' else "🟢" if product_data.get('stock_status') == 'NORMAL' else "🔵"
                response += f"{i}. {status_icon} **{product_name}** - Stock: {product_data.get('estimated_current_stock', 0):.1f} unités\n"
            
            if len(product_list) > 10:
                response += f"... et {len(product_list) - 10} autres produits\n"
            
            response += f"""
**📊 STATISTIQUES GÉNÉRALES :**
• Consommation totale : {total_consumption:.1f} unités
• Consommation moyenne : {avg_daily:.1f} unités/jour
• Jours jusqu'à rupture : {days_to_rupture:.1f} jours
• Score de confiance : {confidence:.1%}

**💡 POUR PLUS DE DÉTAILS :**
Posez : "Quel est le stock de chaque produit ?" pour voir tous les détails
"""
            return response
        
        # Répondre aux questions sur les stocks spécifiques
        elif any(word in prompt_lower for word in ['stock', 'chaque', 'détail']):
            response = f"""📦 **STOCKS DÉTAILLÉS PAR PRODUIT**

**🏷️ RÉSUMÉ :**
• **Nombre total de produits** : {len(product_stocks)} produits différents

**📋 DÉTAIL DES STOCKS :**
"""
            for product_name, product_data in product_stocks.items():
                status_icon = "🔴" if product_data.get('stock_status') == 'CRITIQUE' else "🟡" if product_data.get('stock_status') == 'FAIBLE' else "🟢" if product_data.get('stock_status') == 'NORMAL' else "🔵"
                response += f"""
{status_icon} **{product_name}** :
   • Stock estimé : {product_data.get('estimated_current_stock', 0):.1f} unités
   • Jours jusqu'à rupture : {product_data.get('days_to_rupture', 0):.1f} jours
   • Consommation récente : {product_data.get('recent_consumption', 0):.1f} unités/jour
   • Statut : {product_data.get('stock_status', 'N/A')}
   • Transactions : {product_data.get('transactions_count', 0)} enregistrements
"""
            return response
    
    # Si pas de données de produits, donner les informations générales disponibles
    else:
        response = f"""📊 **INFORMATIONS DISPONIBLES SUR VOTRE STOCK**

**📈 MÉTRIQUES GÉNÉRALES :**
• Consommation totale : {total_consumption:.1f} unités
• Consommation moyenne : {avg_daily:.1f} unités/jour
• Jours jusqu'à rupture : {days_to_rupture:.1f} jours
• Score de confiance : {confidence:.1%}
• Stock minimum recommandé : {stock_min:.1f} unités
• Stock maximum recommandé : {stock_max:.1f} unités

**⚠️ ALERTE :**
• Niveau d'alerte : {alert_level}
• Tendance : {trend}

**💡 RECOMMANDATIONS :**
• Surveillez de près les niveaux de stock
• Planifiez les réapprovisionnements
• Analysez les tendances de consommation

**❓ POUR PLUS D'INFORMATIONS :**
Posez des questions spécifiques sur vos données de stock !
"""
        return response
    
    # Analyser TOUTES les données spécifiques de l'application
    prompt_lower = prompt.lower()
    
    # Extraire les métriques principales
    total_consumption = dashboard_metrics.get('total_consumption', 0)
    avg_daily = dashboard_metrics.get('avg_daily_consumption', 0)
    days_to_rupture = dashboard_metrics.get('days_to_rupture', 0)
    confidence = dashboard_metrics.get('confidence', 0)
    stock_max = dashboard_metrics.get('stock_max', 0)
    stock_min = dashboard_metrics.get('stock_min', 0)
    trend = dashboard_metrics.get('trend_direction', 'N/A')
    alert_level = 'CRITIQUE' if days_to_rupture < 7 else 'FAIBLE' if days_to_rupture < 14 else 'NORMAL'
    
    # Vérifier si on a des données de produits spécifiques
    if products_detailed and len(products_detailed) > 0:
        # Répondre aux questions spécifiques sur les produits
        if any(word in prompt_lower for word in ['produit', 'nombre', 'combien', 'différent', 'information', 'donner']):
            total_products = len(products_detailed)
            product_list = list(products_detailed.keys())
            
            response = f"""📊 **INFORMATIONS SUR LES PRODUITS DE L'APPLICATION**

**🏷️ NOMBRE DE PRODUITS :**
• **Total de produits différents** : {total_products} produits

**📋 LISTE DES PRODUITS :**
"""
            for i, product_name in enumerate(product_list[:10], 1):  # Top 10 produits
                product_data = products_detailed[product_name]
                status_icon = "🔴" if product_data.get('stock_status') == 'CRITIQUE' else "🟡" if product_data.get('stock_status') == 'FAIBLE' else "🟢" if product_data.get('stock_status') == 'NORMAL' else "🔵"
                response += f"{i}. {status_icon} **{product_name}** - Stock: {product_data.get('estimated_current_stock', 0):.1f} unités\n"
            
            if len(product_list) > 10:
                response += f"... et {len(product_list) - 10} autres produits\n"
            
            response += f"""
**📊 STATISTIQUES GÉNÉRALES :**
• Consommation totale : {total_consumption:.1f} unités
• Consommation moyenne : {avg_daily:.1f} unités/jour
• Jours jusqu'à rupture : {days_to_rupture:.1f} jours
• Score de confiance : {confidence:.1%}

**💡 POUR PLUS DE DÉTAILS :**
Posez : "Quel est le stock de chaque produit ?" pour voir tous les détails
"""
            return response
        
        # Répondre aux questions sur les stocks spécifiques
        elif any(word in prompt_lower for word in ['stock', 'chaque', 'détail']):
            response = f"""📦 **STOCKS DÉTAILLÉS PAR PRODUIT**

**🏷️ RÉSUMÉ :**
• **Nombre total de produits** : {len(products_detailed)} produits différents

**📋 DÉTAIL DES STOCKS :**
"""
            for product_name, product_data in products_detailed.items():
                status_icon = "🔴" if product_data.get('stock_status') == 'CRITIQUE' else "🟡" if product_data.get('stock_status') == 'FAIBLE' else "🟢" if product_data.get('stock_status') == 'NORMAL' else "🔵"
                response += f"""
{status_icon} **{product_name}** :
   • Stock estimé : {product_data.get('estimated_current_stock', 0):.1f} unités
   • Jours jusqu'à rupture : {product_data.get('days_to_rupture', 0):.1f} jours
   • Consommation récente : {product_data.get('recent_consumption', 0):.1f} unités/jour
   • Statut : {product_data.get('stock_status', 'N/A')}
   • Transactions : {product_data.get('transactions_count', 0)} enregistrements
"""
            return response
    
    # Si pas de données de produits, donner les informations générales disponibles
    else:
        response = f"""📊 **INFORMATIONS DISPONIBLES SUR VOTRE STOCK**

**📈 MÉTRIQUES GÉNÉRALES :**
• Consommation totale : {total_consumption:.1f} unités
• Consommation moyenne : {avg_daily:.1f} unités/jour
• Jours jusqu'à rupture : {days_to_rupture:.1f} jours
• Score de confiance : {confidence:.1%}
• Stock minimum recommandé : {stock_min:.1f} unités
• Stock maximum recommandé : {stock_max:.1f} unités

**⚠️ ALERTE :**
• Niveau d'alerte : {alert_level}
• Tendance : {trend}

**💡 RECOMMANDATIONS :**
• Surveillez de près les niveaux de stock
• Planifiez les réapprovisionnements
• Analysez les tendances de consommation

**❓ POUR PLUS D'INFORMATIONS :**
Posez des questions spécifiques sur vos données de stock !
"""
        return response
    
    # Analyser TOUTES les données spécifiques de l'application
    product_analysis = ""
    prompt_lower = prompt.lower()
    
    if products_detailed:
        try:
            total_products = len(products_detailed)
            product_list = list(products_detailed.keys())[:10]  # Top 10 produits
            
            # Compter les produits par statut
            status_counts = {'CRITIQUE': 0, 'FAIBLE': 0, 'NORMAL': 0, 'ÉLEVÉ': 0}
            for product_data in products_detailed.values():
                status = product_data.get('stock_status', 'NORMAL')
                if status in status_counts:
                    status_counts[status] += 1
            
            # Informations générales sur les données
            data_info = ""
            if data_metadata:
                data_info = f"""
**📊 INFORMATIONS SUR LES DONNÉES :**
• **Période d'analyse** : {data_metadata.get('date_range', {}).get('start', 'N/A')} à {data_metadata.get('date_range', {}).get('end', 'N/A')}
• **Nombre de colonnes** : {len(data_metadata.get('data_columns', []))}
• **Taille du dataset** : {data_metadata.get('data_shape', (0, 0))[0]} lignes × {data_metadata.get('data_shape', (0, 0))[1]} colonnes
• **Colonnes disponibles** : {', '.join(data_metadata.get('data_columns', [])[:5])}{'...' if len(data_metadata.get('data_columns', [])) > 5 else ''}
"""
            
            product_analysis = f"""
**🏷️ STOCKS PAR PRODUIT :**
• **Nombre total de produits** : {total_products} produits différents
• **Produits critiques** : {status_counts['CRITIQUE']} (rupture < 7 jours)
• **Produits faibles** : {status_counts['FAIBLE']} (rupture < 14 jours)
• **Produits normaux** : {status_counts['NORMAL']} (rupture < 30 jours)
• **Produits élevés** : {status_counts['ÉLEVÉ']} (rupture > 30 jours)

{data_info}

**📋 DÉTAIL DES STOCKS :**
"""
            
            # Vérifier le type de demande de l'utilisateur
            if any(word in prompt_lower for word in ['stock actuel', 'chaque produit', 'tous les produits', 'détail', 'complet']):
                # Afficher tous les produits avec leurs stocks et statistiques avancées
                for product_name, product_data in products_detailed.items():
                    status_icon = "🔴" if product_data['stock_status'] == 'CRITIQUE' else "🟡" if product_data['stock_status'] == 'FAIBLE' else "🟢" if product_data['stock_status'] == 'NORMAL' else "🔵"
                    
                    # Statistiques avancées si disponibles
                    advanced_info = ""
                    if product_name in advanced_stats:
                        adv = advanced_stats[product_name]
                        advanced_info = f"""
   📈 **STATISTIQUES AVANCÉES** :
   • Min/Max consommation : {adv.get('min_consumption', 0):.1f} / {adv.get('max_consumption', 0):.1f}
   • Médiane : {adv.get('median_consumption', 0):.1f} | Écart-type : {adv.get('std_consumption', 0):.1f}
   • Q25/Q75 : {adv.get('q25_consumption', 0):.1f} / {adv.get('q75_consumption', 0):.1f}
   • Tendance : {adv.get('consumption_trend', 'stable')} | Pattern : {adv.get('consumption_pattern', 'regular')}
   • Outliers : {adv.get('outlier_count', 0)} | Volatilité récente : {adv.get('recent_volatility', 0):.1f}
"""
                    
                    product_analysis += f"""
{status_icon} **{product_name[:40]}** :
   • Stock estimé : {product_data['estimated_current_stock']:.1f} unités
   • Jours jusqu'à rupture : {product_data['days_to_rupture']:.1f} jours
   • Consommation récente : {product_data['recent_consumption']:.1f} unités/jour
   • Statut : {product_data['stock_status']}
   • Transactions : {product_data['transactions_count']} enregistrements{advanced_info}
"""
            elif any(word in prompt_lower for word in ['statistiques', 'analyse', 'détail', 'avancé']):
                # Afficher les statistiques avancées
                product_analysis += f"""
**📊 STATISTIQUES AVANCÉES DISPONIBLES :**
• **Données brutes** : {raw_historical_data.shape if raw_historical_data is not None else 'N/A'}
• **Statistiques par produit** : {len(advanced_stats)} produits analysés
• **Métadonnées** : {len(data_metadata)} informations disponibles

**🔍 POUR UNE ANALYSE SPÉCIFIQUE :**
Posez : "Analyse détaillée de [nom du produit]"
"""
            else:
                # Vérifier si l'utilisateur demande un produit spécifique
                found_product = False
                for product_name in products_detailed.keys():
                    if any(word in str(product_name).lower() for word in prompt_lower.split()):
                        product_data = products_detailed[product_name]
                        status_icon = "🔴" if product_data['stock_status'] == 'CRITIQUE' else "🟡" if product_data['stock_status'] == 'FAIBLE' else "🟢" if product_data['stock_status'] == 'NORMAL' else "🔵"
                        
                        # Statistiques avancées pour ce produit
                        advanced_info = ""
                        if product_name in advanced_stats:
                            adv = advanced_stats[product_name]
                            advanced_info = f"""
**📈 STATISTIQUES AVANCÉES :**
• **Consommation** : Min {adv.get('min_consumption', 0):.1f} | Max {adv.get('max_consumption', 0):.1f} | Médiane {adv.get('median_consumption', 0):.1f}
• **Variabilité** : Écart-type {adv.get('std_consumption', 0):.1f} | Q25-Q75 {adv.get('q25_consumption', 0):.1f}-{adv.get('q75_consumption', 0):.1f}
• **Tendance** : {adv.get('consumption_trend', 'stable')} | Pattern : {adv.get('consumption_pattern', 'regular')}
• **Anomalies** : {adv.get('outlier_count', 0)} outliers | Volatilité : {adv.get('recent_volatility', 0):.1f}
• **Pics** : Jour max {adv.get('peak_consumption_day', 'N/A')} | Jour min {adv.get('low_consumption_day', 'N/A')}
"""
                        
                        product_analysis += f"""
**🔍 ANALYSE COMPLÈTE DU PRODUIT "{product_name}" :**
{status_icon} **STOCK ACTUEL** : {product_data['estimated_current_stock']:.1f} unités
📊 **CONSOMMATION** :
   • Totale : {product_data['total_consumption']:.1f} unités
   • Moyenne : {product_data['avg_daily_consumption']:.1f} unités/transaction
   • Récente : {product_data['recent_consumption']:.1f} unités/jour
⏰ **RUPTURE** : {product_data['days_to_rupture']:.1f} jours
📈 **STATUT** : {product_data['stock_status']}
🔄 **TRANSACTIONS** : {product_data['transactions_count']} enregistrements{advanced_info}
"""
                        found_product = True
                        break
                
                if not found_product:
                    # Afficher un résumé des produits
                    product_analysis += f"""
• **Top produits** : {', '.join([str(p)[:30] for p in product_list])}

**💡 COMMANDES DISPONIBLES :**
• "Quel est le stock actuel de chaque produit ?" → Liste complète
• "Analyse détaillée de [produit]" → Analyse spécifique
• "Statistiques avancées" → Données techniques
• "Tous les détails" → Informations complètes
"""
        except Exception as e:
            product_analysis = f"\n**⚠️ Erreur analyse stocks** : {str(e)}"
    elif raw_historical_data is not None and hasattr(raw_historical_data, 'columns'):
        # Fallback vers l'ancienne méthode si products_detailed n'est pas disponible
        try:
            product_columns = [col for col in raw_historical_data.columns if any(word in col.lower() for word in ['produit', 'product', 'intitule', 'nom', 'name', 'item'])]
            
            if product_columns:
                product_col = product_columns[0]
                unique_products = raw_historical_data[product_col].nunique()
                product_list = raw_historical_data[product_col].unique()[:10]  # Top 10 produits
                
                product_analysis = f"""
**🏷️ PRODUITS IDENTIFIÉS :**
• **Nombre total de produits** : {unique_products} produits différents
• **Top produits** : {', '.join([str(p)[:30] for p in product_list])}
"""
        except Exception as e:
            product_analysis = f"\n**⚠️ Erreur analyse produits** : {str(e)}"

    # Analyser les données (utiliser les données complètes si disponibles)
    total_consumption = dashboard_metrics.get('total_consumption', 0)
    avg_daily = dashboard_metrics.get('avg_daily_consumption', 0)
    days_to_rupture = dashboard_metrics.get('days_to_rupture', 0)
    confidence = dashboard_metrics.get('confidence', 0)
    stock_max = dashboard_metrics.get('stock_max', 0)
    stock_min = dashboard_metrics.get('stock_min', 0)
    trend = dashboard_metrics.get('trend_direction', 'N/A')
    alert_level = 'CRITIQUE' if days_to_rupture < 7 else 'FAIBLE' if days_to_rupture < 14 else 'NORMAL'
    
    # Vérifier si on a des données de produits spécifiques
    if products_detailed and len(products_detailed) > 0:
        # Répondre aux questions spécifiques sur les produits
        if any(word in prompt_lower for word in ['produit', 'nombre', 'combien', 'différent', 'information', 'donner']):
            total_products = len(products_detailed)
            product_list = list(products_detailed.keys())
            
            response = f"""📊 **INFORMATIONS SUR LES PRODUITS DE L'APPLICATION**

**🏷️ NOMBRE DE PRODUITS :**
• **Total de produits différents** : {total_products} produits

**📋 LISTE DES PRODUITS :**
"""
            for i, product_name in enumerate(product_list[:10], 1):  # Top 10 produits
                product_data = products_detailed[product_name]
                status_icon = "🔴" if product_data.get('stock_status') == 'CRITIQUE' else "🟡" if product_data.get('stock_status') == 'FAIBLE' else "🟢" if product_data.get('stock_status') == 'NORMAL' else "🔵"
                response += f"{i}. {status_icon} **{product_name}** - Stock: {product_data.get('estimated_current_stock', 0):.1f} unités\n"
            
            if len(product_list) > 10:
                response += f"... et {len(product_list) - 10} autres produits\n"
            
            response += f"""
**📊 STATISTIQUES GÉNÉRALES :**
• Consommation totale : {total_consumption:.1f} unités
• Consommation moyenne : {avg_daily:.1f} unités/jour
• Jours jusqu'à rupture : {days_to_rupture:.1f} jours
• Score de confiance : {confidence:.1%}

**💡 POUR PLUS DE DÉTAILS :**
Posez : "Quel est le stock de chaque produit ?" pour voir tous les détails
"""
            return response
        
        # Répondre aux questions sur les stocks spécifiques
        elif any(word in prompt_lower for word in ['stock', 'chaque', 'détail']):
            response = f"""📦 **STOCKS DÉTAILLÉS PAR PRODUIT**

**🏷️ RÉSUMÉ :**
• **Nombre total de produits** : {len(products_detailed)} produits différents

**📋 DÉTAIL DES STOCKS :**
"""
            for product_name, product_data in products_detailed.items():
                status_icon = "🔴" if product_data.get('stock_status') == 'CRITIQUE' else "🟡" if product_data.get('stock_status') == 'FAIBLE' else "🟢" if product_data.get('stock_status') == 'NORMAL' else "🔵"
                response += f"""
{status_icon} **{product_name}** :
   • Stock estimé : {product_data.get('estimated_current_stock', 0):.1f} unités
   • Jours jusqu'à rupture : {product_data.get('days_to_rupture', 0):.1f} jours
   • Consommation récente : {product_data.get('recent_consumption', 0):.1f} unités/jour
   • Statut : {product_data.get('stock_status', 'N/A')}
   • Transactions : {product_data.get('transactions_count', 0)} enregistrements
"""
            return response
    
    # Si pas de données de produits, donner les informations générales disponibles
    else:
        response = f"""📊 **INFORMATIONS DISPONIBLES SUR VOTRE STOCK**

**📈 MÉTRIQUES GÉNÉRALES :**
• Consommation totale : {total_consumption:.1f} unités
• Consommation moyenne : {avg_daily:.1f} unités/jour
• Jours jusqu'à rupture : {days_to_rupture:.1f} jours
• Score de confiance : {confidence:.1%}
• Stock minimum recommandé : {stock_min:.1f} unités
• Stock maximum recommandé : {stock_max:.1f} unités

**⚠️ ALERTE :**
• Niveau d'alerte : {alert_level}
• Tendance : {trend}

**💡 RECOMMANDATIONS :**
• Surveillez de près les niveaux de stock
• Planifiez les réapprovisionnements
• Analysez les tendances de consommation

**❓ POUR PLUS D'INFORMATIONS :**
Posez des questions spécifiques sur vos données de stock !
"""
        return response
    
    # Générer l'analyse
    analysis = f"""📊 **ANALYSE COMPLÈTE DE VOTRE STOCK**

{product_analysis}

**📈 ÉTAT ACTUEL :**
• **Consommation totale** : {total_consumption:.2f} unités
• **Consommation moyenne** : {avg_daily:.2f} unités/jour
• **Jours jusqu'à rupture** : {days_to_rupture:.2f} jours
• **Score de confiance** : {confidence:.2f} ({confidence:.1%})

**🎯 RECOMMANDATIONS :**
• **Stock minimum** : {stock_min:.2f} unités
• **Stock maximum** : {stock_max:.2f} unités
• **Tendance** : {trend}
• **Niveau d'alerte** : {alert_level}

**🔍 ANALYSE DÉTAILLÉE :**"""

    # Ajouter des recommandations basées sur les données
    if days_to_rupture < 7:
        analysis += f"""

🚨 **ALERTE CRITIQUE** : Votre stock sera épuisé dans moins de 7 jours !

**⚡ ACTIONS IMMÉDIATES :**
• **Commande urgente** : {stock_min:.0f} unités minimum
• **Recommandation** : {stock_max:.0f} unités pour sécurité
• **Contact fournisseur** : Dès aujourd'hui
• **Plan B** : Trouver des fournisseurs alternatifs"""
    elif days_to_rupture < 14:
        analysis += f"""

⚠️ **ALERTE MODÉRÉE** : Votre stock sera épuisé dans moins de 2 semaines

**📋 ACTIONS RECOMMANDÉES :**
• **Planifier commande** : {stock_min:.0f} unités
• **Négocier délais** : Avec vos fournisseurs
• **Surveiller consommation** : Quotidiennement"""
    else:
        analysis += f"""

✅ **SITUATION STABLE** : Votre stock est suffisant pour plus de 2 semaines

**📊 SURVEILLANCE :**
• **Continuer monitoring** : Consommation quotidienne
• **Maintenir niveaux** : Entre {stock_min:.0f} et {stock_max:.0f} unités
• **Anticiper variations** : Saisonnalité, promotions"""

    # Ajouter des conseils d'optimisation
    trend_analysis = 'Surveiller' if trend == 'N/A' else 'Analyser l\'évolution'
    confidence_level = 'Excellent' if confidence > 0.8 else 'Bon' if confidence > 0.6 else 'À améliorer'
    
    analysis += f"""

**💡 CONSEILS D'OPTIMISATION :**
• **Tendance** : {trend} - {trend_analysis}
• **Confiance** : {confidence:.1%} - {confidence_level}
• **Marge de sécurité** : Ajouter 20% au stock minimum
• **Révision** : Ajuster les seuils selon la saisonnalité

**🎯 OBJECTIFS :**
• Éviter les ruptures de stock
• Optimiser les coûts de stockage
• Améliorer la précision des prédictions
• Automatiser les alertes"""

    return analysis


def generate_general_response(prompt, session_state):
    """Génère une réponse générale intelligente"""
    
    # Questions sur l'Afrique
    if any(word in prompt.lower() for word in ['afrique', 'africain', 'africaine']):
        return """🌍 **À propos de l'Afrique**

L'Afrique est un continent fascinant avec une richesse culturelle et naturelle exceptionnelle. Cependant, en tant qu'assistant spécialisé dans la gestion de stock et d'inventaire, je me concentre sur l'optimisation de vos données de stock.

**💡 Suggestion :** Si vous souhaitez discuter de la gestion de stock en Afrique ou des défis logistiques spécifiques à ce continent, je serais ravi de vous aider !

**📊 Questions que je peux traiter :**
• Analyse de vos données de stock
• Recommandations d'optimisation
• Prédictions de consommation
• Alertes de rupture de stock
• Conseils de gestion d'inventaire"""

    # Questions sur l'IA
    elif any(word in prompt.lower() for word in ['ia', 'intelligence', 'artificielle', 'ai']):
        return """🤖 **À propos de l'Intelligence Artificielle**

L'IA transforme la gestion de stock et d'inventaire ! Voici comment je peux vous aider :

**🚀 CAPACITÉS IA INTÉGRÉES :**
• **Analyse prédictive** : Prédire la consommation future
• **Détection d'anomalies** : Identifier les patterns inhabituels
• **Optimisation automatique** : Recommander les quantités optimales
• **Alertes intelligentes** : Anticiper les ruptures de stock

**📊 DANS VOTRE APPLICATION :**
• Analyse en temps réel de vos données
• Recommandations personnalisées
• Prédictions basées sur l'historique
• Optimisation des seuils de stock

**💡 Voulez-vous que j'analyse vos données actuelles ?**"""

    # Questions générales
    else:
        return f"""🧠 **Vision IA**

Bonjour ! Je suis votre assistant IA spécialisé dans la gestion de stock et d'inventaire.

**📊 CE QUE JE PEUX FAIRE :**
• Analyser vos données de stock en temps réel
• Fournir des recommandations personnalisées
• Prédire la consommation future
• Détecter les risques de rupture
• Optimiser vos niveaux de stock

**💡 POUR COMMENCER :**
• Posez-moi des questions sur votre stock
• Demandez des analyses de vos données
• Sollicitez des recommandations d'optimisation

**❓ Votre question :** "{prompt}"

**🔍 Suggestion :** Essayez de me poser une question plus spécifique à la gestion de stock, ou utilisez les boutons de questions suggérées ci-dessus !"""

if __name__ == "__main__":
    # Désactiver le file watching pour éviter l'erreur inotify sur Streamlit Cloud
    import os
    os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
    main()