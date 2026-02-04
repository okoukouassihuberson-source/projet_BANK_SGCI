import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURATION ET STYLE ---
st.set_page_config(page_title="SGCI Credit Scoring", layout="wide", page_icon="🏦")

# Simulation de Bootstrap / CSS Personnalisé
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #E2001A; color: white; border: none; }
    .stButton>button:hover { background-color: #b30015; color: white; }
    .footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #343a40; color: white; text-align: center; padding: 10px; font-style: italic; }
    .card { padding: 20px; border-radius: 10px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .result-box { padding: 25px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 24px; }
    </style>
    """, unsafe_allow_html=True)

# --- CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_assets():
    model = joblib.load('model_sgci.pkl')
    le = joblib.load('encoder_sgci.pkl')
    scaler = joblib.load('scaler_sgci.pkl')
    data = pd.read_excel('operations_bancaires_SGCI.xlsx')
    return model, le, scaler, data

model, le, scaler, data = load_assets()

# --- HEADER AVEC LOGO ---
col_logo, col_title = st.columns([1, 4])
with col_logo:
    # Utilisation d'une URL placeholder pour le logo SGCI (ou mettez votre fichier local)
    st.image("sgci.webp", width=500)
with col_title:
    st.title("Système Expert de Cotation du Risque Crédit")
    st.subheader("Société Générale Côte d'Ivoire (SGCI)")

st.markdown("---")

# --- CORPS DE L'APPLICATION ---
st.sidebar.header("🕹️ Panneau de Contrôle")
client_id = st.sidebar.number_input("Saisir l'ID Client (Base SGCI)", min_value=1, step=1)
analyze_btn = st.sidebar.button("Lancer l'Analyse")

if analyze_btn:
    client_info = data[data['ID_Client'] == client_id]
    
    if not client_info.empty:
        # Extraction et préparation
        features_raw = client_info.drop(columns=['ID_Client', 'Statut_Pret'])
        features_encoded = features_raw.copy()
        features_encoded['Secteur_Activite'] = le.transform(features_encoded['Secteur_Activite'])
        
        # Scaling et Prédiction
        features_scaled = scaler.transform(features_encoded)
        probabilite = model.predict_proba(features_scaled)[0][1]
        
        # Affichage des résultats
        col_res, col_data = st.columns([1, 1.5])
        
        with col_res:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("### 📊 Diagnostic du Risque")
            score_final = round(probabilite * 100, 2)
            
            if score_final < 30:
                st.markdown(f'<div class="result-box" style="background-color: #d4edda; color: #155724;">FAIBLE RISQUE <br> {score_final}%</div>', unsafe_allow_html=True)
                avis = "Approuvé sans réserve."
            elif score_final < 60:
                st.markdown(f'<div class="result-box" style="background-color: #fff3cd; color: #856404;">RISQUE MODÉRÉ <br> {score_final}%</div>', unsafe_allow_html=True)
                avis = "Nécessite des garanties additionnelles."
            else:
                st.markdown(f'<div class="result-box" style="background-color: #f8d7da; color: #721c24;">RISQUE ÉLEVÉ <br> {score_final}%</div>', unsafe_allow_html=True)
                avis = "Rejet automatique du dossier."
            # Gauge de risque simple
            st.progress(probabilite)
            st.write("Le score représente la probabilité que le client soit en défaut de paiement.")
            
            st.metric("Probabilité de Défaut", f"{score_final}%")
            st.markdown(f"**Décision suggérée :** {avis}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_data:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("### 🔍 Analyse Factorielle")
            
            # Analyse des points clés
            ratio = client_info['Ratio_Dette_Revenu'].values[0]
            mm_score = client_info['Score_Mobile_Money'].values[0]
            revenu = client_info['Revenu_Mensuel_FCFA'].values[0]
            
            st.write(f"**1. Capacité de remboursement :** Avec un ratio de {ratio*100:.1f}%, le client " + 
                     ("présente une charge de dette saine." if ratio < 0.4 else "est en situation de surendettement potentiel."))
            
            st.write(f"**2. Comportement Mobile Money :** Un score de {mm_score}/100 indique une " + 
                     ("excellente" if mm_score > 70 else "faible") + " fiabilité financière digitale.")
            
            st.write(f"**3. Stabilité professionnelle :** L'ancienneté de {client_info['Anciennete_Pro_Mois'].values[0]} mois " +
                     "est un facteur de réassurance pour la banque.")
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.error(f"❌ L'ID Client {client_id} n'existe pas dans les registres de la SGCI.")

else:
    st.info("👋 Bienvenue. Veuillez entrer un identifiant client dans le panneau de gauche pour générer un rapport de risque.")

# --- FOOTER ---
st.markdown(f"""
    <div class="footer">
        <p> Expertise Risque Crédit SGCI | Analyste financier : <b>Okou Kouassi Huberson</b> | © 2025</p>
    </div>
    """, unsafe_allow_html=True)