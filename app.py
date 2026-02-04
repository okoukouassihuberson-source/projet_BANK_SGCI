import streamlit as st
import pandas as pd
import joblib

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="SGCI - Scoring Risque", layout="wide", page_icon="🏦")

# --- STYLE CSS PERSONNALISÉ (INSPIRATION BOOTSTRAP & SGCI) ---
st.markdown("""
    <style>
    /* Importation d'une police plus pro */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }

    /* Couleurs SGCI */
    :root {
        --sg-red: #E2001A;
        --sg-black: #000000;
        --sg-gray: #F4F4F4;
    }

    .main { background-color: var(--sg-gray); }
    
    /* Style des cartes type Bootstrap */
    .st-card {
        background-color: white;
        padding: 25px;
        border-radius: 8px;
        border-left: 5px solid var(--sg-red);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: var(--sg-black);
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        z-index: 100;
    }
    
    /* Custom Headers */
    .header-text { color: var(--sg-black); font-weight: 700; border-bottom: 2px solid var(--sg-red); padding-bottom: 5px; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- CHARGEMENT DES DONNÉES ET MODÈLE ---
@st.cache_resource
def load_resources():
    model = joblib.load('model_sgci.pkl')
    le = joblib.load('encoder_sgci.pkl')
    scaler = joblib.load('scaler_sgci.pkl') # Assurez-vous d'avoir sauvegardé le scaler
    data = pd.read_excel('operations_bancaires_SGCI.xlsx')
    return model, le, scaler, data

try:
    model, le, scaler, data = load_resources()
except:
    st.error("⚠️ Erreur : Fichiers modèles (.pkl) introuvables. Lancez d'abord le script d'entraînement.")
    st.stop()

# --- HEADER AVEC LOGO ET TITRE ---
col_l, col_r = st.columns([1, 4])
with col_l:
    st.image("sgci.webp", width=120)
with col_r:
    st.markdown("<h1 style='color: #E2001A; margin-bottom: 0;'>SGCI - Risk Intelligence Suite</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: gray; font-size: 18px;'>Outil Décisionnel Crédit pour Particuliers & PME</p>", unsafe_allow_html=True)

st.markdown("---")

# --- BARRE LATÉRALE ---
st.sidebar.markdown("<h2 style='text-align: center; color: #E2001A;'>NAVIGATION</h2>", unsafe_allow_html=True)
client_id = st.sidebar.number_input("Entrez l'ID Client", min_value=1, step=1)
btn_analyser = st.sidebar.button("🚀 LANCER L'ANALYSE EXPERTE")

# --- LOGIQUE PRINCIPALE ---
if btn_analyser:
    client_row = data[data['ID_Client'] == client_id]
    
    if not client_row.empty:
        # Préparation technique (fidèle à votre logique)
        features = client_row.drop(columns=['ID_Client', 'Statut_Pret']).copy()
        features['Secteur_Activite'] = le.transform(features['Secteur_Activite'])
        features_scaled = scaler.transform(features)
        
        probabilite = model.predict_proba(features_scaled)[0][1]
        prediction = model.predict(features_scaled)[0]
        score_percent = round(probabilite * 100, 2)

        # AFFICHAGE STRUCTUREL
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="st-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="header-text">📋 Profil du Client</h3>', unsafe_allow_html=True)
            # Présentation propre des données client
            st.table(client_row.drop(columns=['Statut_Pret']).T.rename(columns={client_row.index[0]: 'Valeurs'}))
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="st-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="header-text">📊 Analyse du Risque</h3>', unsafe_allow_html=True)
            
            # Indicateur Visuel dynamique
            if prediction == 0:
                st.success(f"### ✅ DOSSIER APPROUVÉ")
                st.markdown(f"**Niveau de Risque :** {score_percent}% (Faible)")
                st.progress(probabilite)
                st.info("**Analyse Expert :** Le profil présente des garanties de solvabilité supérieures aux seuils de sécurité de la SGCI.")
            else:
                st.error(f"### ❌ PRÊT DÉCONSEILLÉ")
                st.markdown(f"**Niveau de Risque :** {score_percent}% (Élevé)")
                st.progress(probabilite)
                st.warning("**Analyse Expert :** Risque de défaut critique détecté. Le ratio d'endettement ou le score mobile money est hors limites.")
            
            # Ajout d'une explication technique pour les dirigeants
            st.markdown("---")
            st.markdown("#### Métriques Clés")
            m1, m2 = st.columns(2)
            m1.metric("Ratio Dette/Revenu", f"{client_row['Ratio_Dette_Revenu'].values[0]*100}%")
            m2.metric("Score Mobile Money", f"{client_row['Score_Mobile_Money'].values[0]}/100")
            st.markdown('</div>', unsafe_allow_html=True)
            
    else:
        st.error(f"⚠️ L'ID Client {client_id} est introuvable dans la base de données SGCI.")
else:
    # État d'attente pro
    st.markdown("""
        <div style="text-align: center; padding: 50px; border: 2px dashed #ccc; border-radius: 10px;">
            <h3 style="color: #666;">Prêt pour l'analyse</h3>
            <p>Veuillez saisir un identifiant client dans le panneau latéral pour obtenir le rapport de solvabilité.</p>
        </div>
    """, unsafe_allow_html=True)

# --- FOOTER PROFESSIONNEL ---
st.markdown(f"""
    <div class="footer">
        Analyste Financier : <b>Okou Kouassi Huberson</b> | Département Gestion des Risques SGCI | © 2026
    </div>
    """, unsafe_allow_html=True)
