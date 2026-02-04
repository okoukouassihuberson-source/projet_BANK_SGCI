import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Chargement
df = pd.read_excel('operations_bancaires_SGCI.xlsx')

# Prétraitement
le = LabelEncoder()
df['Secteur_Activite'] = le.fit_transform(df['Secteur_Activite'])

X = df.drop(columns=['ID_Client', 'Statut_Pret'])
y = df['Statut_Pret']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modèle
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Sauvegarde
joblib.dump(model, 'model_sgci.pkl')
joblib.dump(le, 'encoder_sgci.pkl')
joblib.dump(scaler, 'scaler_sgci.pkl')