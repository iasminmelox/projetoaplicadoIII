# projetoaplicadoIII
Neste repositório contém atividade para projeto aplicado III



# recomendador_paris_adaptado.py
import pandas as pd
import numpy as np
import unicodedata
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from surprise import Dataset, Reader, SVD

# ---------- utils ----------
def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8')
    return s

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ---------- 1) Carregamento e pré-processamento ----------
def load_and_prepare(rotas_path, hotels_path):
    # Ajuste de encoding conforme seus arquivos
    rotas = pd.read_csv(rotas_path, encoding='utf-8')
    hotels = pd.read_csv(hotels_path, encoding='latin-1')

    # Normalizar colunas textuais relevantes
    for c in ['Route', 'Airline Name', 'Review', 'Review_Title']:
        if c in rotas.columns:
            rotas[c + '_norm'] = rotas[c].apply(normalize_text)

    for c in ['Hotel', 'Location', 'Country', 'Region', 'Theme', 'Company']:
        if c in hotels.columns:
            hotels[c + '_norm'] = hotels[c].apply(normalize_text)

    # Filtrar hotéis da França
    hotels = hotels[hotels['Country'].str.lower().str.contains('france', na=False)].copy()

    # Garantir Score numérico nos hotéis
    if 'Score' in hotels.columns:
        hotels['Score'] = pd.to_numeric(hotels['Score'], errors='coerce').fillna(0)
    else:
        hotels['Score'] = 0.0

    # Filtrar voos que tenham destino Paris na coluna Route (case-insensitive)
    if 'Route' in rotas.columns:
        rotas['Route_norm'] = rotas['Route'].apply(normalize_text)
        rotas_paris = rotas[rotas['Route_norm'].str.contains('to paris', na=False)].copy()
    else:
        rotas_paris = rotas.copy()

    # Criar ids (se não existirem)
    if 'route_id' not in rotas_paris.columns:
        rotas_paris = rotas_paris.reset_index(drop=True)
        rotas_paris['route_id'] = rotas_paris.index.astype(str)
    if 'hotel_id' not in hotels.columns:
        hotels = hotels.reset_index(drop=True)
        hotels['hotel_id'] = hotels.index.astype(str)

    return rotas_paris, hotels

# ---------- 2) Content-based (TF-IDF sobre Hotel+City+Theme) ----------
def build_content_model(hotels, text_cols=['Hotel_norm','Location_norm','Theme_norm']):
    hotels = hotels.copy()
    for col in text_cols:
        if col not in hotels.columns:
            hotels[col] = ''
    hotels['content'] = hotels[text_cols].fillna('').agg(' '.join, axis=1)
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
    tfidf = vectorizer.fit_transform(hotels['content'])
    # não precisamos da matriz hotel-hotel, só vamos comparar tokens simples com route text
    return vectorizer, tfidf

# ---------- 3) Gerar score de conteúdo entre rota -> hotel ----------
def score_content_route_to_hotels(rotas, hotels, vectorizer):
    # Para cada rota criamos um texto representativo: origem + airline + route description
    hotels = hotels.copy()
    # prepare hotel content tokens
    hotels['content_tokens'] = hotels['content'].apply(lambda x: set(str(x).split()))
    scores_dict = {}
    for idx, r in rotas.iterrows():
        parts = []
        for col in ['Airline Name', 'Route', 'Review_Title']:
            if col in rotas.columns:
                parts.append(str(r.get(col, '')))
        text = normalize_text(' '.join(parts))
        tokens = set(text.split())
        # score por intersecção simples com conteúdo do hotel
        score_vec = hotels['content_tokens'].apply(lambda s: len(tokens & s)).values.astype(float)
        # normalizar 0-1
        if score_vec.max() > score_vec.min():
            score_vec = (score_vec - score_vec.min())/(score_vec.max()-score_vec.min())
        else:
            score_vec = np.zeros_like(score_vec)
        scores_dict[str(r['route_id'])] = score_vec
    return scores_dict

# ---------- 4) Criar interações simuladas para CF (se não houver interações reais) ----------
def create_simulated_interactions(hotels, rotas, per_route=80):
    # usa Score do hotel como proxy para ratings e cria interações por route_id (tratamos cada rota como "usuário")
    hotels = hotels.copy()
    # escala Score para 1-5
    scaler = MinMaxScaler(feature_range=(1,5))
    hotels['score_scaled'] = scaler.fit_transform(hotels[['Score']].fillna(0))
    rows = []
    for r in rotas['route_id'].unique():
        # selecionar hotéis aleatoriamente mas ponderados por Score
        probs = hotels['Score'].values + 1e-6
        probs = probs / probs.sum()
        chosen_idx = np.random.choice(hotels.index, size=min(per_route, len(hotels)), replace=False, p=probs)
        for i in chosen_idx:
            rating = float(np.clip(hotels.loc[i,'score_scaled'] + np.random.normal(0,0.4), 1, 5))
            rows.append({'user_id': str(r), 'hotel_id': str(hotels.loc[i,'hotel_id']), 'rating': rating})
    interactions = pd.DataFrame(rows)
    return interactions

# ---------- 5) Treinar CF (SVD com surprise) ----------
def train_cf(interactions):
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(interactions[['user_id','hotel_id','rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=40, n_epochs=20, random_state=42)
    algo.fit(trainset)
    return algo

# ---------- 6) Combinar scores (conteúdo + CF) ----------
def combine_and_rank(rotas, hotels, content_scores_dict, cf_algo=None, alpha=0.6, top_n=20):
    hotels = hotels.copy()
    results = []
    for idx, r in rotas.iterrows():
        uid = str(r['route_id'])
        content_vec = content_scores_dict[uid]
        # CF predictions for this user (route)
        cf_vec = np.zeros(len(hotels), dtype=float)
        if cf_algo is not None:
            for i, hid in enumerate(hotels['hotel_id']):
                try:
                    pred = cf_algo.predict(uid, str(hid))
                    cf_vec[i] = pred.est
                except:
                    cf_vec[i] = np.nan
            # normalizar cf_vec para 0-1
            cf_vec = np.nan_to_num(cf_vec)
            if cf_vec.max() > cf_vec.min():
                cf_vec = (cf_vec - cf_vec.min())/(cf_vec.max() - cf_vec.min())
        # normalizar content
        if content_vec.max() > content_vec.min():
            cvec = (content_vec - content_vec.min())/(content_vec.max()-content_vec.min())
        else:
            cvec = np.zeros_like(content_vec)
        final = alpha * cvec + (1-alpha) * cf_vec
        hotels['final_score'] = final
        top = hotels.sort_values('final_score', ascending=False).head(top_n)
        for _, h in top.iterrows():
            results.append({
                'route_id': uid,
                'route': r.get('Route',''),
                'airline': r.get('Airline Name',''),
                'hotel_id': h['hotel_id'],
                'hotel_name': h['Hotel'],
                'hotel_city': h.get('Location',''),
                'hotel_score_original': h.get('Score',0),
                'final_score': h['final_score']
            })
    return pd.DataFrame(results)

# ---------- 7) Pipeline principal ----------
def pipeline(rotas_csv='data/rotas_paris.csv', hotels_csv='data/hotels.csv', out_dir='outputs',
             alpha=0.6, top_n=20, simulate_interactions_per_route=80):
    ensure_dir(out_dir)
    rotas, hotels = load_and_prepare(rotas_csv, hotels_csv)
    # build content model
    vectorizer, tfidf = build_content_model(hotels)
    hotels['content'] = hotels[['Hotel_norm','Location_norm','Theme_norm']].fillna('').agg(' '.join, axis=1)
    # score content route->hotels
    content_scores = score_content_route_to_hotels(rotas, hotels, vectorizer)
    # simular interacoes e treinar CF
    interactions = create_simulated_interactions(hotels, rotas, per_route=simulate_interactions_per_route)
    cf_algo = train_cf(interactions)
    # combinar e rankear
    recs = combine_and_rank(rotas, hotels, content_scores, cf_algo=cf_algo, alpha=alpha, top_n=top_n)
    out_path = os.path.join(out_dir, 'rotas_paris_x_recomendacoes.csv')
    recs.to_csv(out_path, index=False, encoding='utf-8')
    print("Recomendações salvas em:", out_path)
    return recs

# ---------- executa se chamado diretamente ----------
if __name__ == "__main__":
    # ajuste caminhos se necessário
    recs_df = pipeline(rotas_csv='data/rotas_paris.csv', hotels_csv='data/hotels.csv',
                       out_dir='outputs', alpha=0.6, top_n=20, simulate_interactions_per_route=80)
    print(recs_df.head())
