# Importer streamlit et les fonctions de génération et d'évaluation
import streamlit as st
from src.util import generate_text, evaluate_similarity, gpt2, gpt2_tokenizer, gpt_neo, gpt_neo_tokenizer, \
    create_poem_prompt

st.set_page_config(layout="wide")

# Configuration de la page
st.title("Générateur de Poèmes avec GPT-2 et GPT-Neo")
st.subheader("Entrez un mot-clé en anglais pour générer un poème avec deux modèles différents.")

# Zone d'entrée pour le mot-clé
keyword = st.text_input("Mot-clé :", value="Life")

# Bouton pour générer le poème
if st.button("Générer les poèmes"):
    cols = st.columns(2)
    with st.spinner("Génération en cours..."):
        gpt2_poem = generate_text(
            gpt2,
            gpt2_tokenizer,
            create_poem_prompt(keyword, "gpt2")
        )
        gpt_neo_poem = generate_text(
            gpt_neo,
            gpt_neo_tokenizer,
            create_poem_prompt(keyword, "gpt_neo")
        )
        bleu, rouge = evaluate_similarity(gpt2_poem, gpt_neo_poem)

    # Afficher les résultats
    cols[0].write("### Poème généré par GPT-2 :")
    cols[0].write(gpt2_poem)

    cols[1].write("### Poème généré par GPT-Neo :")
    cols[1].write(gpt_neo_poem)

    st.write("### Scores de similarité :")
    st.write(f"- **Score BLEU** : {bleu:.4f}")
    st.write("- **Scores ROUGE** :")
    st.json(rouge)
